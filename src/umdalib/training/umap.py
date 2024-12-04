import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path as pt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from umdalib.logger import logger
from umdalib.utils.json import safe_json_dump
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


@dataclass
class Args:
    n_neighbors: int
    min_dist: float
    n_components: int
    umap_metric: str
    n_jobs: int
    scale_embedding: bool
    label_issues_file: Optional[str]
    processed_df_file: str
    columnX: str
    dbscan_eps: float
    dbscan_min_samples: int
    training_filename: str
    random_state: Optional[int]


umap_dir: pt = None


def main(args: Args):
    global umap_dir

    processed_df_file = pt(args.processed_df_file)
    umap_dir = processed_df_file.parent / "umap"
    umap_dir.mkdir(exist_ok=True)

    safe_json_dump(args.__dict__, umap_dir / "umap_args.json")

    # return
    df = pd.read_parquet(processed_df_file)

    if args.label_issues_file:
        label_issues_df = pd.read_parquet(args.label_issues_file)
        logger.info(f"Label issues: {label_issues_df.shape}")

        df = df[~label_issues_df["is_label_issue"]]

    smiles_list = df[args.columnX].to_list()
    logger.info(len(smiles_list))

    embeddings = df.iloc[:, 2:].to_numpy()
    y = df["y"].to_numpy()
    logger.info(f"X shape: {embeddings.shape}, y shape: {y.shape}")

    if args.scale_embedding:
        # Scale embeddings
        logger.info("Scaling embeddings...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    # Perform UMAP
    logger.info("Performing UMAP...")
    if args.random_state is not None:
        logger.info(f"Random state: {args.random_state}")
        args.n_jobs = 1

    reducer = UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    reduced_embeddings = reducer.fit_transform(embeddings)
    logger.info(reduced_embeddings.shape)

    logger.info("Analyzing chemical clusters...")

    analyzer = ChemicalClusterAnalyzer()
    labels, cluster_analysis = analyzer.analyze_cluster_chemistry(
        reduced_embeddings,
        smiles_list,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    )

    umap_df = pd.DataFrame(
        {
            df.index.name: df.index,
            "UMAP1": reduced_embeddings[:, 0],
            "UMAP2": reduced_embeddings[:, 1],
            "Cluster": labels,
            args.columnX: smiles_list,
            "y": y,
        }
    )

    umap_df_file = umap_dir / "umap_df.parquet"
    umap_df.to_parquet(umap_df_file)
    logger.success(f"UMAP embeddings saved to {umap_dir}")

    plot_figure_static(
        umap_df, args.training_filename, fig_size=(12, 8), point_size=50, alpha=0.6
    )

    cluster_analysis_df = pd.DataFrame(cluster_analysis).T
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        cluster_data: pd.DataFrame = umap_df[umap_df["Cluster"] == cluster_id]
        cluster_analysis_df.loc[cluster_id, "Mean"] = cluster_data["y"].mean()
        cluster_analysis_df.loc[cluster_id, "Std"] = cluster_data["y"].std()
        cluster_analysis_df.loc[cluster_id, "Min"] = cluster_data["y"].min()
        cluster_analysis_df.loc[cluster_id, "Max"] = cluster_data["y"].max()

    cluster_analysis_file = umap_dir / "cluster_analysis.parquet"
    cluster_analysis_df.to_parquet(cluster_analysis_file)
    logger.success(f"Cluster analysis saved to {umap_dir}")

    return {
        "cluster_analysis_file": cluster_analysis_file,
        "umap_df_file": umap_df_file,
    }


class ChemicalClusterAnalyzer:
    """Analyzes chemical structure clusters and their functional groups."""

    def __init__(self):
        # Define common functional groups and their SMARTS patterns
        self.functional_groups = {
            "Alcohol": "[OH]",
            "Carboxylic Acid": "[CX3](=O)[OX2H1]",
            "Ester": "[#6][CX3](=O)[OX2H0][#6]",
            "Ether": "[OD2]([#6])[#6]",
            "Aldehyde": "[CX3H1](=O)[#6]",
            "Ketone": "[#6][CX3](=O)[#6]",
            "Amine": "[NX3;H2,H1;!$(NC=O)]",
            "Amide": "[NX3][CX3](=[OX1])[#6]",
            "Aromatic": "a1aaaaa1",
            "Alkene": "[CX3]=[CX3]",
            "Alkyne": "[CX2]#[CX2]",
            "Nitrile": "[NX1]#[CX2]",
            "Nitro": "[NX3](=O)=O",
            "Sulfonic Acid": "[SX4](=[OX1])(=[OX1])[OX2H]",
            "Phosphate": "[PX4](=[OX1])([OX2H])([OX2H])[OX2H]",
            "Halogen": "[F,Cl,Br,I]",
        }

    def identify_functional_groups(self, smiles: str) -> Dict[str, int]:
        """
        Identifies functional groups in a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary of functional group counts
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            groups = {}
            for name, smarts in self.functional_groups.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    if matches:
                        groups[name] = len(matches)
            return groups
        except Exception:
            return {}

    def analyze_cluster_chemistry(
        self,
        reduced_embeddings: np.ndarray,
        smiles_list: List[str],
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Performs clustering and analyzes chemical features of each cluster.

        Args:
            reduced_embeddings: UMAP-reduced embeddings
            smiles_list: List of SMILES strings
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter

        Returns:
            Cluster labels and cluster analysis results
        """
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_embeddings)
        labels = clustering.labels_

        # Analyze each cluster
        cluster_analysis = {}
        for cluster_id in tqdm(set(labels)):
            if cluster_id == -1:  # Skip noise points
                continue

            # Get SMILES strings for this cluster
            cluster_mask = labels == cluster_id
            cluster_smiles = np.array(smiles_list)[cluster_mask]

            # Analyze functional groups
            all_groups = []
            for smiles in cluster_smiles:
                groups = self.identify_functional_groups(smiles)
                all_groups.extend(groups.keys())

            # Calculate group frequencies
            group_freq = Counter(all_groups)
            total_mols = len(cluster_smiles)
            group_freq = {k: v / total_mols for k, v in group_freq.items()}

            # Store analysis results
            cluster_analysis[cluster_id] = {
                "size": sum(cluster_mask),
                "functional_groups": group_freq,
                "center": reduced_embeddings[cluster_mask].mean(axis=0),
            }

        return labels, cluster_analysis


def plot_figure_static(
    df_plot: pd.DataFrame,
    fname: str,
    fig_size: tuple = (12, 8),
    point_size: int = 50,
    alpha: float = 0.6,
) -> plt.Figure:
    """
    Create a static visualization of UMAP embeddings colored by molecular property using seaborn.
    """

    # Set up the matplotlib figure
    plt.clf()
    fig, ax = plt.subplots(figsize=fig_size)

    # Create the scatter plot using seaborn
    scatter = sns.scatterplot(
        data=df_plot,
        x="UMAP1",
        y="UMAP2",
        hue="y",
        palette="viridis",
        s=point_size,
        alpha=alpha,
        ax=ax,
    )

    # Remove the automatic legend created by seaborn
    scatter.legend_.remove()

    # Create and customize the colorbar
    norm = plt.Normalize(df_plot["y"].min(), df_plot["y"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax)  # Pass the axis to the colorbar
    # colorbar.set_label(property_name_with_unit, fontsize=12)

    # Add title and labels
    # ax.set_title(
    #     f"Chemical Structure Space",
    #     fontsize=14,
    #     pad=20,
    # )
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)

    # Add statistics annotation
    # stats_text = (
    #     f"{property_name_with_unit} Statistics:\n"
    #     f"Mean: {y.mean():.2f}\n"
    #     f"Std: {y.std():.2f}\n"
    #     f"Min: {y.min():.2f}\n"
    #     f"Max: {y.max():.2f}"
    # )

    # # Position the text box in figure coords
    # props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    # ax.text(
    #     1.2,
    #     0.98,
    #     stats_text,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox=props,
    # )

    # # Optional: Add cluster centers and labels
    # for cluster_id in set(labels):
    #     if cluster_id != -1:
    #         cluster_mask = df_plot['Cluster'] == cluster_id
    #         cluster_data = df_plot[cluster_mask]
    #         center_x = cluster_data['UMAP1'].mean()
    #         center_y = cluster_data['UMAP2'].mean()
    #         mean_prop = cluster_data['molecular_property'].mean()

    #         # Add cluster label with mean property value
    #         # ax.annotate(f'Cluster {cluster_id}\n{mean_prop:.1f}',
    #         ax.annotate(f'{cluster_id}',
    #                    (center_x, center_y),
    #                    xytext=(5, 5),
    #                    textcoords='offset points',
    #                    fontsize=8,
    #                 #    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    #                    )
    plt.tight_layout()

    save_path = umap_dir / f"{fname}_umap_property_static"
    plt.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    # plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.success(f"Property visualization saved to: {save_path}")

    return fig
