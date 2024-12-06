import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path as pt
from typing import Dict, List, Optional, Tuple

# from umdalib.utils.json import safe_json_dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from umdalib.logger import logger
from umdalib.utils.json import safe_json_dump

warnings.filterwarnings("ignore")


def get_plotly_data(
    umap_df: pd.DataFrame,
    smi_column: str,
    property_name_with_unit: str,
) -> None:
    umap_1 = umap_df["UMAP1"].to_list()
    umap_2 = umap_df["UMAP2"].to_list()
    y = umap_df["y"].to_list()
    labels = umap_df["Cluster"].to_list()
    smiles_list = umap_df[smi_column].to_list()

    # Calculate statistics
    stats = {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
    }

    # Create the plot data structure
    plot_data = {
        "data": [
            {
                "type": "scatter",
                "x": umap_1,
                "y": umap_2,
                "mode": "markers",
                "marker": {
                    "size": 8,
                    "color": y,
                    "colorscale": "Viridis",
                    "colorbar": {"title": property_name_with_unit},
                    "showscale": True,
                },
                "text": [
                    f"SMILES: {s}<br>{property_name_with_unit}: {v:.2f}<br>Cluster: {c}"
                    for s, v, c in zip(smiles_list, y, labels)
                ],
                "hoverinfo": "text",
            }
        ],
        "layout": {
            "title": f"Chemical Structure Space Colored by {property_name_with_unit}",
            "template": "plotly_white",
            # "width": 1200,
            # "height": 800,
            "showlegend": False,
            "hovermode": "closest",
            "xaxis": {"title": "UMAP1"},
            "yaxis": {"title": "UMAP2"},
            "annotations": [
                {
                    "x": 0.02,
                    "y": 0.98,
                    "xref": "paper",
                    "yref": "paper",
                    "text": (
                        f"{property_name_with_unit} Statistics:<br>"
                        f"Mean: {stats['mean']:.2f}<br>"
                        f"Std: {stats['std']:.2f}<br>"
                        f"Min: {stats['min']:.2f}<br>"
                        f"Max: {stats['max']:.2f}"
                    ),
                    "showarrow": False,
                    "font": {"size": 12},
                    "bgcolor": "white",
                    "bordercolor": "black",
                    "borderwidth": 1,
                    "align": "left",
                }
            ],
        },
    }

    return plot_data


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


def annotate_clusters(umap_df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    # Define distinct colors for clusters
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(umap_df["Cluster"].unique())))

    # Add cluster circles and labels
    for idx, cluster_id in enumerate(sorted(umap_df["Cluster"].unique())):
        if cluster_id == -1:  # Skip noise points
            continue

        # Get points for this cluster
        cluster_points = umap_df[umap_df["Cluster"] == cluster_id]

        # Calculate cluster center
        center_x = cluster_points["UMAP1"].mean()
        center_y = cluster_points["UMAP2"].mean()

        # Calculate cluster radius
        std_x = cluster_points["UMAP2"].std() * 2
        std_y = cluster_points["UMAP2"].std() * 2
        radius = max(std_x, std_y)

        # Get color for this cluster
        cluster_color = cluster_colors[idx]

        # Draw circle around cluster
        circle = plt.Circle(
            (center_x, center_y),
            radius,
            fill=False,
            linestyle="-",
            color=cluster_color,
            alpha=0.8,
            linewidth=2,
        )
        ax.add_patch(circle)

        # Add just the cluster ID at the circle's boundary
        # Calculate position on the circle's circumference (top of circle)
        label_x = center_x
        label_y = center_y + radius

        # Add cluster label with white background for better visibility
        ax.text(
            label_x,
            label_y,
            str(cluster_id),
            horizontalalignment="center",
            verticalalignment="bottom",
            color=cluster_color,
            fontweight="bold",
            fontsize=12,
            bbox=dict(
                facecolor="white",
                edgecolor=cluster_color,
                alpha=0.9,
                pad=0.5,
                boxstyle="round",
            ),
        )
    return ax


def plot_figure_static(
    df_plot: pd.DataFrame,
    fig_size: tuple = (12, 8),
    point_size: int = 50,
    alpha: float = 0.6,
    colorbar_label: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
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
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=12)
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
    return fig, ax


@dataclass
class Args:
    n_neighbors: int
    min_dist: float
    n_components: int
    umap_metric: str
    n_jobs: int
    scale_embedding: bool
    annotate_clusters: bool
    label_issues_file: Optional[str]
    processed_df_file: str
    columnX: str
    dbscan_eps: float
    dbscan_min_samples: int
    training_filename: str
    random_state: Optional[int]
    fig_title: Optional[str]


def validate_args(args: Args) -> Args:
    args.n_neighbors = int(args.n_neighbors)
    args.min_dist = float(args.min_dist)
    args.n_components = int(args.n_components)
    args.n_jobs = int(args.n_jobs)
    args.scale_embedding = bool(args.scale_embedding)
    args.annotate_clusters = bool(args.annotate_clusters)
    args.label_issues_file = args.label_issues_file or None
    args.dbscan_eps = float(args.dbscan_eps)
    args.dbscan_min_samples = int(args.dbscan_min_samples)
    args.random_state = args.random_state or None
    args.fig_title = args.fig_title or None
    return args


def get_save_fname(args: Args) -> str:
    save_fname = pt(args.training_filename).stem

    if args.scale_embedding:
        save_fname += "_scaled"

    if args.random_state is not None:
        save_fname += f"_random_state_{args.random_state}"

    save_fname += f"_umap_{args.n_neighbors}_{args.min_dist}_{args.n_components}"
    save_fname += (
        f"_cluster_eps_{args.dbscan_eps}_min_samples_{args.dbscan_min_samples}"
    )

    return save_fname


def main(args: Args):
    args = validate_args(args)
    processed_df_file = pt(args.processed_df_file)
    umap_dir = processed_df_file.parent / "umap"

    if args.label_issues_file:
        label_issues_file = pt(args.label_issues_file)
        umap_dir = umap_dir / f"cleaned_{label_issues_file.stem}"
    umap_dir.mkdir(exist_ok=True)

    save_fname = get_save_fname(args)
    fig_file = umap_dir / f"[figure]_{save_fname}.pdf"
    plotly_data_file = umap_dir / f"[plotly_data]_{save_fname}.json"
    umap_df_file = umap_dir / f"[umap_df]_{save_fname}.parquet"

    def savefig(umap_df: pd.DataFrame):
        plotly_data = get_plotly_data(umap_df, args.columnX, args.fig_title)
        safe_json_dump(plotly_data, plotly_data_file)
        fig, ax = plot_figure_static(
            umap_df,
            fig_size=(12, 8),
            point_size=50,
            alpha=0.6,
            colorbar_label=args.fig_title,
        )
        if args.annotate_clusters:
            ax = annotate_clusters(umap_df, ax)

        # Add statistics annotation
        clusters = umap_df["Cluster"].unique()
        n_clusters = len([c for c in clusters if c != -1])
        stats_text = (
            "UMAP Parameters:\n"
            f"n_neighbors: {args.n_neighbors}\n"
            f"min_dist: {args.min_dist}\n"
            f"n_components: {args.n_components}\n"
            f"DBSCAN Parameters (N={n_clusters}):\n"
            f"eps: {args.dbscan_eps}\n"
            f"min_samples: {args.dbscan_min_samples}"
        )

        # Position the text box in figure coords
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            1.15,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        fig.savefig(fig_file, dpi=300, bbox_inches="tight")
        logger.success(f"Property visualization saved to: {fig_file}")
        # plt.show()

    if umap_df_file.exists():
        logger.info("Loading UMAP dataframe...")
        umap_df = pd.read_parquet(umap_df_file)
        savefig(umap_df)
        return {
            "umap_df_file": umap_df_file,
            "plotly_data_file": plotly_data_file,
        }

    logger.info(f"Processing UMAP for {processed_df_file}")
    df = pd.read_parquet(processed_df_file)
    logger.info(f"Loaded df: {df.shape}")

    if args.label_issues_file:
        label_issues_df = pd.read_parquet(args.label_issues_file)
        logger.info(f"Label issues: {label_issues_df.shape}")

        cleaned_label_df = label_issues_df[~label_issues_df["is_label_issue"]]
        cleaned_df = df.loc[cleaned_label_df.index]
        if (cleaned_df.index == cleaned_label_df.index).all():
            df = cleaned_df
            logger.info(f"Cleaned df: {df.shape}")

    smiles_list = df[args.columnX].to_list()
    logger.info(len(smiles_list))

    embeddings = df.iloc[:, 2:].to_numpy()
    y = df["y"].to_numpy()
    logger.info(f"X shape: {embeddings.shape}, y shape: {y.shape}")

    if args.scale_embedding:
        logger.info("Scaling embeddings...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    if args.random_state is not None:
        logger.info(f"Random state: {args.random_state}")
        args.n_jobs = 1

    logger.info("Performing UMAP...")
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

    umap_df.to_parquet(umap_df_file)
    logger.success(f"UMAP embeddings saved to {umap_dir}")
    savefig(umap_df)

    # cluster_analysis_stats_df = pd.DataFrame(cluster_analysis).T
    # cluster_analysis_stats_df_file = (
    #     umap_dir / f"[cluster_analysis_stats_df]_{save_fname}.parquet"
    # )
    # if not cluster_analysis_stats_df_file.exists():
    #     for cluster_id in sorted(set(labels)):
    #         if cluster_id == -1:
    #             continue
    #         cluster_data: pd.DataFrame = umap_df[umap_df["Cluster"] == cluster_id]
    #         cluster_analysis_stats_df.loc[cluster_id, "Mean"] = cluster_data["y"].mean()
    #         cluster_analysis_stats_df.loc[cluster_id, "Std"] = cluster_data["y"].std()
    #         cluster_analysis_stats_df.loc[cluster_id, "Min"] = cluster_data["y"].min()
    #         cluster_analysis_stats_df.loc[cluster_id, "Max"] = cluster_data["y"].max()

    #     cluster_analysis_stats_df.to_parquet(cluster_analysis_file)
    #     logger.success(f"Cluster analysis saved to {umap_dir}")

    return {
        "umap_df_file": umap_df_file,
        "plotly_data_file": plotly_data_file,
    }
