import numpy as np
from rdkit import Chem
from collections import Counter
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


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
