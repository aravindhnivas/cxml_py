import pandas as pd
import numpy as np
from rdkit import Chem
from collections import Counter
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class ChemicalClusterAnalyzer:
    """Analyzes chemical structure clusters and their functional groups."""
    
    def __init__(self):
        # Define common functional groups and their SMARTS patterns
        self.functional_groups = {
            'Alcohol': '[OH]',
            'Carboxylic Acid': '[CX3](=O)[OX2H1]',
            'Ester': '[#6][CX3](=O)[OX2H0][#6]',
            'Ether': '[OD2]([#6])[#6]',
            'Aldehyde': '[CX3H1](=O)[#6]',
            'Ketone': '[#6][CX3](=O)[#6]',
            'Amine': '[NX3;H2,H1;!$(NC=O)]',
            'Amide': '[NX3][CX3](=[OX1])[#6]',
            'Aromatic': 'a1aaaaa1',
            'Alkene': '[CX3]=[CX3]',
            'Alkyne': '[CX2]#[CX2]',
            'Nitrile': '[NX1]#[CX2]',
            'Nitro': '[NX3](=O)=O',
            'Sulfonic Acid': '[SX4](=[OX1])(=[OX1])[OX2H]',
            'Phosphate': '[PX4](=[OX1])([OX2H])([OX2H])[OX2H]',
            'Halogen': '[F,Cl,Br,I]'
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
        except:
            return {}

    def analyze_cluster_chemistry(self, 
                                reduced_embeddings: np.ndarray, 
                                smiles_list: List[str], 
                                eps: float = 0.5, 
                                min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
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
        for cluster_id in set(labels):
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
            group_freq = {k: v/total_mols for k, v in group_freq.items()}
            
            # Store analysis results
            cluster_analysis[cluster_id] = {
                'size': sum(cluster_mask),
                'functional_groups': group_freq,
                'center': reduced_embeddings[cluster_mask].mean(axis=0)
            }
        
        return labels, cluster_analysis

    def plot_cluster_analysis(self, 
                            reduced_embeddings: np.ndarray,
                            smiles_list: List[str],
                            labels: np.ndarray,
                            cluster_analysis: Dict,
                            output_path: str = None):
        """
        Creates an interactive visualization of clusters with chemical analysis.
        
        Args:
            reduced_embeddings: UMAP-reduced embeddings
            smiles_list: List of SMILES strings
            labels: Cluster labels
            cluster_analysis: Cluster analysis results
            output_path: Path to save the plot
        """
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'UMAP1': reduced_embeddings[:, 0],
            'UMAP2': reduced_embeddings[:, 1],
            'Cluster': labels,
            'SMILES': smiles_list
        })
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add scatter points for each cluster
        for cluster_id in set(labels):
            cluster_data = df_plot[df_plot['Cluster'] == cluster_id]
            
            # Get dominant functional groups for hover text
            if cluster_id in cluster_analysis:
                top_groups = sorted(
                    cluster_analysis[cluster_id]['functional_groups'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                hover_text = [
                    f"SMILES: {s}<br>Cluster: {cluster_id}<br>" +
                    "<br>".join([f"{g}: {v:.1%}" for g, v in top_groups])
                    for s in cluster_data['SMILES']
                ]
            else:
                hover_text = [f"SMILES: {s}<br>Cluster: Noise" for s in cluster_data['SMILES']]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['UMAP1'],
                y=cluster_data['UMAP2'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=hover_text,
                hoverinfo='text',
                marker=dict(size=8)
            ))
        
        # Add cluster annotations
        for cluster_id, info in cluster_analysis.items():
            center = info['center']
            top_groups = sorted(
                info['functional_groups'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            annotation_text = f"Cluster {cluster_id}<br>" + "<br>".join(
                [f"{g}: {v:.1%}" for g, v in top_groups]
            )
            
            fig.add_annotation(
                x=center[0],
                y=center[1],
                text=annotation_text,
                showarrow=True,
                arrowhead=1,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        
        # Update layout
        fig.update_layout(
            title='Chemical Structure Clusters Analysis',
            template='plotly_white',
            width=1200,
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()

def main(input_filepath: str,
         output_filepath: str = None,
         n_neighbors: int = 15,
         min_dist: float = 0.1,
         cluster_eps: float = 0.5,
         cluster_min_samples: int = 5):
    """
    Main function to run the chemical cluster analysis pipeline.
    """
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv(input_filepath)
    smiles_col = next(col for col in df.columns if 'smiles' in col.lower())
    smiles = df[smiles_col].values
    embedding_cols = [col for col in df.columns if col != smiles_col]
    embeddings = df[embedding_cols].values
    
    # Scale embeddings
    print("Scaling embeddings...")
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # Perform UMAP
    print("Performing UMAP reduction...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(scaled_embeddings)
    
    # Analyze clusters
    print("Analyzing chemical clusters...")
    analyzer = ChemicalClusterAnalyzer()
    labels, cluster_analysis = analyzer.analyze_cluster_chemistry(
        reduced_embeddings,
        smiles,
        eps=cluster_eps,
        min_samples=cluster_min_samples
    )
    
    # Create visualization
    print("Creating visualization...")
    analyzer.plot_cluster_analysis(
        reduced_embeddings,
        smiles,
        labels,
        cluster_analysis,
        output_filepath
    )
    
    print("Analysis complete!")
    return reduced_embeddings, labels, cluster_analysis

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chemical structure cluster analysis')
    parser.add_argument('input_file', help='Path to input CSV file containing embeddings')
    parser.add_argument('--output', help='Path to save interactive visualization')
    parser.add_argument('--n-neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--cluster-eps', type=float, default=0.5, help='DBSCAN eps parameter')
    parser.add_argument('--cluster-min-samples', type=int, default=5, help='DBSCAN min_samples parameter')
    
    args = parser.parse_args()
    
    main(
        args.input_file,
        args.output,
        args.n_neighbors,
        args.min_dist,
        args.cluster_eps,
        args.cluster_min_samples
    )