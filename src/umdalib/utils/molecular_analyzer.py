from rdkit import Chem
from rdkit.Chem import Descriptors
from dataclasses import dataclass
from umdalib.logger import logger


class MoleculeAnalyzer:
    """
    A class for analyzing molecules from SMILES strings, calculating various
    properties, including ring information, aromaticity, functional groups,
    and chain analysis (alkane, alkene, alkyne).
    """

    def __init__(self, smiles):
        """
        Initializes the MoleculeAnalyzer with a SMILES string.

        Args:
            smiles (str): The SMILES string of the molecule.
        """
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.results = {}  # Initialize an empty dictionary to store results

        if self.mol is None:
            raise ValueError("Invalid SMILES string.")

    def _calculate_basic_properties(self):
        """Calculates basic molecular properties."""
        results = {}
        results["num_atoms"] = self.mol.GetNumAtoms()
        results["num_bonds"] = self.mol.GetNumBonds()
        results["molecular_weight"] = Descriptors.MolWt(self.mol)

        self.results["basic_properties"] = results

    def _calculate_ring_information(self):
        """Calculates information about rings in the molecule."""
        ri = self.mol.GetRingInfo()
        results = {}
        results["num_rings"] = ri.NumRings()
        results["ring_sizes"] = [len(ring) for ring in ri.AtomRings()]

        self.results["ring_information"] = results

    def _calculate_aromaticity(self):
        """
        Calculates aromaticity information, including the number of aromatic rings,
        atoms, and bonds.
        """
        Chem.Kekulize(self.mol, clearAromaticFlags=True)

        ri = self.mol.GetRingInfo()
        num_aromatic_rings = 0
        aromatic_atoms = set()
        aromatic_bonds = set()  # Store aromatic bond indices

        for ring in ri.AtomRings():
            is_aromatic = True
            for atom_idx in ring:
                atom = self.mol.GetAtomWithIdx(atom_idx)
                if not (
                    atom.GetIsAromatic()
                    or atom.GetHybridization() == Chem.HybridizationType.SP2
                ):
                    is_aromatic = False
                    break

            if is_aromatic:
                num_aromatic_rings += 1
                aromatic_atoms.update(ring)
                for i in range(len(ring)):
                    atom1_idx = ring[i]
                    atom2_idx = ring[(i + 1) % len(ring)]
                    bond = self.mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                    if bond:
                        aromatic_bonds.add(bond.GetIdx())

        results = {}
        results["num_aromatic_rings"] = num_aromatic_rings
        results["num_aromatic_atoms"] = len(aromatic_atoms)
        results["aromatic_bonds"] = list(aromatic_bonds)  # Store indices as a list

        self.results["aromaticity"] = results

    def _calculate_physicochemical_properties(self):
        """Calculates LogP, TPSA, HBD, HBA, and rotatable bonds."""

        results = {}
        results["logp"] = Descriptors.MolLogP(self.mol)
        results["tpsa"] = Descriptors.TPSA(self.mol)
        results["num_hbd"] = Descriptors.NumHDonors(self.mol)
        results["num_hba"] = Descriptors.NumHAcceptors(self.mol)
        results["num_rotatable_bonds"] = Descriptors.NumRotatableBonds(self.mol)

        self.results["physicochemical_properties"] = results

    def _remove_rings_and_get_molecule(self):
        """
        Removes bonds in rings from the molecule and returns a modified molecule
        without rings.
        """

        # Create an editable molecule to remove specific bonds
        editable_mol = Chem.EditableMol(self.mol)

        # Remove aromatic bonds
        for bond_idx in self.results["aromaticity"][
            "aromatic_bonds"
        ]:  # Correctly iterate over bond indices
            bond = self.mol.GetBondWithIdx(bond_idx)
            if bond:
                editable_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        # Remove non-aromatic ring bonds
        ri = self.mol.GetRingInfo()
        for ring in ri.AtomRings():
            for i in range(len(ring)):
                atom1_idx = ring[i]
                atom2_idx = ring[(i + 1) % len(ring)]
                bond = self.mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                if (
                    bond
                    and bond.GetIdx()
                    not in self.results["aromaticity"]["aromatic_bonds"]
                ):
                    editable_mol.RemoveBond(atom1_idx, atom2_idx)

        return editable_mol.GetMol()

    def _get_connected_components(self, chains):
        """
        Helper function to find connected components in a list of bonds using
        Depth-First Search (DFS).
        """
        visited = set()
        components = []

        def dfs(atom_idx, current_component):
            visited.add(atom_idx)
            current_component.append(atom_idx)
            for neighbor in graph.get(atom_idx, []):
                if neighbor not in visited:
                    dfs(neighbor, current_component)

        # Build an adjacency list (graph)
        graph = {}
        for atom1_idx, atom2_idx in chains:
            graph.setdefault(atom1_idx, []).append(atom2_idx)
            graph.setdefault(atom2_idx, []).append(atom1_idx)

        # Perform DFS
        for atom_idx in graph:
            if atom_idx not in visited:
                component = []
                dfs(atom_idx, component)
                components.append(component)

        return components

    def _get_longest_chain(self, components):
        """
        Calculates the length of the longest chain in a set of connected components.
        """
        longest_chain_length = 0
        for component in components:
            longest_chain_length = max(longest_chain_length, len(component))
        return longest_chain_length

    def _calculate_chain_information(self):
        """Calculates information about alkane, alkene, and alkyne chains."""
        mol_no_rings = self._remove_rings_and_get_molecule()

        alkane_chains = []
        alkene_chains = []
        alkyne_chains = []

        # Iterate through bonds and classify them
        for bond in mol_no_rings.GetBonds():
            bond_type = bond.GetBondType()
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            # Consider only carbon-carbon bonds
            if atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6:
                if bond_type == Chem.BondType.SINGLE:
                    alkane_chains.append((atom1.GetIdx(), atom2.GetIdx()))
                elif bond_type == Chem.BondType.DOUBLE:
                    alkene_chains.append((atom1.GetIdx(), atom2.GetIdx()))
                elif bond_type == Chem.BondType.TRIPLE:
                    alkyne_chains.append((atom1.GetIdx(), atom2.GetIdx()))

        # Get connected components for each chain type
        alkane_components = self._get_connected_components(alkane_chains)
        alkene_components = self._get_connected_components(alkene_chains)
        alkyne_components = self._get_connected_components(alkyne_chains)

        results = {}

        # Calculate the longest chain and number of chains for each type
        results["longest_alkane_chain"] = self._get_longest_chain(alkane_components)
        results["longest_alkene_chain"] = self._get_longest_chain(alkene_components)
        results["longest_alkyne_chain"] = self._get_longest_chain(alkyne_components)

        results["num_alkane_chains"] = len(alkane_components)
        results["num_alkene_chains"] = len(alkene_components)
        results["num_alkyne_chains"] = len(alkyne_components)

        self.results["chain_information"] = results

    def _calculate_heterocyclicity(self):
        """
        Calculates heterocyclicity information, including the number of
        heterocyclic rings and the percentage of heteroatoms in those rings.
        """
        ri = self.mol.GetRingInfo()
        heterocycle_count = 0
        heteroatom_count = 0
        total_atoms_in_heterocycles = 0

        for ring in ri.AtomRings():
            is_heterocycle = False
            ring_atom_count = 0
            for atom_idx in ring:
                atom = self.mol.GetAtomWithIdx(atom_idx)
                ring_atom_count += 1
                if atom.GetAtomicNum() != 6:  # Check if it's not a carbon atom
                    is_heterocycle = True
                    heteroatom_count += 1

            if is_heterocycle:
                heterocycle_count += 1
                total_atoms_in_heterocycles += ring_atom_count

        results = {}
        results["num_heterocycles"] = heterocycle_count

        if total_atoms_in_heterocycles > 0:
            results["percent_heteroatoms_in_heterocycles"] = (
                heteroatom_count / total_atoms_in_heterocycles
            ) * 100
        else:
            results["percent_heteroatoms_in_heterocycles"] = 0.0

        self.results["heterocyclicity"] = results

    def _analyze_functional_groups(self):
        """
        Analyzes the molecule for a variety of common functional groups using
        SMARTS patterns.
        """

        # Dictionary of functional group names and their corresponding SMARTS patterns
        functional_groups = {
            "alcohol": "[OX2H]",
            "aldehyde": "[CX3H1](=O)",
            "ketone": "[CX3](=[OX1])",
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "ester": "[CX3](=O)[OX2][#6]",  # Changed from "[CX3](=O)[OX2H0]"
            "ether": "[OD2]([#6])[#6]",
            "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
            "amide": "[NX3][CX3](=[OX1])",
            "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
            "nitrile": "[NX1]#[CX2]",
            "sulfide": "[#16X2H0]",
            "sulfoxide": "[#16X3](=[OX1])",
            "sulfone": "[#16X4](=[OX1])(=[OX1])",
            "thiol": "[#16X2H]",
            "alkene": "[CX3]=[CX3]",
            "alkyne": "[CX2]#[CX2]",
            "aromatic": "a",  # RDKit recognizes 'a' as aromatic in SMARTS
            "halide_alkyl": "[#6][F,Cl,Br,I]",
            "halide_aryl": "[c][F,Cl,Br,I]",
            "phenol": "[OX2H][c]",
        }

        # Analyze each functional group
        results = {}
        for group_name, smarts in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = self.mol.GetSubstructMatches(pattern)
                results[f"num_{group_name}"] = len(matches)
            else:
                print(f"Warning: Invalid SMARTS pattern for {group_name}")

        self.results["functional_groups"] = results

    def analyze(self):
        """
        Performs the complete molecular analysis by calling all the individual
        calculation methods.
        """
        self._calculate_basic_properties()
        self._calculate_ring_information()
        self._calculate_aromaticity()
        self._calculate_physicochemical_properties()
        self._calculate_chain_information()
        self._calculate_heterocyclicity()
        self._analyze_functional_groups()

        return self.results


@dataclass
class Args:
    smiles: str


def main(args: Args):
    logger.info(f"Analyzing molecule with SMILES string: {args.smiles}")
    analyzer = MoleculeAnalyzer(args.smiles)
    results = analyzer.analyze()
    logger.info(f"Analysis results: {results}")

    # save results to a file
    # file_path = pt("molecule_analysis_results.json")
    # logger.info(f"Saving results to {file_path}")
    # safe_json_dump(results, file_path)

    return {
        "full_analysis": results,
    }


# "aromatic_bonds"
# "logp"
# "longest_alkane_chain"
# "longest_alkene_chain"
# "longest_alkyne_chain"
# "molecular_weight"
# "num_alcohol"
# "num_aldehyde"
# "num_alkane_chains"
# "num_alkene"
# "num_alkene_chains"
# "num_alkyne"
# "num_alkyne_chains"
# "num_amide"
# "num_amine"
# "num_amines"
# "num_aromatic"
# "num_aromatic_atoms"
# "num_aromatic_rings"
# "num_atoms"
# "num_bonds"
# "num_carboxylic_acid"
# "num_carboxylic_acids"
# "num_ester"
# "num_ether"
# "num_halide_alkyl"
# "num_halide_aryl"
# "num_hba"
# "num_hbd"
# "num_heterocycles"
# "num_ketone"
# "num_nitrile"
# "num_nitro"
# "num_phenol"
# "num_rings"
# "num_rotatable_bonds"
# "num_sulfide"
# "num_sulfone"
# "num_sulfoxide"
# "num_thiol"
# "percent_heteroatoms_in_heterocycles"
# "ring_sizes"
# "tpsa"
