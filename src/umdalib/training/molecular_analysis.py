from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from collections import Counter
from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger
import multiprocessing
from pathlib import Path as pt

# from utils import loc

RDLogger.DisableLog("rdApp.*")


def is_aromatic(mol):
    """Check if the molecule is aromatic."""
    return any(atom.GetIsAromatic() for atom in mol.GetAtoms())


def is_non_cyclic(mol):
    """Check if the molecule is non-cyclic."""
    return mol.GetRingInfo().NumRings() == 0


def is_cyclic(mol):
    """Check if the molecule is cyclic."""
    return mol.GetRingInfo().NumRings() > 0 and not is_aromatic(mol)


def categorize_element(atomic_num):
    """Categorize an element based on its atomic number."""
    if atomic_num in list(range(3, 11)) + list(range(19, 37)) + list(
        range(37, 55)
    ) + list(range(55, 86)):
        return "Metal"
    elif atomic_num in [3, 11, 19, 37, 55, 87]:
        return "Alkali Metal"
    elif atomic_num in range(57, 72):
        return "Lanthanide"
    elif atomic_num in [1, 2] + list(range(6, 10)) + list(range(14, 18)) + list(
        range(32, 36)
    ) + list(range(50, 54)) + list(range(82, 86)):
        return "Non-metal"
    else:
        return "Other"


def categorize_molecule(mol):
    """Categorize a molecule as organic or inorganic."""
    if any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        return "Organic"
    else:
        return "Inorganic"


def analyze_single_molecule(smi):
    """Analyze a single molecule."""
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None

    return {
        "SMILES": smi,
        "MolecularWeight": Descriptors.ExactMolWt(mol),
        "No. of atoms": mol.GetNumAtoms(),
        "IsAromatic": is_aromatic(mol),
        "IsNonCyclic": is_non_cyclic(mol),
        "IsCyclicNonAromatic": is_cyclic(mol),
        "Category": categorize_molecule(mol),
        "Elements": Counter(atom.GetSymbol() for atom in mol.GetAtoms()),
        "ElementCategories": Counter(
            categorize_element(atom.GetAtomicNum()) for atom in mol.GetAtoms()
        ),
    }


invalid_smiles = []


def analyze_molecules(smiles_list: list[str], parallel=True):
    """Analyze a list of SMILES strings in parallel and return a DataFrame with results."""

    global invalid_smiles

    results = []
    if parallel:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.map(analyze_single_molecule, smiles_list)

        results = np.array(results)
    else:
        results = np.array(list(map(analyze_single_molecule, smiles_list)))

    if len(results) == 0:
        logger.error("No valid molecules found.")
        raise ValueError("No valid molecules found.")

    # None values are returned for invalid SMILES strings
    invalid_smiles_indices = np.where(results == None)[0]
    invalid_smiles = [smiles_list[i] for i in invalid_smiles_indices]

    if len(invalid_smiles) == 0:
        logger.success("All molecules are valid.")
        return pd.DataFrame(results.tolist())

    logger.warning(f"{len(invalid_smiles)} invalid molecules found.")
    with open(loc / "invalid_smiles_and_indices.txt", "w") as f:
        f.write("# SMILES\tIndex\n")
        for i, smi in zip(invalid_smiles_indices, invalid_smiles):
            f.write(f"{smi}\t{i}\n")
        logger.warning(
            f"Invalid SMILES strings saved to {str(loc / 'invalid_smiles_and_indices.txt')}."
        )

    results = results[results != None]
    return pd.DataFrame(results.tolist())


loc: pt = None


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    use_dask: bool
    smiles_column_name: str


def main(args: Args):

    global loc
    logger.info("Analyzing molecules...")

    loc = pt(args.filename).parent
    df = read_as_ddf(
        args.filetype,
        args.filename,
        args.key,
        use_dask=args.use_dask,
        computed=args.use_dask,
    )

    smiles_list = df[args.smiles_column_name].tolist()

    logger.info(f"Analyzing {len(smiles_list)} molecules...")
    df = analyze_molecules(smiles_list, parallel=True)
    logger.info(f"Analysis complete. {len(df)} valid molecules processed.")

    logger.info("Analysis Summary:")
    # logger.info(df.head())
    logger.info(f"Total molecules analyzed: {len(df)}")
    logger.info(f"Organic molecules: {df['Category'].value_counts().get('Organic', 0)}")
    logger.info(
        f"Inorganic molecules: {df['Category'].value_counts().get('Inorganic', 0)}"
    )
    logger.info(f"Aromatic molecules: {df['IsAromatic'].sum()}")
    logger.info(f"Non-cyclic molecules: {df['IsNonCyclic'].sum()}")
    logger.info(f"Cyclic molecules: {df['IsCyclicNonAromatic'].sum()}")

    logger.info("Top 10 Elements:")
    elements = Counter()
    for e in df["Elements"]:
        elements.update(e)
    for element, count in elements.most_common(10):
        logger.info(f"{element}: {count}")

    logger.info("Element Categories:")

    elem_cats = Counter()
    for ec in df["ElementCategories"]:
        elem_cats.update(ec)
    for category, count in elem_cats.items():
        logger.info(f"{category}: {count}")

    # Convert Counter objects to JSON strings
    df["ElementCategories"] = df["ElementCategories"].apply(lambda x: json.dumps(x))
    df["Elements"] = df["Elements"].apply(lambda x: json.dumps(x))

    analysis_file = loc / "molecule_analysis_results.csv"
    df.to_csv(analysis_file, index=False)
    logger.success(f"Results saved as {analysis_file}")

    return {"analysis_file": str(analysis_file)}
