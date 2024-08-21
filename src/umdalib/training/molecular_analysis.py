from dataclasses import dataclass
import json
from typing import Literal
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
    invalid_smiles_indices = np.where(results == None)[0]  # noqa: E711
    invalid_smiles = [smiles_list[i] for i in invalid_smiles_indices]

    if len(invalid_smiles) == 0:
        logger.success("All molecules are valid.")
        return pd.DataFrame(results.tolist())

    logger.warning(f"{len(invalid_smiles)} invalid molecules found.")
    with open(loc / "invalid_smiles_and_indices.csv", "w") as f:
        f.write("SMILES,Index\n")
        for i, smi in zip(invalid_smiles_indices, invalid_smiles):
            f.write(f"{smi},{i}\n")
        logger.warning(
            f"Invalid SMILES strings saved to {str(loc / 'invalid_smiles_and_indices.txt')}."
        )

    # results = np.array([r for r in results if r is not None])
    results = results[results != None]  # noqa: E711
    return pd.DataFrame(results.tolist())


loc: pt = None


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    use_dask: bool
    smiles_column_name: str
    atoms_bin_size: int
    analysis_file: str
    mode: Literal[
        "all", "size_distribution", "structural_distribution", "elemental_distribution"
    ]


def main(args: Args):
    global loc

    if args.analysis_file or args.mode != "all":
        logger.info(f"Analyzing molecules from file... mode: {args.mode}")

        analysis_file = pt(args.analysis_file)
        loc = analysis_file.parent
        logger.info(f"Analysis file: {analysis_file}")
        logger.info(f"Location: {loc}")

        if not loc.exists():
            loc.mkdir(parents=True)
        molecular_analysis(args.analysis_file, args.atoms_bin_size, args.mode)
        logger.success("Analysis complete.")
        return

    logger.info("Analyzing molecules...")

    filename = pt(args.filename)
    logger.info(f"Filename: {filename}")
    loc = pt(filename).parent / f"{filename.stem}_analysis"
    logger.info(f"Location: {loc}")

    if not loc.exists():
        loc.mkdir(parents=True)

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
    df["ElementCategories"] = df["ElementCategories"].apply(
        lambda x: json.dumps(dict(x))
    )
    df["Elements"] = df["Elements"].apply(lambda x: json.dumps(dict(x)))

    analysis_file = loc / "molecule_analysis_results.csv"
    df.to_csv(analysis_file, index=False)
    logger.success(f"Results saved as {analysis_file}")

    molecular_analysis(analysis_file, args.atoms_bin_size)
    logger.success("Analysis complete.")

    return {"analysis_file": str(analysis_file)}


def size_distribution(df: pd.DataFrame, bin_size=10):
    logger.info("Analyzing size distribution...")
    logger.info(f"Binning size: {bin_size}")

    # No. of atoms
    number_of_atoms_distribution = Counter(df["No. of atoms"].values)
    # logger.info(f"Number of atoms distribution: {number_of_atoms_distribution}")

    # Convert the Counter to a DataFrame
    number_of_atoms_distribution_df = pd.DataFrame.from_dict(
        number_of_atoms_distribution, orient="index"
    ).reset_index()
    number_of_atoms_distribution_df.columns = ["No. of atoms", "Count"]
    number_of_atoms_distribution_df = number_of_atoms_distribution_df.sort_values(
        by="No. of atoms"
    )
    number_of_atoms_distribution_df.to_csv(loc / "size_distribution.csv", index=False)

    min_atom_size = number_of_atoms_distribution_df["No. of atoms"].min()
    max_atom_size = number_of_atoms_distribution_df["No. of atoms"].max()
    logger.info(f"Min atomic size: {min_atom_size}")
    logger.info(f"Max atomic size: {max_atom_size}")

    # Create bins
    bins = list(np.arange(min_atom_size, max_atom_size, bin_size))
    bins.append(max_atom_size)  # Add max_atom_size as the last bin edge

    # Create labels
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

    # Apply pd.cut with the new bins and labels
    number_of_atoms_distribution_df["Bins"] = pd.cut(
        number_of_atoms_distribution_df["No. of atoms"].astype(int),
        bins=bins,
        labels=labels,
        right=True,  # Changed to True to include the right edge in each bin
        include_lowest=True,  # Ensures the lowest value is included in the first bin
    )

    #  the observed parameter is used to control whether only the observed categories are included in the result when grouping by a categorical variable.
    binned_df = (
        number_of_atoms_distribution_df.groupby("Bins", observed=True)["Count"]
        .sum()
        .reset_index()
    )

    # binned_df = binned_df.sort_values(by="Count", ascending=False)
    logger.info(f"Binned distribution of number of atoms: {binned_df}")
    binned_file = loc / "binned_size_distribution.csv"
    binned_df.to_csv(binned_file, index=False)

    logger.success(f"Binned distribution saved as {binned_file}")

    return binned_df


def structural_distribution(df: pd.DataFrame):
    logger.info("Analyzing structural distribution...")

    IsAromatic_counts = df["IsAromatic"].sum()
    IsNonCyclic_counts = df["IsNonCyclic"].sum()
    IsCyclicNonAromatic_counts = df["IsCyclicNonAromatic"].sum()

    counts = [IsAromatic_counts, IsNonCyclic_counts, IsCyclicNonAromatic_counts]
    labels = ["aromatic", "non-cyclic", "cyclic non-aromatic"]

    # save to file
    structural_distribution_file = loc / "structural_distribution.csv"
    df = pd.DataFrame({"Structural Category": labels, "Count": counts})
    df.to_csv(structural_distribution_file, index=False)

    logger.success(f"Structural distribution saved as {structural_distribution_file}")
    return df


def elemental_distribution(df: pd.DataFrame):
    logger.info("Analyzing elemental distribution...")
    elements = Counter()
    for e in df["Elements"]:
        elements.update(e)

    logger.info(elements)

    logger.info("Total elements: ", len(elements))

    logger.info("Top 10 elements counts:")
    for element, count in elements.most_common(10):
        logger.info(f"{element}: {count}")

    elements_containing = Counter()
    for element in elements.keys():
        elements_containing[element] = (
            df["Elements"].apply(lambda x: element in x).sum()
        )
    elements_containing_df = pd.DataFrame.from_dict(
        elements_containing, orient="index"
    ).reset_index()
    elements_containing_df.columns = ["Element", "Count"]
    elements_containing_df = elements_containing_df.sort_values(
        by="Count", ascending=False
    )

    # save to file
    elements_containing_file = loc / "elemental_distribution.csv"
    elements_containing_df.to_csv(elements_containing_file, index=False)
    logger.success(
        f"Elements containing distribution saved as {elements_containing_file}"
    )

    return elements_containing_df


def molecular_analysis(csv_file: str = None, bin_size=10, mode="all"):
    logger.info("Analyzing molecules from file...")
    csv_file = pt(csv_file)
    df = pd.read_csv(csv_file, index_col=False)

    logger.info(f"Fetched {len(df)} molecules from {csv_file}")
    logger.info(f"File columns: {df.columns}\n {df.iloc[0]}")

    logger.info("Converting JSON strings to Counter objects...")
    logger.info(
        f"{df.iloc[0]['ElementCategories']=}\n{type(df.iloc[0]['ElementCategories'])=}"
    )
    logger.info(f"{df.iloc[0]['Elements']=}\n{type(df.iloc[0]['Elements'])=}")

    df["ElementCategories"] = df["ElementCategories"].apply(
        lambda x: Counter(json.loads(x))
    )
    df["Elements"] = df["Elements"].apply(lambda x: Counter(json.loads(x)))
    logger.info("Converted JSON strings to Counter objects.")

    logger.info("Analyzing molecules...")
    logger.info(f"Analyzing {len(df)} molecules...")

    if mode == "size_distribution":
        size_distribution(df, bin_size)
        return
    elif mode == "structural_distribution":
        structural_distribution(df)
        return
    elif mode == "elemental_distribution":
        elemental_distribution(df)
        return

    size_distribution(df, bin_size)
    structural_distribution(df)
    elemental_distribution(df)
    return
