import json
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path as pt
from typing import Any, Callable, Optional

import pandas as pd

from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger

# Constants for column names
COLUMN_ATOMS = "No. of atoms"
COLUMN_ELEMENTS = "Elements"
COLUMN_IS_AROMATIC = "IsAromatic"
COLUMN_IS_NON_CYCLIC = "IsNonCyclic"
COLUMN_IS_CYCLIC_NON_AROMATIC = "IsCyclicNonAromatic"

removed_indices_condition = []


def apply_filters_to_df(
    row: pd.Series,
    min_atomic_number: Optional[int],
    max_atomic_number: Optional[int],
    filter_elements: list[str],
    filter_structures: list[str],
) -> bool:
    # Filter based on atomic number
    if min_atomic_number:
        if row[COLUMN_ATOMS] < int(min_atomic_number):
            return False

    if max_atomic_number:
        if row[COLUMN_ATOMS] > int(max_atomic_number):
            return False

    # Filter based on elements
    if filter_elements:
        for element in filter_elements:
            if element in row[COLUMN_ELEMENTS]:
                return False

    # Filter based on structures
    if filter_structures:
        if "aromatic" in filter_structures and row[COLUMN_IS_AROMATIC]:
            return False
        if "non-cyclic" in filter_structures and row[COLUMN_IS_NON_CYCLIC]:
            return False
        if (
            "cyclic non-aromatic" in filter_structures
            and row[COLUMN_IS_CYCLIC_NON_AROMATIC]
        ):
            return False

    return True


def parallel_apply(
    df: pd.DataFrame, func: Callable[[pd.Series, Any], bool], *args
) -> pd.DataFrame:
    global removed_indices_condition

    with Pool(cpu_count()) as pool:
        result = pool.starmap(func, [(row, *args) for _, row in df.iterrows()])

    return df[result]


@dataclass
class Args:
    analysis_file: str
    min_atomic_number: int
    max_atomic_number: int
    size_count_threshold: int
    elemental_count_threshold: int
    filter_elements: list[str]
    filter_structures: list[str]
    filtered_filename: str
    index_column_name: str


# parallel = True
parallel = False


def main(args: Args):
    global removed_indices_condition

    analysis_file = pt(args.analysis_file)

    analysis_df = pd.read_csv(analysis_file)
    analysis_df["ElementCategories"] = analysis_df["ElementCategories"].apply(
        lambda x: Counter(json.loads(x))
    )
    analysis_df["Elements"] = analysis_df["Elements"].apply(
        lambda x: Counter(json.loads(x))
    )

    logger.info(f"Original DataFrame length: {len(analysis_df)}")
    logger.info(f"Min atomic number: {args.min_atomic_number}")
    logger.info(f"Max atomic number: {args.max_atomic_number}")
    logger.info(f"Filter elements: {args.filter_elements}")
    logger.info(f"Filter structures: {args.filter_structures}")
    logger.info(f"Elemental count threshold: {args.elemental_count_threshold}")
    logger.info(f"Atomic size count threshold: {args.size_count_threshold}")

    if args.elemental_count_threshold:
        logger.info("Filtering based on element count threshold")
        # Filter based on element count threshold
        elements = Counter()
        for e in analysis_df["Elements"]:
            elements.update(e)

        elements_containing = Counter()
        for element in elements.keys():
            elements_containing[element] = (
                analysis_df["Elements"].apply(lambda x: element in x).sum()
            )

        include_elements = {
            key: count
            for key, count in elements_containing.items()
            if count > int(args.elemental_count_threshold)
        }
        include_elements_keys = set(include_elements.keys())
        all_elements = set(elements_containing.keys())
        filter_elements = all_elements - include_elements_keys

        args.filter_elements = filter_elements.update(args.filter_elements)
        logger.info(
            f"Filtering out elements: {filter_elements} based on threshold count"
        )

    # filter based on atomic size threshold
    if args.size_count_threshold:
        logger.info("Filtering based on atomic size count threshold")
        atoms_distribution = Counter(analysis_df["No. of atoms"].values)
        atoms_distribution_df = pd.DataFrame.from_dict(
            atoms_distribution, orient="index"
        ).reset_index()
        atoms_distribution_df.columns = ["No. of atoms", "Count"]

        # filter by atomic size count threshold
        atoms_distribution_df = atoms_distribution_df[
            atoms_distribution_df["Count"] > int(args.size_count_threshold)
        ]
        no_of_atoms = set(atoms_distribution_df["No. of atoms"].values)
        analysis_df = analysis_df[analysis_df["No. of atoms"].isin(no_of_atoms)]
        logger.info(
            f"Filtering out atomic sizes: {no_of_atoms} based on threshold count"
        )
        logger.info(
            f"Filtered atomic size threshold DataFrame length: {len(analysis_df)}"
        )

    if parallel:
        logger.info("Using parallel processing")
        final_analysis_df = parallel_apply(
            analysis_df,
            apply_filters_to_df,
            args.min_atomic_number,
            args.max_atomic_number,
            args.filter_elements,
            args.filter_structures,
        )
    else:
        logger.info("Using single processing")
        removed_indices_condition = analysis_df.apply(
            apply_filters_to_df,
            axis=1,
            args=(
                args.min_atomic_number,
                args.max_atomic_number,
                args.filter_elements,
                args.filter_structures,
            ),
        )
        final_analysis_df: pd.DataFrame = analysis_df[removed_indices_condition]

    final_analysis_df = final_analysis_df.set_index(args.index_column_name)
    # final_analysis_df = final_analysis_df.dropna()  # Drop rows that were filtered out
    logger.info(f"Filtered DataFrame length: {len(final_analysis_df)}")

    final_analysis_df["ElementCategories"] = final_analysis_df[
        "ElementCategories"
    ].apply(lambda x: json.dumps(dict(x)))
    final_analysis_df["Elements"] = final_analysis_df["Elements"].apply(
        lambda x: json.dumps(dict(x))
    )

    analysis_dir = analysis_file.parent
    metadata_file = analysis_dir / "metadata.json"
    data = json.loads(metadata_file.read_text())
    logger.info(data)

    filename = pt(data["filename"])
    filtered_data_filename = (
        f"{filename.stem}_{args.filtered_filename.lower()}_filtered"
    )

    filtered_dir = analysis_file.parent / "filtered" / args.filtered_filename
    if not filtered_dir.exists():
        filtered_dir.mkdir(parents=True)

    filtered_analysis_dir = filtered_dir / f"{filtered_data_filename}_analysis"
    filtered_file_path: pt = filtered_analysis_dir / "molecule_analysis_results.csv"
    logger.info(f"Filtered file path: {filtered_file_path}")
    if not filtered_file_path.parent.exists():
        filtered_file_path.parent.mkdir(parents=True)
    final_analysis_df.to_csv(filtered_file_path)

    training_df: pd.DataFrame = read_as_ddf(
        data["filetype"],
        filename,
        data["key"],
    )
    index_column_name = data["index_column_name"]
    training_df = training_df.set_index(index_column_name)
    logger.info(
        f"Index name: {training_df.index.name}\n{training_df.index.values[:10]=}\n{training_df.columns=}"
    )
    logger.info(f"{training_df.head()=}")
    final_training_df = training_df.loc[training_df.index.isin(final_analysis_df.index)]
    final_filtered_training_df_file = (
        filtered_dir / f"{filtered_data_filename}_training.csv"
    )
    final_training_df.to_csv(final_filtered_training_df_file)
    logger.info(
        f"Final filtered training data saved at {final_filtered_training_df_file}"
    )

    return {"filtered_file": str(filtered_file_path)}
