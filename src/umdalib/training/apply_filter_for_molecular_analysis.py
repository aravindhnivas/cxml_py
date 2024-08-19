from dataclasses import dataclass
import json
import pandas as pd
from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path as pt
from typing import Any, Callable, Optional

# Constants for column names
COLUMN_ATOMS = "No. of atoms"
COLUMN_ELEMENTS = "Elements"
COLUMN_IS_AROMATIC = "IsAromatic"
COLUMN_IS_NON_CYCLIC = "IsNonCyclic"
COLUMN_IS_CYCLIC_NON_AROMATIC = "IsCyclicNonAromatic"


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


# parallel = True
parallel = False


def main(args: Args):
    # logger.info(args.min_atomic_number)

    analysis_file = pt(args.analysis_file)
    df = pd.read_csv(analysis_file)
    df["ElementCategories"] = df["ElementCategories"].apply(
        lambda x: Counter(json.loads(x))
    )
    df["Elements"] = df["Elements"].apply(lambda x: Counter(json.loads(x)))

    logger.info(f"Original DataFrame length: {len(df)}")
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
        for e in df["Elements"]:
            elements.update(e)

        elements_containing = Counter()
        for element in elements.keys():
            elements_containing[element] = (
                df["Elements"].apply(lambda x: element in x).sum()
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
        atoms_distribution = Counter(df["No. of atoms"].values)
        atoms_distribution_df = pd.DataFrame.from_dict(
            atoms_distribution, orient="index"
        ).reset_index()
        atoms_distribution_df.columns = ["No. of atoms", "Count"]

        # filter by atomic size count threshold
        atoms_distribution_df = atoms_distribution_df[
            atoms_distribution_df["Count"] > int(args.size_count_threshold)
        ]
        no_of_atoms = set(atoms_distribution_df["No. of atoms"].values)
        df = df[df["No. of atoms"].isin(no_of_atoms)]
        logger.info(
            f"Filtering out atomic sizes: {no_of_atoms} based on threshold count"
        )
        logger.info(f"Filtered atomic size threshold DataFrame length: {len(df)}")

    if parallel:
        logger.info("Using parallel processing")
        final_df = parallel_apply(
            df,
            apply_filters_to_df,
            args.min_atomic_number,
            args.max_atomic_number,
            args.filter_elements,
            args.filter_structures,
        )
    else:
        logger.info("Using single processing")
        final_df: pd.DataFrame = df[
            df.apply(
                apply_filters_to_df,
                axis=1,
                args=(
                    args.min_atomic_number,
                    args.max_atomic_number,
                    args.filter_elements,
                    args.filter_structures,
                ),
            )
        ]

    # final_df = final_df.dropna()  # Drop rows that were filtered out
    logger.info(f"Filtered DataFrame length: {len(final_df)}")
    filtered_file_path = (
        analysis_file.parent / "filtered_" / f"{analysis_file.stem}_filtered.csv"
    )
    if not filtered_file_path.parent.exists():
        filtered_file_path.parent.mkdir(parents=True)
    final_df.to_csv(filtered_file_path)

    return {"filtered_file": str(filtered_file_path)}
