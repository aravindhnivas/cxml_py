from dataclasses import dataclass
import json
import pandas as pd
from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path as pt
from typing import Optional

# Constants for column names
COLUMN_ATOMS = "No. of atoms"
COLUMN_ELEMENTS = "Elements"
COLUMN_IS_AROMATIC = "IsAromatic"
COLUMN_IS_NON_CYCLIC = "IsNonCyclic"
COLUMN_IS_CYCLIC_NON_AROMATIC = "IsCyclicNonAromatic"


def apply_filters_to_df(
    x: pd.Series,
    min_atomic_number: Optional[int],
    max_atomic_number: Optional[int],
    filter_elements: list[str],
    filter_structures: list[str],
) -> Optional[pd.Series]:

    # Filter based on atomic number
    if min_atomic_number is not None and x[COLUMN_ATOMS] < int(min_atomic_number):
        return None
    if max_atomic_number is not None and x[COLUMN_ATOMS] > int(max_atomic_number):
        return None

    # Filter based on elements
    if filter_elements:
        if any(key in x[COLUMN_ELEMENTS] for key in filter_elements):
            return None

    # Filter based on structures
    if filter_structures:
        if "aromatic" in filter_structures and x[COLUMN_IS_AROMATIC]:
            return None
        if "non-cyclic" in filter_structures and x[COLUMN_IS_NON_CYCLIC]:
            return None
        if (
            "cyclic non-aromatic" in filter_structures
            and x[COLUMN_IS_CYCLIC_NON_AROMATIC]
        ):
            return None

    return x


def parallel_apply(df: pd.DataFrame, func, *args) -> pd.DataFrame:
    with Pool(cpu_count()) as pool:
        result = pool.starmap(func, [(row, *args) for _, row in df.iterrows()])
    return pd.DataFrame([r for r in result if r is not None])


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

    if not any(
        [
            args.min_atomic_number,
            args.max_atomic_number,
            args.filter_elements,
            args.filter_structures,
        ]
    ):
        logger.info("No filters applied")
        return {"filtered_file": str(analysis_file)}

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
        final_df: pd.DataFrame = df.apply(
            apply_filters_to_df,
            axis=1,
            args=(
                args.min_atomic_number,
                args.max_atomic_number,
                args.filter_elements,
                args.filter_structures,
            ),
        )
    final_df = final_df.dropna()  # Drop rows that were filtered out
    logger.info(f"Filtered DataFrame length: {len(final_df)}")
    filtered_file_path = (
        analysis_file.parent / "filtered_" / f"{analysis_file.stem}_filtered.csv"
    )
    if not filtered_file_path.parent.exists():
        filtered_file_path.parent.mkdir(parents=True)
    final_df.to_csv(filtered_file_path)

    return {"filtered_file": str(filtered_file_path)}
