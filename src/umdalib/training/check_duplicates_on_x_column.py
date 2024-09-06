from dataclasses import dataclass
from pathlib import Path as pt

import pandas as pd

from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    use_dask: bool
    smiles_column_name: str


def drop_duplicates_on_x_column(df: pd.DataFrame, column: str):
    """
    Drops duplicates on a given column and prints the number of duplicates and the indices of the duplicates.
    """

    column_lower = f"{column}_lower"

    # Create a temporary column with lowercase SMILES
    df[column_lower] = df[column].str.lower()

    # Get the original length
    original_length = len(df)

    # Remove duplicates and get the new dataframe
    df_deduplicated = df.drop_duplicates(subset=[column_lower])

    # Get the new length
    new_length = len(df_deduplicated)

    # Get the indices of dropped rows
    dropped_indices = df.index.difference(df_deduplicated.index)

    # Drop the temporary column
    df_deduplicated = df_deduplicated.drop(columns=[column_lower])

    logger.info(f"Number of dropped SMILES: {original_length - new_length}")
    logger.info(f"Indices of dropped SMILES: {list(dropped_indices)}")

    # Update the original dataframe
    df = df_deduplicated

    return df, dropped_indices


def main(args: Args):
    logger.info(f"Checking duplicates on {args.filename}")
    logger.info(f"Using Dask: {args.use_dask}")

    df = read_as_ddf(
        args.filetype,
        args.filename,
        args.key,
        use_dask=args.use_dask,
        computed=True,
    )
    logger.info(f"{df.columns=}")
    deduplicated_df, dropped_indices = drop_duplicates_on_x_column(
        df.copy(), args.smiles_column_name
    )

    training_filename = pt(args.filename)
    deduplicated_filename = (
        training_filename.parent / f"[FIXED-DUPLICATES]_{training_filename.stem}.csv"
    )
    duplicated_filename = (
        training_filename.parent / f"[DUPLICATES]_{training_filename.stem}.csv"
    )

    if dropped_indices.size > 0:
        deduplicated_df.to_csv(
            deduplicated_filename, index=True, index_label="OriginalIndex"
        )
        logger.info(f"Saved deduplicated data to {deduplicated_filename}")
        # duplicated_df = df.loc[dropped_indices]
        # duplicated_df.to_csv(duplicated_filename, index=True, index_label='OriginalIndex')
        # logger.info(f"Saved duplicated data to {duplicated_filename}")
    else:
        logger.info("No duplicates found")

    return {
        "deduplicated_filename": deduplicated_filename,
        "duplicated_filename": duplicated_filename,
        "duplicates": len(dropped_indices),
    }
