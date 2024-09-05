from dataclasses import dataclass

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
        computed=args.use_dask,
    )
    deduplicated_df, dropped_indices = drop_duplicates_on_x_column(
        df, args.smiles_column_name
    )

    deduplicated_filename = args.filename.replace(".csv", "_deduplicated.csv")
    deduplicated_df.to_csv(deduplicated_filename, index=False)
    logger.info(f"Saved deduplicated data to {deduplicated_filename}")

    return {
        "duplicates": len(dropped_indices),
        "dropped_indices": dropped_indices.values,
    }
