from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Dict, Union, Optional, Any
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from cxml_lib.logger import logger

NPARTITIONS: int = cpu_count() * 5


def read_as_ddf(
    filetype: str,
    filename: str,
    key: Optional[str] = None,
    computed: bool = False,
    use_dask: bool = False,
) -> Union[dd.DataFrame, pd.DataFrame]:
    """
    Read a file into a Dask or Pandas DataFrame based on the file type.

    Args:
        filetype: Type of the file (smi, csv, parquet, hdf, json)
        filename: Path to the file
        key: Key for HDF files
        computed: Whether to compute the Dask DataFrame immediately
        use_dask: Whether to use Dask instead of Pandas

    Returns:
        Union[dd.DataFrame, pd.DataFrame]: The loaded DataFrame

    Raises:
        ValueError: If filetype is unknown or key is missing for HDF files
        FileNotFoundError: If the file doesn't exist
    """
    logger.info(f"Reading {filename} as {filetype} using dask: {use_dask}")

    if not filetype:
        filetype = filename.split(".")[-1]
        logger.info(f"{filetype=}")

    df_fn = dd if use_dask else pd
    logger.info(f"Using {'Dask' if use_dask else 'Pandas'}: {df_fn=}")

    try:
        if filetype == "smi":
            data = np.loadtxt(filename, dtype=str, ndmin=2)
            if data[0][0].lower() == "smiles":
                data = data[1:]

            ddf = pd.DataFrame(data, columns=["SMILES"])
            if use_dask:
                ddf = dd.from_pandas(ddf, npartitions=NPARTITIONS)

        elif filetype == "csv":
            ddf = df_fn.read_csv(filename)
        elif filetype == "parquet":
            ddf = df_fn.read_parquet(filename)
        elif filetype == "hdf":
            if not key:
                raise ValueError("Key is required for HDF files")
            ddf = df_fn.read_hdf(filename, key)
        elif filetype == "json":
            ddf = df_fn.read_json(filename)
        else:
            raise ValueError(f"Unknown filetype: {filetype}")

        if computed and use_dask:
            with ProgressBar():
                ddf = ddf.compute()

        logger.info(f"{type(ddf)=}")
        return ddf

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        raise


@dataclass
class Args:
    """Arguments for the main function."""

    filename: str
    filetype: str
    key: str
    rows: Dict[str, Union[int, str]]
    use_dask: bool


def main(args: Args) -> Dict[str, Any]:
    """
    Main function to read and process data files.

    Args:
        args: Args object containing file information and processing options

    Returns:
        Dict[str, Any]: Dictionary containing processed data information

    Raises:
        ValueError: If there are issues with the input arguments
    """
    logger.info(f"Reading {args.filename} as {args.filetype}")
    logger.info(f"Using Dask: {args.use_dask}")

    ddf = read_as_ddf(args.filetype, args.filename, args.key, use_dask=args.use_dask)
    logger.info(f"{type(ddf)=}")

    shape = ddf.shape[0]
    if args.use_dask:
        shape = shape.compute()
    logger.info(f"read_data file: Shape: {shape}")

    data: Dict[str, Any] = {
        "columns": ddf.columns.values.tolist(),
    }

    try:
        count = int(args.rows["value"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid rows value: {e}")

    with ProgressBar():
        if args.rows["where"] == "head":
            nrows = ddf.head(count).fillna("")
        elif args.rows["where"] == "tail":
            nrows = ddf.tail(count).fillna("")
        else:
            raise ValueError(f"Invalid 'where' value: {args.rows['where']}")

        data["nrows"] = nrows.to_dict(orient="records")
        data["shape"] = shape
        data["index_name"] = ddf.index.name

    logger.info(f"{type(data)=}")
    return data
