from dataclasses import dataclass
import dask.dataframe as dd
import pandas as pd
from umdalib.utils import logger
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count
from typing import Dict, Union

NPARTITIONS = cpu_count() * 5


def read_as_ddf(
    filetype: str, filename: str, key: str = None, computed=False, use_dask=False
):

    logger.info(f"Reading {filename} as {filetype} using dask: {use_dask}")

    df_fn = None
    if use_dask:
        df_fn = dd
        logger.info(f"Using Dask: {df_fn=}")
    else:
        df_fn = pd
        logger.info(f"Using Pandas: {df_fn=}")

    # df_fn = dd

    ddf: Union[dd.DataFrame, pd.DataFrame] = None

    if filetype == "smi":
        ddf = df_fn.read_csv(filename)
        logger.info(f"Columns in the DataFrame: {ddf.columns.tolist()}")
        if ddf.columns[0].lower() != "smiles":
            ddf.columns = ["SMILES"]
    elif filetype == "csv":
        ddf = df_fn.read_csv(filename)
    elif filetype == "parquet":
        ddf = df_fn.read_parquet(filename)
    elif filetype == "hdf":
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


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    rows: Dict[str, Union[int, str]]
    use_dask: bool


def main(args: Args):

    logger.info(f"Reading {args.filename} as {args.filetype}")
    logger.info(f"Using Dask: {args.use_dask}")

    ddf = read_as_ddf(args.filetype, args.filename, args.key, use_dask=args.use_dask)
    logger.info(f"{type(ddf)=}")

    shape = ddf.shape[0]
    if args.use_dask:
        shape = shape.compute()
    logger.info(f"read_data file: Shape: {shape}")

    data = {
        "columns": ddf.columns.values.tolist(),
    }

    count = int(args.rows["value"])

    with ProgressBar():
        if args.rows["where"] == "head":
            nrows = ddf.head(count).fillna("")
        elif args.rows["where"] == "tail":
            nrows = ddf.tail(count).fillna("")
        data["nrows"] = nrows.to_dict(orient="records")
        data["shape"] = shape

    logger.info(f"{type(data)=}")

    return data
