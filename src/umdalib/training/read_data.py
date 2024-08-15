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

    df_fn = dd if use_dask else pd

    ddf = None
    if filename.endswith(".smi"):
        ddf = df_fn.read_csv(filename, header=None, names=["SMILES"])
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
    return ddf


# use_dask = False
# df_fn: Union[dd.DataFrame, pd.DataFrame] = pd


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    rows: Dict[str, Union[int, str]]
    use_dask: bool


def main(args: Args):
    # global use_dask, df_fn

    if args.use_dask:
        logger.info("Using Dask")
    else:
        logger.warning("Not using Dask")

    logger.info(f"Reading {args.filename} as {args.filetype}")

    ddf = read_as_ddf(args.filetype, args.filename, args.key, args.use_dask)
    shape = ddf.shape[0]
    if args.use_dask:
        shape = shape.compute()
    logger.info(f"read_data file: Shape: {shape}")

    # if ddf.columns[0] == "Unnamed: 0":
    #     ddf = ddf.drop(columns=["Unnamed: 0"])

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
