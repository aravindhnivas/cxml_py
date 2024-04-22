from dataclasses import dataclass
import dask.dataframe as dd

# from loguru import logger
from umdalib.utils import logger
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count

NPARTITIONS = cpu_count() * 5

# pbar = ProgressBar()
# pbar.register()


def read_as_ddf(filetype: str, filename: str, key: str = None, computed=False):
    ddf = None
    if filetype == "smi":
        ddf = dd.read_csv(filename, header=None, names=["SMILES"])
    elif filetype == "csv":
        ddf = dd.read_csv(filename)
    elif filetype == "parquet":
        ddf = dd.read_parquet(filename)
    elif filetype == "hdf":
        ddf = dd.read_hdf(filename, key)
    elif filetype == "json":
        ddf = dd.read_json(filename)
    else:
        raise ValueError(f"Unknown filetype: {filetype}")

    if computed:
        with ProgressBar():
            ddf = ddf.compute()
    return ddf


@dataclass
class Args:
    filename: str
    filetype: str
    key: str


def main(args: Args):
    print(f"Reading {args.filename} as {args.filetype}")

    ddf = read_as_ddf(args.filetype, args.filename, args.key)

    logger.info(f"read_data file: Shape: {ddf.shape[0].compute()}")
    # if ddf.columns[0] == "Unnamed: 0":
    #     ddf = ddf.drop(columns=["Unnamed: 0"])

    data = {
        "columns": ddf.columns.values.tolist(),
    }

    with ProgressBar():
        head = ddf.head(10)
        data["head"] = head.to_dict(orient="records")
        data["shape"] = ddf.shape[0].compute()

    return data
