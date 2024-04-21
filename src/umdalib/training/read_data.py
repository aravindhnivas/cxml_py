from dataclasses import dataclass
import dask.dataframe as dd

# from loguru import logger
from umdalib.utils import logger
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count

NPARTITIONS = cpu_count() * 5

# pbar = ProgressBar()
# pbar.register()


@dataclass
class Args:
    filename: str
    filetype: str
    key: str


def main(args: Args):
    print(f"Reading {args.filename} as {args.filetype}")
    if args.filetype == "csv":
        df = dd.read_csv(args.filename)
    elif args.filetype == "parquet":
        df = dd.read_parquet(args.filename)
    elif args.filetype == "hdf":
        df = dd.read_hdf(args.filename, args.key)
    elif args.filetype == "json":
        df = dd.read_json(args.filename)
    else:
        raise ValueError(f"Unknown filetype: {args.filetype}")

    logger.info(f"read_data file: Shape: {df.shape[0].compute()}")
    # if df.columns[0] == "Unnamed: 0":
    #     df = df.drop(columns=["Unnamed: 0"])

    data = {
        "columns": df.columns.values.tolist(),
    }

    with ProgressBar():
        head = df.head(10)
        data["head"] = head.to_dict(orient="records")
        data["shape"] = df.shape[0].compute()

    return data
