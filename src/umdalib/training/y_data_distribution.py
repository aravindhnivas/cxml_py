import json
from dataclasses import dataclass
from pathlib import Path as pt
import numpy as np
import pandas as pd
from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    use_dask: bool
    column_name: str
    training_save_directory: str
    bin_size: int
    auto_bin_size: bool


def main(args: Args):
    logger.info(f"{args.column_name=}")
    training_save_directory = pt(args.training_save_directory)

    df = read_as_ddf(
        args.filetype,
        args.filename,
        args.key,
        use_dask=args.use_dask,
        computed=True,
    )

    y: pd.Series = df[args.column_name]

    if args.auto_bin_size:
        n = len(y)
        bin_size = int(np.ceil(np.sqrt(n)))
        logger.info(f"Auto bin size: {args.bin_size}")
    else:
        bin_size = int(args.bin_size)

    # Compute histogram data
    hist, bin_edges = np.histogram(y, bins=bin_size)

    histogram_data = {
        "hist": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
    }

    # Save histogram data to JSON file
    filename = training_save_directory / "histogram_data.json"
    with open(filename, "w") as f:
        json.dump(histogram_data, f, indent=2)
        logger.info(f"Saved histogram data to {filename}")

    return {
        "filename": str(filename),
        "bin_size": bin_size,
        "min": y.min(),
        "max": y.max(),
    }
