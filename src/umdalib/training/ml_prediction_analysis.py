from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import TypedDict
from umdalib.utils import logger
from umdalib.training.read_data import read_as_ddf


class AnalysisFile(TypedDict):
    filename: str
    filetype: str
    key: str


@dataclass
class Args:
    analysis_file: AnalysisFile
    columnX: str
    columnY: str
    use_dask: bool


def linear_fit(x, m, c):
    return m * x + c


def main(args: Args):

    df = read_as_ddf(
        args.analysis_file["filetype"],
        args.analysis_file["filename"],
        args.analysis_file["key"],
        use_dask=args.use_dask,
        computed=args.use_dask,
    )

    y_true = df[args.columnX].values
    y_pred = df[args.columnY].values

    popt, _ = curve_fit(linear_fit, y_true, y_pred)
    y_fit = linear_fit(y_true, *popt)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "popt": popt,
        "y_fit": y_fit,
        "y_true": y_true,
        "y_pred": y_pred,
    }
