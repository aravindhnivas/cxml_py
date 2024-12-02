from dataclasses import dataclass
from umdalib.logger import logger
from pathlib import Path as pt
import pandas as pd
import numpy as np


@dataclass
class Args:
    metrics_loc: str


def main(args: Args):
    metrics_loc = pt(args.metrics_loc)
    logger.info(f"Exporting all metrics from {metrics_loc}")

    csv_files = [
        m.name
        for m in metrics_loc.iterdir()
        if m.name.endswith(".csv") and m.name != "all_metrics.csv"
    ]
    logger.info(f"Found {len(csv_files)} csv files")

    if len(csv_files) == 0:
        logger.error("No csv files found")
        return {
            "csv_files": csv_files,
            "metrics_final_csv": None,
        }

    metrics_final_csv = metrics_loc / "all_metrics.csv"
    metrics_df: pd.DataFrame = None

    for csv in csv_files:
        model_name = csv.split("_")[0]

        # add a new column for model name
        if metrics_df is None:
            metrics_df = pd.read_csv(metrics_loc / csv)
            models_name_col = np.array([model_name] * metrics_df.shape[0])
            metrics_df["model"] = models_name_col
            metrics_df = metrics_df[
                ["model"] + [col for col in metrics_df.columns if col != "model"]
            ]
        else:
            metrics_df_ = pd.read_csv(metrics_loc / csv)
            models_name_col = np.array([model_name] * metrics_df_.shape[0])
            metrics_df_["model"] = models_name_col
            metrics_df_ = metrics_df_[
                ["model"] + [col for col in metrics_df_.columns if col != "model"]
            ]
            metrics_df = pd.concat([metrics_df, metrics_df_], axis=0)

    metrics_df = metrics_df.reset_index(drop=True)
    # drop NaN rows
    metrics_df = metrics_df.dropna()

    metrics_df.to_csv(metrics_final_csv, index=False)
    logger.success("Saved metrics.csv")

    return {
        "csv_files": csv_files,
        "metrics_final_csv": metrics_final_csv,
    }
