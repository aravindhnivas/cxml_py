from dataclasses import dataclass
from typing import Literal
from cxml_lib.logger import logger
from pathlib import Path as pt
import pandas as pd
import numpy as np
from uncertainties import ufloat, ufloat_fromstr


@dataclass
class Args:
    metrics_loc: str


def parse_metric_with_uncertainty(value: str):
    """Parse metrics in format like '0.3(2)' to ufloat."""
    if isinstance(value, str):
        # Handle cases where the uncertainty is in parentheses
        try:
            # Try parsing as a direct ufloat string
            return ufloat_fromstr(value)
        except Exception as e:
            print(e)
            # If that fails, try manual parsing
            parts = value.replace("(", " ").replace(")", "").split()
            if len(parts) == 2:
                nominal = float(parts[0])
                # Convert uncertainty to the same decimal places as nominal
                uncertainty = float(parts[1]) * 10 ** (-len(parts[0].split(".")[-1]))
                return ufloat(nominal, uncertainty)
    return ufloat(float(value), 0)


def get_best_metrics(df: pd.DataFrame, unique_name: Literal["model", "Embedder"]):
    """Get the best metrics for each model."""

    other_name = "model" if unique_name == "Embedder" else "Embedder"

    model_performance = []
    for model_type in df[unique_name].unique():
        model_data = df[df[unique_name] == model_type]

        # For R2 (highest value)
        best_r2_idx = model_data["R2_value"].apply(lambda x: x.nominal_value).idxmax()
        best_r2_row = model_data.loc[best_r2_idx]

        # For MSE (lowest value)
        best_mse_idx = model_data["MSE_value"].apply(lambda x: x.nominal_value).idxmin()
        best_mse_row = model_data.loc[best_mse_idx]

        # For RMSE (lowest value)
        best_rmse_idx = (
            model_data["RMSE_value"].apply(lambda x: x.nominal_value).idxmin()
        )
        best_rmse_row = model_data.loc[best_rmse_idx]

        # For MAE (lowest value)
        best_mae_idx = model_data["MAE_value"].apply(lambda x: x.nominal_value).idxmin()
        best_mae_row = model_data.loc[best_mae_idx]

        row_data = {
            unique_name: model_type,
            "best_R2": best_r2_row["R2"],
            "best_R2_mode": best_r2_row["Mode"],
            f"best_R2_{other_name}": best_r2_row[other_name],
            "best_MSE": best_mse_row["MSE"],
            "best_MSE_mode": best_mse_row["Mode"],
            f"best_MSE_{other_name}": best_mse_row[other_name],
            "best_RMSE": best_rmse_row["RMSE"],
            "best_RMSE_mode": best_rmse_row["Mode"],
            f"best_RMSE_{other_name}": best_rmse_row[other_name],
            "best_MAE": best_mae_row["MAE"],
            "best_MAE_mode": best_mae_row["Mode"],
            f"best_MAE_{other_name}": best_mae_row[other_name],
        }
        model_performance.append(row_data)

    model_performance_df = pd.DataFrame(model_performance)

    return model_performance_df


def analyze_best_metrics(df: pd.DataFrame):
    """Analyze and find best performing models across different metrics."""
    # Convert metrics to ufloat values
    metrics = ["R2", "MSE", "RMSE", "MAE"]
    for metric in metrics:
        df[f"{metric}_value"] = df[metric].apply(parse_metric_with_uncertainty)

    best_models = {}
    for metric in metrics:
        # Sort by nominal value
        if metric == "R2":
            # For R2, higher is better, and we want to access the nominal value directly
            sorted_df = df.sort_values(
                by=f"{metric}_value",
                key=lambda x: [v.nominal_value for v in x],
                ascending=False,
            )
        else:
            # For MSE, RMSE, MAE lower is better, and we want to access the nominal value directly
            sorted_df = df.sort_values(
                by=f"{metric}_value",
                key=lambda x: [v.nominal_value for v in x],
                ascending=True,
            )
        best_models[metric] = sorted_df.head(5)[["model", "Mode", "Embedder", metric]]

    model_performance_df = get_best_metrics(df, "model")
    embedder_performance_df = get_best_metrics(df, "Embedder")

    # Model-embedder performance: For each model, find the best performing metric for each embedder
    # Columns are: model -> embedder -> best_R2, best_MSE, best_RMSE, best_MAE
    model_embedder_performance = []
    for model_type in df["model"].unique():
        for embedder in df["Embedder"].unique():
            data: pd.DataFrame = df[
                (df["model"] == model_type) & (df["Embedder"] == embedder)
            ]

            if len(data) > 0:
                best_r2_idx = data["R2_value"].apply(lambda x: x.nominal_value).idxmax()
                best_mse_idx = (
                    data["MSE_value"].apply(lambda x: x.nominal_value).idxmin()
                )
                best_rmse_idx = (
                    data["RMSE_value"].apply(lambda x: x.nominal_value).idxmin()
                )
                best_mae_idx = (
                    data["MAE_value"].apply(lambda x: x.nominal_value).idxmin()
                )

                row_data = {
                    "model": model_type,
                    "embedder": embedder,
                    "best_R2": data.loc[best_r2_idx, "R2"],
                    "R2_mode": data.loc[best_r2_idx, "Mode"],
                    "best_MSE": data.loc[best_mse_idx, "MSE"],
                    "MSE_mode": data.loc[best_mse_idx, "Mode"],
                    "best_RMSE": data.loc[best_rmse_idx, "RMSE"],
                    "RMSE_mode": data.loc[best_rmse_idx, "Mode"],
                    "best_MAE": data.loc[best_mae_idx, "MAE"],
                    "MAE_mode": data.loc[best_mae_idx, "Mode"],
                }
                model_embedder_performance.append(row_data)

    model_embedder_performance_df = pd.DataFrame(model_embedder_performance)

    return {
        "best_models": best_models,
        "model_performance_df": model_performance_df,
        "embedder_performance_df": embedder_performance_df,
        "model_embedder_performance_df": model_embedder_performance_df,
    }


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

    metrics_df = metrics_df.dropna()
    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df.to_csv(metrics_final_csv)
    logger.success(
        f"Saved metrics to {metrics_final_csv}. Total rows: {metrics_df.shape[0]}"
    )

    best_metrics_results: dict[str, pd.DataFrame] = analyze_best_metrics(metrics_df)

    best_metrics_loc = metrics_loc / "best_metrics"
    best_metrics_loc.mkdir(exist_ok=True)

    best_metrics_results["best_models"]["R2"].to_csv(
        best_metrics_loc / "best_models_R2.csv", index=False
    )
    best_metrics_results["best_models"]["MSE"].to_csv(
        best_metrics_loc / "best_models_MSE.csv", index=False
    )
    best_metrics_results["best_models"]["RMSE"].to_csv(
        best_metrics_loc / "best_models_RMSE.csv", index=False
    )
    best_metrics_results["best_models"]["MAE"].to_csv(
        best_metrics_loc / "best_models_MAE.csv", index=False
    )

    best_metrics_results["model_performance_df"].to_csv(
        best_metrics_loc / "model_performance.csv", index=False
    )

    best_metrics_results["embedder_performance_df"].to_csv(
        best_metrics_loc / "embedder_performance.csv", index=False
    )

    best_metrics_results["model_embedder_performance_df"].to_csv(
        best_metrics_loc / "model_embedder_performance.csv", index=False
    )

    return {
        "csv_files": csv_files,
        "metrics_final_csv": metrics_final_csv,
    }
