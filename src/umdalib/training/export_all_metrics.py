from dataclasses import dataclass
from umdalib.logger import logger
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


def analyze_best_metrics(df: pd.DataFrame):
    """Analyze and find best performing models across different metrics."""
    # Convert metrics to ufloat values
    metrics = ["R2", "MSE", "RMSE", "MAE"]
    for metric in metrics:
        df[f"{metric}_value"] = df[metric].apply(ufloat_fromstr)

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
        best_models[metric] = sorted_df.head(5)[
            ["model", "Mode", "Embedder", metric, f"{metric}_value"]
        ]

    # Analyze by model type
    model_performance = {}
    for model_type in df["model"].unique():
        model_data = df[df["model"] == model_type]
        model_performance[model_type] = {
            "best_R2": max(model_data["R2_value"], key=lambda x: x.nominal_value),
            "best_MSE": min(model_data["MSE_value"], key=lambda x: x.nominal_value),
            "best_RMSE": min(model_data["RMSE_value"], key=lambda x: x.nominal_value),
            "best_MAE": min(model_data["MAE_value"], key=lambda x: x.nominal_value),
        }

    # Analyze by embedder
    embedder_performance = {}
    for embedder in df["Embedder"].unique():
        embedder_data = df[df["Embedder"] == embedder]
        embedder_performance[embedder] = {
            "best_R2": max(embedder_data["R2_value"], key=lambda x: x.nominal_value),
            "best_MSE": min(embedder_data["MSE_value"], key=lambda x: x.nominal_value),
            "best_RMSE": min(
                embedder_data["RMSE_value"], key=lambda x: x.nominal_value
            ),
            "best_MAE": min(embedder_data["MAE_value"], key=lambda x: x.nominal_value),
        }

    return {
        "best_models": best_models,
        "model_performance": model_performance,
        "embedder_performance": embedder_performance,
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

    best_metrics_results = analyze_best_metrics(metrics_df)
    logger.info("Best metrics analysis:")
    for metric, best_models in best_metrics_results["best_models"].items():
        logger.info(f"Best models for {metric}:")
        logger.info(best_models)

    logger.info("Model performance analysis:")
    for model_type, performance in best_metrics_results["model_performance"].items():
        logger.info(f"Performance for {model_type}:")
        logger.info(performance)

    logger.info("Embedder performance analysis:")
    for embedder, performance in best_metrics_results["embedder_performance"].items():
        logger.info(f"Performance for {embedder}:")
        logger.info(performance)

    best_metrics_loc = metrics_loc / "best_metrics"
    best_metrics_loc.mkdir(exist_ok=True)

    # save best metrics, model and embedder performance to csv
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

    pd.DataFrame(best_metrics_results["model_performance"]).T.to_csv(
        best_metrics_loc / "model_performance.csv"
    )

    pd.DataFrame(best_metrics_results["embedder_performance"]).T.to_csv(
        best_metrics_loc / "embedder_performance.csv"
    )

    return {
        "csv_files": csv_files,
        "metrics_final_csv": metrics_final_csv,
    }
