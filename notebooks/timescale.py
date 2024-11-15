import time
from pathlib import Path as pt

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd


def time_model_training(
    model: BaseEstimator, X_train, y_train, n_runs: int = 5
) -> tuple:
    """
    Time the training of a model multiple times and return statistics.

    Parameters:
    -----------
    model : sklearn-like estimator
        The model to train
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    n_runs : int, default=5
        Number of times to run the training

    Returns:
    --------
    tuple : (mean_time, std_time, all_times)
    """
    times = []
    start = time.perf_counter()

    for _ in range(n_runs):
        # Clone the model to ensure fresh instance each time
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        end = time.perf_counter()
        t = end - start
        times.append(t)

    return np.mean(times), np.std(times), times


# Dictionary of models to test
models = {
    "GBR": GradientBoostingRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "XGBoost": XGBRegressor(verbosity=0),
    "LightGBM": LGBMRegressor(verbose=-1),
}


def compute(embeddings="mol2vec_embeddings"):
    base_loc = pt(
        "/Users/aravindhnivas/Documents/ML-properties/[PHYSICAL CONSTANTS OF ORGANIC COMPOUNDS]/tmp_C_processed_data/analysis_data/filtered/tmpC_topelements_processed_data/embedded_vectors"
    )
    processed_vec_dir = base_loc / f"processed_{embeddings}"
    X = np.load(processed_vec_dir / "processed.X.npy", allow_pickle=True)
    y = np.load(processed_vec_dir / "processed.y.npy", allow_pickle=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run timing analysis for each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        mean_time, std_time, all_times = time_model_training(model, X_train, y_train)
        results[name] = {
            "mean_time": mean_time,
            "std_time": std_time,
            "all_times": all_times,
        }
        print(f"{name}:")
        print(f"Mean training time: {mean_time:.3f} ± {std_time:.3f} seconds")
        print(f"Individual runs: {[f'{t:.3f}' for t in all_times]}")

    # Optional: Create a summary DataFrame
    summary_df = pd.DataFrame(
        {
            name: {
                "Mean Time (s)": results[name]["mean_time"],
                "Std Dev (s)": results[name]["std_time"],
                "Min Time (s)": min(results[name]["all_times"]),
                "Max Time (s)": max(results[name]["all_times"]),
            }
            for name in results
        }
    ).round(3)

    print("\nSummary Statistics:")
    print(summary_df)
    summary_df.to_csv(processed_vec_dir / "model_training_times.csv")


if __name__ == "__main__":
    for embeddings in ["mol2vec_embeddings", "VICGAE_embeddings"]:
        compute(embeddings=embeddings)
