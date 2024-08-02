try:
    from umdalib.utils import logger
except ImportError:
    from loguru import logger

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Union, TypedDict
from joblib import parallel_backend

import numpy as np
from pathlib import Path as pt
from datetime import datetime

# for processing
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import pandas as pd
from sklearn import metrics, __version__ as sklearn_version

logger.info(f"Using scikit-learn version {sklearn_version}")

# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# from sklearn.model_selection import KFold, GridSearchCV, ShuffleSplit
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler

# from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV

# for saving models
from joblib import dump
from sklearn.utils import resample

from umdalib.training.read_data import read_as_ddf
from dask.diagnostics import ProgressBar

import json
from scipy.optimize import curve_fit

from dask.distributed import Client


# Set up Dask client
client = Client()  # This will start a local cluster


def linear(x, m, c):
    return m * x + c


# models_dict
models = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "svr": SVR,
    "knn": KNeighborsRegressor,
    "rfr": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
    "gpr": GaussianProcessRegressor,
}

kernels_dict = {
    "Constant": kernels.ConstantKernel,
    "RBF": kernels.RBF,
    "Matern": kernels.Matern,
    "RationalQuadratic": kernels.RationalQuadratic,
    "ExpSineSquared": kernels.ExpSineSquared,
    "DotProduct": kernels.DotProduct,
    "WhiteKernel": kernels.WhiteKernel,
}

random_state_supported_models = ["rfr", "gbr", "gpr"]

# seed = 2024
# rng = np.random.RandomState(seed)
rng = None


class TrainingFile(TypedDict):
    filename: str
    filetype: str
    key: str


@dataclass
class Args:
    model: str
    test_size: float
    bootstrap: bool
    bootstrap_nsamples: int
    parameters: Dict[str, Union[str, int, None]]
    fine_tuned_hyperparameters: Dict[str, Union[str, int, float, None]]
    fine_tune_model: bool
    pre_trained_file: str
    cv_fold: int
    cross_validation: bool
    training_file: TrainingFile
    training_column_name_y: str
    npartitions: int
    vectors_file: str
    noise_scale: float
    logYscale: bool
    scaleYdata: bool
    embedding: str
    pca: bool
    save_pretrained_model: bool


def bootstrap_small_dataset(X, y, n_samples=800, noise_scale=0.0):
    """
    Bootstrap a small dataset to create a larger training set.

    :X: Feature matrix
    :y: Target vector
    :n_samples: Number of samples in the bootstrapped dataset
    :noise_scale: Scale of Gaussian noise to add to y
    :return: Bootstrapped X and y
    """

    X_boot, y_boot = resample(X, y, n_samples=n_samples, replace=True, random_state=rng)

    if noise_scale > 0:
        if rng is None:
            y_boot += np.random.normal(0, noise_scale, y_boot.shape)
        else:
            y_boot += rng.normal(0, noise_scale, y_boot.shape)

    return X_boot, y_boot


def make_custom_kernels(kernel_dict: Dict[str, Dict[str, str]]):

    constants_kernels = None
    other_kernels = None

    for kernel_key in kernel_dict.keys():
        kernel_params = kernel_dict[kernel_key]

        for kernel_params_key, kernel_params_value in kernel_params.items():

            if "," in kernel_params_value:
                kernel_params[kernel_params_key] = tuple(
                    float(x) for x in kernel_params_value.split(",")
                )
            elif kernel_params_value != "fixed":
                kernel_params[kernel_params_key] = float(kernel_params_value)
        logger.info(f"{kernel_key=}, {kernel_params=}")

        if kernel_key == "Constant":
            constants_kernels = kernels_dict[kernel_key](**kernel_params)
        else:
            if other_kernels is None:
                other_kernels = kernels_dict[kernel_key](**kernel_params)
            else:
                other_kernels += kernels_dict[kernel_key](**kernel_params)

    kernel = constants_kernels * other_kernels
    return kernel


def main(args: Args):
    start_time = perf_counter()
    logger.info(f"Training {args.model} model")
    logger.info(f"{args.training_file['filename']}")

    pre_trained_file = pt(args.pre_trained_file)
    if pre_trained_file.suffix != ".pkl":
        pre_trained_file = pre_trained_file.with_suffix(".pkl")

    estimator = None
    grid_search = None

    X = np.load(args.vectors_file, allow_pickle=True)
    invalid_indices = [i for i, arr in enumerate(X) if np.any(arr == 0)]
    valid_mask = np.ones(len(X), dtype=bool)  # Initially, mark all as valid
    valid_mask[invalid_indices] = False  # Mark invalid indices as False
    X = X[valid_mask]  # Keep only the rows that are marked as True in the valid_mask

    # load training data from file
    ddf = read_as_ddf(
        args.training_file["filetype"],
        args.training_file["filename"],
        args.training_file["key"],
    )
    ddf = ddf.repartition(npartitions=args.npartitions)
    y: np.ndarray = None
    with ProgressBar():
        y = ddf[args.training_column_name_y].compute()

    y = y.to_numpy()
    if args.logYscale:
        y = np.log10(y)

    y = y[valid_mask]

    logger.info(f"{y[:5]=}, {type(y)=}")

    with parallel_backend("multiprocessing", n_jobs=-1):
        # bootstrap data
        if args.bootstrap:
            logger.info("Bootstrapping data")
            X, y = bootstrap_small_dataset(
                X,
                y,
                n_samples=int(args.bootstrap_nsamples),
                noise_scale=float(args.noise_scale),
            )

        # stack the arrays (n_samples, n_features)
        if len(X.shape) == 1:
            logger.info("Reshaping X")
            X = np.vstack(X)

        logger.info(f"{X[0].shape=}\n{y[0]=}")
        logger.info(f"Loaded data: {X.shape=}, {y.shape=}")

        # scale data if needed
        scaler = None
        if args.scaleYdata:
            scaler = StandardScaler()
            y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # split data
        logger.info("Splitting data for training and testing")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(args.test_size), random_state=rng
        )

        if (
            args.model in random_state_supported_models
            and "random_state" not in args.parameters
            and rng is not None
        ):
            args.parameters["random_state"] = rng

        kernel = None
        if args.model == "gpr":
            logger.info("Using Gaussian Process Regressor with custom kernel")

            if "kernel" in args.parameters and args.parameters["kernel"]:
                kernel = make_custom_kernels(args.parameters["kernel"])
                args.parameters.pop("kernel", None)
                # return {"error": "Kernel not implemented yet"}

        if args.fine_tune_model:
            logger.info("Fine-tuning model")
            opts = {
                k: v
                for k, v in args.parameters.items()
                if k not in args.fine_tuned_hyperparameters.keys()
            }
            initial_estimator = models[args.model](**opts)

            logger.info("Running grid search")
            # Grid-search
            cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True, random_state=rng)
            grid_search = GridSearchCV(
                # grid_search = DaskGridSearchCV(
                initial_estimator,
                args.fine_tuned_hyperparameters,
                cv=cv_fold,
            )
            logger.info("Fitting grid search")

            # run grid search
            grid_search.fit(X_train, y_train)
            estimator = grid_search.best_estimator_

            logger.info("Grid search complete")
            logger.info(f"Best score: {grid_search.best_score_}")
            logger.info(f"Best parameters: {grid_search.best_params_}")

            client.close()

            # save grid search
            # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            if args.save_pretrained_model:
                grid_savefile = pre_trained_file.with_name(
                    f"{pre_trained_file.stem}_grid_search"
                ).with_suffix(".pkl")
                dump(grid_search, grid_savefile)

                df = pd.DataFrame(grid_search.cv_results_)
                df = df.sort_values(by="rank_test_score")
                df.to_csv(grid_savefile.with_suffix(".csv"))
                logger.info(f"Grid search saved to {grid_savefile}")
        else:
            if kernel is not None:
                estimator = models[args.model](kernel=kernel, **args.parameters)
            else:
                estimator = models[args.model](**args.parameters)

        # train model
        if not args.fine_tune_model:
            logger.info("Training model")
            estimator.fit(X_train, y_train)
            logger.info("Training complete")
        else:
            logger.info("Using best estimator from grid search")

        y_pred: np.ndarray = estimator.predict(X_test)

        # inverse transform if data was scaled
        if args.scaleYdata and scaler is not None:
            logger.info("Inverse transforming Y-data")
            y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        logger.info("Evaluating model")
        # evaluate model
        r2 = metrics.r2_score(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_test, y_pred)

        # logger.info(f"{y_test[:5]=}, {y_pred[:5]=}")
        logger.info(f"R2: {r2:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}")

        logger.info(f"Saving model to {pre_trained_file}")
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        if args.save_pretrained_model:
            dump(estimator, pre_trained_file)

        parameters_file = pre_trained_file.with_suffix(".parameters.json")
        if args.save_pretrained_model:
            with open(parameters_file, "w") as f:
                json.dump(args.parameters, f, indent=4)
                logger.info(f"Model parameters saved to {parameters_file.name}")

        pop, _ = curve_fit(linear, y_test, y_pred)
        y_linear_fit = linear(y_test, *pop)

        results = {
            "embedding": args.embedding,
            "PCA": args.pca,
            "data_size": len(y),
            "r2": f"{r2:.2f}",
            "mse": f"{mse:.2f}",
            "rmse": f"{rmse:.2f}",
            "mae": f"{mae:.2f}",
            "model": args.model,
        }

        results["bootstrap"] = args.bootstrap
        if args.bootstrap:
            results["bootstrap_nsamples"] = args.bootstrap_nsamples
            results["noise_scale"] = args.noise_scale

        # if args.save_pretrained_model:
        with open(f"{pre_trained_file.with_suffix('.dat.json')}", "w") as f:
            json.dump(
                {
                    "y_true": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "y_linear_fit": y_linear_fit.tolist(),
                },
                f,
                indent=4,
            )

        # Additional validation step
        results["cross_validation"] = args.cross_validation
        if args.cross_validation and not args.fine_tune_model:
            logger.info("Cross-validating model")

            results["cv_fold"] = args.cv_fold

            cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True, random_state=rng)
            cv_scores = cross_val_score(estimator, X, y, cv=cv_fold, scoring="r2")
            logger.info(f"Cross-validation R2 scores: {cv_scores}")
            logger.info(
                f"Mean CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
            )
            results["cv_scores"] = {
                "mean": f"{cv_scores.mean():.2f}",
                "std": f"{cv_scores.std() * 2:.2f}",
                "scores": cv_scores.tolist(),
            }

        if args.fine_tune_model:
            results["best_params"] = grid_search.best_params_
            results["best_score"] = f"{grid_search.best_score_:.2f}"

        results["timeframe"] = current_time

        end_time = perf_counter()
        logger.info(f"Training completed in {(end_time - start_time):.2f} s")
        results["time"] = f"{(end_time - start_time):.2f} s"

        with open(
            pre_trained_file.with_suffix(".results.json"),
            "w",
        ) as f:
            json.dump(results, f, indent=4)
            logger.info(f"Results saved to {pre_trained_file.with_suffix('.json')}")

    return results
