try:
    from umdalib.utils import logger
except ImportError:
    from loguru import logger

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Union, TypedDict
from joblib import (
    parallel_config,
    dump,
    __version__ as joblib_version,
)

logger.info(f"Using joblib version {joblib_version}")

import numpy as np
from pathlib import Path as pt
from datetime import datetime
import pandas as pd

from sklearn import metrics, __version__ as sklearn_version

logger.info(f"Using scikit-learn version {sklearn_version}")

from umdalib.utils import Paths


# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
from sklearn.preprocessing import StandardScaler

from dask_ml.model_selection import (
    GridSearchCV as DaskGridSearchCV,
    RandomizedSearchCV as DaskRandomizedSearchCV,
)

# for saving models
from sklearn.utils import resample

from umdalib.training.read_data import read_as_ddf
from dask.diagnostics import ProgressBar

import json
from scipy.optimize import curve_fit

from xgboost import XGBRegressor, __version__ as xgboost_version
from catboost import CatBoostRegressor, __version__ as catboost_version
from lightgbm import LGBMRegressor, __version__ as lightgbm_version

# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger.info(f"xgboost version {xgboost_version}")
logger.info(f"catboost version {catboost_version}")
logger.info(f"lightgbm version {lightgbm_version}")

# from dask.distributed import Client


# Set up Dask client
# client = Client()  # This will start a local cluster


def linear(x, m, c):
    return m * x + c


# models_dict
models_dict = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "svr": SVR,
    "knn": KNeighborsRegressor,
    "rfr": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
    "gpr": GaussianProcessRegressor,
    "xgboost": XGBRegressor,
    "catboost": CatBoostRegressor,
    "lgbm": LGBMRegressor,
}

n_jobs_keyword_available_models = ["linear_regression", "knn", "rfr", "xgboost", "lgbm"]

kernels_dict = {
    "Constant": kernels.ConstantKernel,
    "RBF": kernels.RBF,
    "Matern": kernels.Matern,
    "RationalQuadratic": kernels.RationalQuadratic,
    "ExpSineSquared": kernels.ExpSineSquared,
    "DotProduct": kernels.DotProduct,
    "WhiteKernel": kernels.WhiteKernel,
}

grid_search_dict = {
    "GridSearchCV": {"function": GridSearchCV, "parameters": []},
    "HalvingGridSearchCV": {"function": HalvingGridSearchCV, "parameters": ["factor"]},
    "RandomizedSearchCV": {"function": RandomizedSearchCV, "parameters": ["n_iter"]},
    "HalvingRandomSearchCV": {
        "function": HalvingRandomSearchCV,
        "parameters": ["factor"],
    },
    "DaskGridSearchCV": {"function": DaskGridSearchCV, "parameters": ["factor"]},
    "DaskRandomizedSearchCV": {
        "function": DaskRandomizedSearchCV,
        "parameters": ["n_iter"],
    },
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
    pre_trained_file: str
    cv_fold: int
    cross_validation: bool
    training_file: TrainingFile
    training_column_name_y: str
    npartitions: int
    vectors_file: str
    noise_percentage: float
    logYscale: bool
    scaleYdata: bool
    embedding: str
    pca: bool
    save_pretrained_model: bool
    fine_tune_model: bool
    grid_search_method: str
    grid_search_parameters: Dict[str, int]
    parallel_computation: bool
    n_jobs: int
    parallel_computation_backend: str
    use_dask: bool


def augment_data(X: np.ndarray, y: np.ndarray, n_samples: int, noise_percentage: float):
    """
    augment_data a small dataset to create a larger training set.

    :X: Feature matrix
    :y: Target vector
    :n_samples: Number of samples in the bootstrapped dataset
    :noise_percentage: Scale of Gaussian noise to add to y
    :return: Bootstrapped X and y
    """

    logger.info(f"Augmenting data with {n_samples} samples")
    X_boot, y_boot = resample(X, y, n_samples=n_samples, replace=True, random_state=rng)

    logger.info(f"Adding noise percentage: {noise_percentage}")
    noise_scale = (noise_percentage / 100) * np.abs(y_boot)

    y_boot += np.random.normal(0, noise_scale)

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


def save_intermediate_results(grid_search, filename="intermediate_results.csv"):
    """Saves intermediate cv_results_ to a CSV file."""
    df = pd.DataFrame(grid_search.cv_results_)
    df = df.sort_values(by="rank_test_score")
    df.to_csv(filename, index=False)


def compute(args: Args, X: np.ndarray, y: np.ndarray):

    start_time = perf_counter()

    estimator = None
    grid_search = None

    pre_trained_file = pt(args.pre_trained_file)
    if pre_trained_file.suffix != ".pkl":
        pre_trained_file = pre_trained_file.with_suffix(".pkl")

    # bootstrap data
    if args.bootstrap:
        logger.info("Bootstrapping data")
        X, y = augment_data(
            X,
            y,
            n_samples=int(args.bootstrap_nsamples),
            noise_percentage=float(args.noise_percentage),
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

    test_size = float(args.test_size)

    if test_size > 0:
        # split data
        logger.info("Splitting data for training and testing")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=True,
            # random_state=rng
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

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

    if args.model == "catboost":
        args.parameters["train_dir"] = str(Paths().app_log_dir / "catboost_info")
        logger.info(f"catboost_info: {args.parameters['train_dir']=}")

    logger.info(f"{models_dict[args.model]=}")
    if args.fine_tune_model:
        logger.info("Fine-tuning model")
        opts = {
            k: v
            for k, v in args.parameters.items()
            if k not in args.fine_tuned_hyperparameters.keys()
        }

        if args.parallel_computation and args.model in n_jobs_keyword_available_models:
            opts["n_jobs"] = n_jobs

        initial_estimator = models_dict[args.model](**opts)

        logger.info("Running grid search")
        # Grid-search
        # cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True, random_state=rng)
        # cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True)
        GridCV = grid_search_dict[args.grid_search_method]["function"]
        GridCV_parameters = {}
        for param in grid_search_dict[args.grid_search_method]["parameters"]:
            if param in args.grid_search_parameters:
                GridCV_parameters[param] = args.grid_search_parameters[param]

        if args.parallel_computation:
            GridCV_parameters["n_jobs"] = n_jobs

        logger.info(f"{GridCV=}, {GridCV_parameters=}")

        grid_search = GridCV(
            initial_estimator,
            args.fine_tuned_hyperparameters,
            cv=int(args.cv_fold),
            **GridCV_parameters,
        )
        logger.info("Fitting grid search")

        # run grid search
        grid_search.fit(X_train, y_train)
        estimator = grid_search.best_estimator_

        logger.info("Grid search complete")
        logger.info(f"Best score: {grid_search.best_score_}")
        logger.info(f"Best parameters: {grid_search.best_params_}")

        # client.close()

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

        if args.parallel_computation and args.model in n_jobs_keyword_available_models:
            args.parameters["n_jobs"] = n_jobs

        if args.model == "gpr" and kernel is not None:
            estimator = models_dict[args.model](kernel, **args.parameters)
        else:
            estimator = models_dict[args.model](**args.parameters)

    # Create the pipeline
    estimator = Pipeline([("estimator", estimator)])  # Train the model

    # train model
    if not args.fine_tune_model:
        logger.info("Training model")
        estimator.fit(X_train, y_train)
        logger.info("Training complete")
    else:
        logger.info("Using best estimator from grid search")

    if args.save_pretrained_model:
        # dump(estimator, pre_trained_file)
        dump((estimator, scaler), pre_trained_file)

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
        "data_size": len(y_test),
        "r2": f"{r2:.2f}",
        "mse": f"{mse:.2f}",
        "rmse": f"{rmse:.2f}",
        "mae": f"{mae:.2f}",
        "model": args.model,
    }

    results["bootstrap"] = args.bootstrap
    if args.bootstrap:
        results["bootstrap_nsamples"] = args.bootstrap_nsamples
        results["noise_percentage"] = args.noise_percentage

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

    if args.cross_validation and not args.fine_tune_model and test_size > 0:
        logger.info("Cross-validating model")

        results["cv_fold"] = args.cv_fold

        cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True)
        cv_scores = cross_val_score(
            estimator, X, y, cv=cv_fold, scoring="r2", n_jobs=n_jobs
        )
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


n_jobs = None
backend = "threading"


def main(args: Args):

    global n_jobs, backend

    if args.parallel_computation:
        n_jobs = int(args.n_jobs)
        backend = args.parallel_computation_backend

    logger.info(f"Training {args.model} model")
    logger.info(f"{args.training_file['filename']}")

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
        use_dask=args.use_dask,
    )
    y: np.ndarray = None
    if args.use_dask:
        ddf = ddf.repartition(npartitions=args.npartitions)
        with ProgressBar():
            y = ddf[args.training_column_name_y].compute()
    else:
        y = ddf[args.training_column_name_y]

    y = y.values
    if args.logYscale:
        y = np.log10(y)
    y = y[valid_mask]

    logger.info(f"{y[:5]=}, {type(y)=}")

    results = None
    if args.parallel_computation:
        with parallel_config(backend, n_jobs=n_jobs):
            logger.info(f"Using {n_jobs} jobs with {backend} backend")
            results = compute(args, X, y)
    else:
        logger.info("Running in serial mode")
        results = compute(args, X, y)

    return results
