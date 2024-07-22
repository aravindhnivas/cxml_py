try:
    from umdalib.utils import logger
except ImportError:
    from loguru import logger

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


# for processing
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn import metrics

# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# Cross-validation and others
# from sklearn.model_selection import KFold, GridSearchCV, ShuffleSplit
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

# for saving models
from joblib import dump
from sklearn.utils import resample

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

random_state_supported_models = ["rfr", "gbr", "gpr"]

seed = 42
# rng = np.random.default_rng(seed)
rng = np.random.RandomState(seed)


@dataclass
class Args:
    model: str
    labels_file: str
    vectors_file: str
    test_size: float
    bootstrap: bool
    bootstrap_nsamples: int
    parameters: Dict[str, Union[str, int, None]]
    fine_tuned_hyperparameters: Dict[str, Union[str, int, float, None]]
    fine_tune_model: bool
    pre_trained_file: str
    kfold_nsamples: int


def main(args: Args):
    logger.info(f"Training {args.model} model")

    estimator = None
    grid_search = None
    best_params = None

    # load data
    X = np.load(args.vectors_file, allow_pickle=True)
    y = np.loadtxt(args.labels_file)

    # bootstrap data
    if args.bootstrap:
        args.bootstrap_nsamples = int(args.bootstrap_nsamples)
        X, y = resample(X, y, n_samples=args.bootstrap_nsamples, random_state=rng)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(args.test_size), random_state=rng
    )

    if args.model in random_state_supported_models:
        args.parameters["random_state"] = rng

    params_grid: list[Dict] = []

    if args.fine_tune_model:
        for key, value in args.fine_tuned_hyperparameters.items():
            params_grid.append({key: [value]})

        # make estimator and pass in the arguments except the fine tuned hyperparameters
        opts = {
            k: v
            for k, v in args.parameters.items()
            if k not in args.fine_tuned_hyperparameters.keys()
        }

        initial_estimator = models[args.model](**opts)

        # Grid-search
        kfold = KFold(n_splits=args.kfold_nsamples, shuffle=True, random_state=rng)
        grid_search = GridSearchCV(initial_estimator, params_grid, cv=kfold)

        # run grid search
        grid_search.fit(X_train, y_train)
        estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_

    else:
        estimator = models[args.model](**args.parameters)

    # train model
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    # evaluate model
    r2 = metrics.r2_score(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    logger.info(f"R2: {r2:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}")

    # save model
    dump(estimator, args.pre_trained_file)

    results = {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}
    if args.fine_tune_model:
        results["best_params"] = best_params

    return results
