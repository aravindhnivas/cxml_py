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
from sklearn.model_selection import train_test_split

# for saving models
from joblib import dump

# bootstrap
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

seed = 42
# rng = np.random.default_rng(seed)
rng = np.random.RandomState(seed)


@dataclass
class Args:
    model: str
    labels_file: str
    vectors_file: str
    test_split_ratio: float
    bootstrap: bool
    bootstrap_nsamples: int
    parameters: Dict[str, Union[str, int]]


def main(args: Args):
    logger.info(f"Training {args.model} model")
    estimator = models[args.model](**args.parameters)

    # load data
    X = np.load(args.vectors_file, allow_pickle=True)
    y = np.loadtxt(args.labels_file)

    # sparse = False
    # if sparse:
    #     X = coo_matrix(X)

    # bootstrap data
    if args.bootstrap:
        X, y = resample(X, y, n_samples=args.bootstrap_nsamples, random_state=rng)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split_ratio, random_state=rng
    )

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
    dump(estimator, f"{args.model}.joblib")

    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}
