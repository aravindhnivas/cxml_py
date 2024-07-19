from dataclasses import dataclass
from typing import Dict, Union
from umdalib.utils import logger
import numpy as np
from joblib import load, dump

# for processing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from sklearn.model_selection import KFold, GridSearchCV, ShuffleSplit

# bootstrap
from sklearn.utils import resample
from scipy.sparse import coo_matrix

# models_dict
models = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(),
    "svr": SVR(),
    "knn": KNeighborsRegressor(),
    "rfr": RandomForestRegressor(),
    "gbr": GradientBoostingRegressor(),
    "gpr": GaussianProcessRegressor(),
}

seed = 42


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
    estimator = models[args.model]

    # load data
    X = np.load(args.vectors_file, allow_pickle=True)
    X_sparse = coo_matrix(X)

    y = np.load(args.labels_file, allow_pickle=True)

    rng = np.random.default_rng(seed)

    # bootstrap data
    if args.bootstrap:
        X_sparse, y = resample(X_sparse, y, n_samples=args.bootstrap_nsamples)
    X = X_sparse.toarray()

    # split data using test_split_ratio
    split = int(len(y) * float(args.test_split_ratio))
    X_train, X_test = X[split:], X[:split]
    y_train, y_test = y[split:], y[:split]

    # train model
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    # evaluate model
    r2 = metrics.r2_score(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    logger.info(f"R2: {r2}, MSE: {mse}")

    return {"r2": r2, "mse": mse}
