from catboost import CatBoostRegressor
from dask_ml.model_selection import (
    RandomizedSearchCV as DaskRandomizedSearchCV,
    GridSearchCV as DaskGridSearchCV,
)
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from typing import Dict, Type, Literal

# Define the literal types for model names
ModelName = Literal[
    "linear_regression",
    "ridge",
    "svr",
    "knn",
    "rfr",
    "gbr",
    "gpr",
    "xgboost",
    "catboost",
    "lgbm",
]

# models_dict
models_dict: Dict[ModelName, Type] = {
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

available_models = models_dict.keys()

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
