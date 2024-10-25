from typing import Dict, Type
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from .config import ModelName

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
