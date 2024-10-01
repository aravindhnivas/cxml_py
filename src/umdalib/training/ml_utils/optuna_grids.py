import optuna
from typing import Dict, Any


def xgboost_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }
    return params


def catboost_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "depth": trial.suggest_int("depth", 1, 16),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 1.0, log=True),
        "border_count": trial.suggest_int("border_count", 1, 255),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 1e-8, 1.0, log=True
        ),
    }
    return params


def lgbm_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }
    return params


# generate param grid for the sklearn models
# "linear_regression", "ridge", "svr", "knn", "rfr", "gbr", "gpr"


def linear_regression_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {}
    return params


def ridge_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        ),
    }
    return params


def svr_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "C": trial.suggest_float("C", 1e-8, 1.0, log=True),
        "kernel": trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "degree": trial.suggest_int("degree", 1, 5),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    return params


def knn_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical(
            "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        ),
        "leaf_size": trial.suggest_int("leaf_size", 1, 100),
    }
    return params


def rfr_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }
    return params


def gbr_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
    }
    return params


def gpr_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # "kernel": trial.suggest_categorical(
        #     "kernel",
        #     [
        #         "rbf",
        #         "matern",
        #         "rational_quadratic",
        #         "exponential",
        #         "dot_product",
        #         "white",
        #     ],
        # ),
        # "optimizer": trial.suggest_categorical(
        #     "optimizer",
        #     [
        #         "fmin_l_bfgs_b",
        #         "fmin_ncg",
        #         "simplex",
        #         "cobyla",
        #         "powell",
        #         "bfgs",
        #         "conjugate_gradient",
        #         "newton_cg",
        #         "trust_ncg",
        #         "dogleg",
        #         "trust_krylov",
        #         "trust_region",
        #     ],
        # ),
    }
    return params


def get_param_grid(model_name: str):
    if model_name == "xgboost":
        return xgboost_param_grid
    elif model_name == "catboost":
        return catboost_param_grid
    elif model_name == "lgbm":
        return lgbm_param_grid
    elif model_name == "linear_regression":
        return linear_regression_param_grid
    elif model_name == "ridge":
        return ridge_param_grid
    elif model_name == "svr":
        return svr_param_grid
    elif model_name == "knn":
        return knn_param_grid
    elif model_name == "rfr":
        return rfr_param_grid
    elif model_name == "gbr":
        return gbr_param_grid
    elif model_name == "gpr":
        return gpr_param_grid
    else:
        raise ValueError(f"Model {model_name} not supported")
