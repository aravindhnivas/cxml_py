import optuna
from typing import Dict, Any
import numpy as np
from .utils import models_dict
from sklearn import metrics
from optuna.integration import CatBoostPruningCallback
import optuna.integration.lightgbm as lgb


def xgboost_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
        # "eta": trial.suggest_float("eta", 1e-3, 1.0, log=True),
        # "max_depth": trial.suggest_int("max_depth", 1, 9),
        # "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    model = models_dict["xgboost"](**param)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False,
    )

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def catboost_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "depth": trial.suggest_int("depth", 1, 16),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 1.0, log=True),
        "border_count": trial.suggest_int("border_count", 1, 255),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "eval_metric": "RMSE",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    pruning_callback = CatBoostPruningCallback(trial, "RMSE")
    model = models_dict["catboost"](**param)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def lgbm_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
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

    lgb.train(
        param, lgb.Dataset(X_train, y_train), valid_sets=lgb.Dataset(X_test, y_test)
    )

    model = models_dict["lgbm"](**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    return rmse


# generate param grid for the sklearn models
# "linear_regression", "ridge", "svr", "knn", "rfr", "gbr", "gpr"


def linear_regression_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    param = {}
    return param


def ridge_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        ),
    }

    model = models_dict["ridge"](**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def svr_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
        "C": trial.suggest_float("C", 1e-8, 1.0, log=True),
        "kernel": trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "degree": trial.suggest_int("degree", 1, 5),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }

    model = models_dict["svr"](**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def knn_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical(
            "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        ),
        "leaf_size": trial.suggest_int("leaf_size", 1, 100),
    }

    model = models_dict["knn"](**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def rfr_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    model = models_dict["rfr"](**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def gbr_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
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

    model = models_dict["gbr"](**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def gpr_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    param = {
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    model = models_dict["gpr"](**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    return rmse


def get_optuna_objective(model_name: str):
    if model_name == "xgboost":
        return xgboost_optuna
    elif model_name == "catboost":
        return catboost_optuna
    elif model_name == "lgbm":
        return lgbm_optuna
    elif model_name == "ridge":
        return ridge_optuna
    elif model_name == "svr":
        return svr_optuna
    elif model_name == "knn":
        return knn_optuna
    elif model_name == "rfr":
        return rfr_optuna
    elif model_name == "gbr":
        return gbr_optuna
    elif model_name == "gpr":
        return gpr_optuna
    else:
        raise ValueError(f"Model {model_name} not supported")
