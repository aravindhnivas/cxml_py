import optuna
from typing import Dict, Any


def xgboost_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
    }
    return params


def catboost_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
        "depth": trial.suggest_int("depth", 1, 16),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 1.0),
        "border_count": trial.suggest_int("border_count", 1, 255),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.5, 1.0),
        "bagging_temperature": trial.suggest_loguniform(
            "bagging_temperature", 1e-8, 1.0
        ),
    }
    return params


def lgbm_param_grid(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
    }
    return params
