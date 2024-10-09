from typing import Literal, TypedDict

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from multiprocessing import cpu_count
from .utils import models_dict


class FineTunedValues(TypedDict):
    value: list[str | int | float | bool]
    type: Literal["string", "integer", "float", "bool"]
    scale: Literal["linear", "log", None]


def get_suggest(
    trial: optuna.Trial,
    name: str,
    value: list[str],
    number_type: Literal["integer", "float"],
    log=False,
) -> float | int:
    step_size = None

    if number_type == "integer":
        low = int(value[0])
        high = int(value[1])
        if not log and len(value) > 2:
            total_steps = int(value[2])
            step_size = (high - low) / total_steps
        if not step_size:
            return trial.suggest_int(name, low, high, log=log)
        return trial.suggest_int(name, low, high, step=int(step_size))

    elif number_type == "float":
        low = float(value[0])
        high = float(value[1])
        if not log and len(value) > 2:
            total_steps = float(value[2])
            step_size = (high - low) / total_steps
        if not step_size:
            return trial.suggest_float(name, low, high, log=log)
        return trial.suggest_float(name, low, high, step=float(step_size))

    return


def get_parm_grid_optuna(trial: optuna.Trial, fine_tuned_values: FineTunedValues):
    param = {}
    for key, value in fine_tuned_values.items():
        if value["type"] == "string":
            param[key] = trial.suggest_categorical(key, value["value"])
        elif value["type"] == "bool":
            param[key] = trial.suggest_categorical(key, [True, False])
        elif value["type"] == "integer" or value["type"] == "float":
            param[key] = get_suggest(
                trial, key, value["value"], value["type"], value["scale"] == "log"
            )
    return param


def xgboost_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fine_tuned_values: FineTunedValues,
    static_params: dict[str, str] = {},
) -> float:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    param = get_parm_grid_optuna(trial, fine_tuned_values)
    param.update(static_params)

    if "booster" in param:
        if param["booster"] == "gbtree" or param["booster"] == "dart":
            if "max_depth" not in param:
                param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            if "learning_rate" not in param:
                param["learning_rate"] = trial.suggest_float(
                    "learning_rate", 1e-8, 1.0, log=True
                )
            if "gamma" not in param:
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)

            if "grow_policy" not in param:
                param["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                )

        if param["booster"] == "dart":
            if "sample_type" not in param:
                param["sample_type"] = trial.suggest_categorical(
                    "sample_type", ["uniform", "weighted"]
                )
            if "normalize_type" not in param:
                param["normalize_type"] = trial.suggest_categorical(
                    "normalize_type", ["tree", "forest"]
                )
            if "rate_drop" not in param:
                param["rate_drop"] = trial.suggest_float(
                    "rate_drop", 1e-8, 1.0, log=True
                )
            if "skip_drop" not in param:
                param["skip_drop"] = trial.suggest_float(
                    "skip_drop", 1e-8, 1.0, log=True
                )

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation-rmse"
    )
    bst = xgb.train(
        param, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback]
    )
    preds = bst.predict(dvalid)
    rmse = root_mean_squared_error(y_test, preds)

    return rmse


def catboost_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fine_tuned_values: FineTunedValues,
    static_params: dict[str, str] = {},
) -> float:
    param = get_parm_grid_optuna(trial, fine_tuned_values)
    param.update(static_params)
    param["eval_metric"] = "RMSE"

    if "bootstrap_type" in param:
        if param["bootstrap_type"] == "Bayesian":
            if "bagging_temperature" not in param:
                param["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0, 10
                )
        elif param["bootstrap_type"] == "Bernoulli":
            if "subsample" not in param:
                param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    pruning_callback = optuna.integration.CatBoostPruningCallback(trial, "RMSE")
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
    rmse = root_mean_squared_error(y_test, y_pred)

    return rmse


def lgbm_optuna(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fine_tuned_values: FineTunedValues,
    static_params: dict[str, str] = {},
) -> float:
    param = get_parm_grid_optuna(trial, fine_tuned_values)
    param.update(static_params)
    param["objective"] = "regression"
    param["metric"] = "rmse"
    param["verbosity"] = -1

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
    gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

    preds = gbm.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)

    return rmse


class ExtremeBoostingModelsObjective(object):
    def __init__(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fine_tuned_values: FineTunedValues,
        static_params: dict[str, str | int | float | bool] = {},
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.fine_tuned_values = fine_tuned_values
        self.static_params = static_params

        self.model = None
        if model_name == "xgboost":
            self.model = xgboost_optuna
        elif model_name == "catboost":
            self.model = catboost_optuna
        elif model_name == "lgbm":
            self.model = lgbm_optuna
        else:
            raise ValueError(f"Model {model_name} not supported")

    def __call__(self, trial):
        return self.model(
            trial,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.fine_tuned_values,
            self.static_params,
        )


sklearn_models_names = ["ridge", "svr", "knn", "rfr", "gbr", "gpr"]


class SklearnModelsObjective(object):
    def __init__(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        fine_tuned_values: FineTunedValues,
        static_params: dict[str, str | int | float | bool] = {},
        cv: int = None,
        n_jobs: int = None,
    ):
        if model_name not in sklearn_models_names:
            raise ValueError(f"Model {model_name} not supported")

        if model_name not in models_dict:
            raise ValueError(f"Model {model_name} not supported")

        if cv and cv < 2:
            raise ValueError("cv must be greater than 1")
        if cv is None:
            cv = 5

        if n_jobs:
            if n_jobs == 0:
                n_jobs = 1
            elif n_jobs < 0:
                n_jobs = cpu_count() + n_jobs

        self.model = models_dict[model_name]
        self.fine_tuned_values = fine_tuned_values
        self.static_params = static_params

        self.X = X
        self.y = y

        self.cv = cv
        self.n_jobs = n_jobs

    def __call__(self, trial):
        param = get_parm_grid_optuna(trial, self.fine_tuned_values)
        param.update(self.static_params)

        model = self.model(**param)
        rmse = self.get_rmse(model)
        return rmse

    def get_rmse(self, model) -> float:
        rmse = -cross_val_score(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=self.n_jobs,
        )

        if self.cv:
            return rmse.mean()

        return rmse
