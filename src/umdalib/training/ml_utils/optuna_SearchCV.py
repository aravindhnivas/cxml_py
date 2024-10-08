from typing import Literal, TypedDict

import numpy as np
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.integration import OptunaSearchCV

from .utils import models_dict


class FineTunedValues(TypedDict):
    value: list[str | int | float | bool]
    type: Literal["string", "integer", "float", "bool"]
    scale: Literal["linear", "log", None]


def get_suggest(
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
            return IntDistribution(name, low, high, log=log)
        return IntDistribution(name, low, high, step=int(step_size))

    elif number_type == "float":
        low = float(value[0])
        high = float(value[1])
        if not log and len(value) > 2:
            total_steps = float(value[2])
            step_size = (high - low) / total_steps
        if not step_size:
            return FloatDistribution(name, low, high, log=log)
        return FloatDistribution(name, low, high, step=float(step_size))

    return


def get_parm_distribution_optunaSearchCV(fine_tuned_values: FineTunedValues):
    param = {}

    for key, value in fine_tuned_values.items():
        if value["type"] == "string":
            param[key] = CategoricalDistribution(key, value["value"])
        elif value["type"] == "bool":
            param[key] = CategoricalDistribution(key, [True, False])
        elif value["type"] == "integer" or value["type"] == "float":
            param[key] = get_suggest(
                key, value["value"], value["type"], value["scale"] == "log"
            )

    return param


def optuna_search_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    fine_tuned_values: FineTunedValues,
    static_params: dict[str, str] = {},
    n_trials: int = None,
) -> float:
    model = models_dict[model_name](**static_params)
    param_distributions = get_parm_distribution_optunaSearchCV(fine_tuned_values)
    optuna_search = OptunaSearchCV(model, param_distributions, n_trials=n_trials)

    optuna_search.fit(X, y)
    return optuna_search
