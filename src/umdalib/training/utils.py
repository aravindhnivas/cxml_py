import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    MaxAbsScaler,
)
from scipy.special import inv_boxcox


Yscalers = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "RobustScaler": RobustScaler,
    "QuantileTransformer": QuantileTransformer,
    "PowerTransformer": PowerTransformer,
}


def get_transformed_data(
    data: np.ndarray, method: str, inverse=False, lambda_param=None
) -> np.ndarray:
    if isinstance(data, list):
        data = np.array(data, dtype=float)

    if method in Yscalers.keys():
        scaler = Yscalers[method]()
        if inverse:
            return scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()

    if method == "log1p":
        if inverse:
            return np.expm1(data)
        return np.log1p(data)
    elif method == "sqrt":
        if inverse:
            return np.power(data, 2)
        return np.sqrt(data)
    elif method == "reciprocal":
        if inverse:
            return (1 / data) - 1
        return 1 / (data + 1)
    elif method == "square":
        if inverse:
            return np.sqrt(data)
        return np.power(data, 2)
    elif method == "exp":
        if inverse:
            return np.log(data)
        return np.exp(data)
    elif method == "boxcox":
        if inverse:
            inverse_transformed = inv_boxcox(data, lambda_param)
            return inverse_transformed

        boxcox_transformed, boxcox_lambda_param = stats.boxcox(data)
        return boxcox_transformed, boxcox_lambda_param
    elif method == "yeo_johnson":
        power_transformer = PowerTransformer(method="yeo-johnson")
        if inverse:
            return power_transformer.inverse_transform(data.reshape(-1, 1)).flatten()

        transformed_data = power_transformer.fit_transform(
            data.reshape(-1, 1)
        ).flatten()
        return power_transformer, transformed_data
    else:
        return data
