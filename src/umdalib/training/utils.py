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
    data = np.array(data, dtype=float)
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
            # raise ValueError("Inverse transformation not supported for Box-Cox.")

            if not lambda_param:
                raise ValueError(
                    "Lambda parameter is required for inverse Box-Cox transformation."
                )

            # Inverse Box-Cox Transformation
            if lambda_param != 0:
                # Inverse transformation when lambda is not 0
                inverse_transformed = np.power(
                    data * lambda_param + 1, 1 / lambda_param
                )
            else:
                # Inverse transformation when lambda is 0
                inverse_transformed = np.exp(data)
            return inverse_transformed

        boxcox_transformed, boxcox_lambda_param = stats.boxcox(data)
        return boxcox_transformed, boxcox_lambda_param
    elif method == "yeo_johnson":
        power_transformer = PowerTransformer(method="yeo-johnson")
        if inverse:
            return power_transformer.inverse_transform(data.reshape(-1, 1)).flatten()
        return power_transformer.fit_transform(data.reshape(-1, 1)).flatten()
    else:
        return data
