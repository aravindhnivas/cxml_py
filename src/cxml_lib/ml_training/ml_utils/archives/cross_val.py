import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from cxml_lib.ml_training.utils import Yscalers, get_transformed_data


class FlexibleYTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformation=None, scaling=None):
        self.transformation = transformation
        self.scaling = scaling
        self.scaler = None
        self.boxcox_lambda = None
        self.yeo_johnson_transformer = None

    def fit(self, y):
        y = y.reshape(-1, 1)

        # Apply transformation
        if self.transformation:
            if self.transformation == "boxcox":
                transformed, self.boxcox_lambda = get_transformed_data(
                    y.flatten(), self.transformation, get_other_params=True
                )
            elif self.transformation == "yeo_johnson":
                transformed, self.yeo_johnson_transformer = get_transformed_data(
                    y.flatten(), self.transformation, get_other_params=True
                )
            else:
                transformed = get_transformed_data(y.flatten(), self.transformation)
            transformed = transformed.reshape(-1, 1)
        else:
            transformed = y

        # Apply scaling
        if self.scaling:
            self.scaler = Yscalers[self.scaling]()
            self.scaler.fit(transformed)

        return self

    def transform(self, y):
        y = y.reshape(-1, 1)

        # Apply transformation
        if self.transformation:
            if self.transformation == "boxcox":
                transformed = get_transformed_data(
                    y.flatten(), self.transformation, lambda_param=self.boxcox_lambda
                )
            elif self.transformation == "yeo_johnson":
                transformed = self.yeo_johnson_transformer.transform(y).flatten()
            else:
                transformed = get_transformed_data(y.flatten(), self.transformation)
            transformed = transformed.reshape(-1, 1)
        else:
            transformed = y

        # Apply scaling
        if self.scaler:
            transformed = self.scaler.transform(transformed)

        return transformed.flatten()

    def inverse_transform(self, y):
        y = y.reshape(-1, 1)

        # Inverse scaling
        if self.scaler:
            y = self.scaler.inverse_transform(y)

        # Inverse transformation
        if self.transformation:
            if self.transformation == "boxcox":
                y = get_transformed_data(
                    y.flatten(),
                    self.transformation,
                    inverse=True,
                    lambda_param=self.boxcox_lambda,
                )
            elif self.transformation == "yeo_johnson":
                y = self.yeo_johnson_transformer.inverse_transform(y)
            else:
                y = get_transformed_data(y.flatten(), self.transformation, inverse=True)

        return y.flatten()


def rmse_scorer(y_transformer):
    def scorer(estimator, X, y):
        # Get predictions in the transformed space
        y_pred_transformed = estimator.predict(X)

        # Inverse transform both true values and predictions
        y_true_original = y_transformer.inverse_transform(y)
        y_pred_original = y_transformer.inverse_transform(y_pred_transformed)

        # Calculate RMSE on the original scale
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        return -rmse  # Return negative because sklearn assumes higher is better

    return scorer


class TransformedTargetKFold:
    def __init__(self, y_transformer, n_splits=5):
        self.y_transformer = y_transformer
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        y_transformed = self.y_transformer.fit_transform(y)
        kf = KFold(n_splits=self.n_splits)
        return kf.split(X, y_transformed)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# Example usage
X = np.random.rand(100, 5)  # Example feature data
y = np.random.rand(100)  # Example target data

# Define your y-data transformations and scaling
ytransformation = "log1p"  # or "boxcox", "yeo_johnson", etc.
yscaling = "StandardScaler"  # or any other scaler from Yscalers

y_transformer = FlexibleYTransformer(transformation=ytransformation, scaling=yscaling)

# Create your model pipeline
model = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])

# Set up cross-validation
cv = TransformedTargetKFold(y_transformer, n_splits=5)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer(y_transformer))

# Print results
print("RMSE scores:", -scores)
print("Mean RMSE:", -scores.mean())
print("Standard deviation of RMSE:", scores.std())

# Fit final model on all data
y_transformed = y_transformer.fit_transform(y)
model.fit(X, y_transformed)

# To make predictions:
# y_pred_transformed = model.predict(X_new)
# y_pred = y_transformer.inverse_transform(y_pred_transformed)
