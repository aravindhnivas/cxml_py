import json
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path as pt
from time import perf_counter
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import sigfig

# from optuna.visualization import plot_param_importances
import optuna.visualization as opv
import optuna.visualization.matplotlib as opm
import pandas as pd
import plotly.io as pio
import shap

# from dask.diagnostics import ProgressBar
from joblib import dump, parallel_config
from loguru import logger
from optuna.importance import get_param_importances
from scipy.optimize import curve_fit
from sklearn import metrics
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import kernels
from sklearn.model_selection import (
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.utils import resample
from tqdm import tqdm

from cxml_lib.logger import Paths
from cxml_lib.load_file.read_data import read_as_ddf
from cxml_lib.ml_training.utils import Yscalers, get_transformed_data

# from cxml_lib.utils.computation import load_model
from cxml_lib.utils.json import safe_json_dump

from .ml_utils.ml_plots import learning_curve_plot, main_plot
from .ml_utils.ml_types import DataType, LearningCurve, LearningCurveData, MLResults
from .ml_utils.utils import grid_search_dict
from .ml_utils.models import models_dict, n_jobs_keyword_available_models, kernels_dict
from .ml_utils.optuna_grids import (
    ExtremeBoostingModelsObjective,
    SklearnModelsObjective,
    sklearn_models_names,
)

from cleanlab.regression.learn import CleanLearning

# from .ml_utils.cleanup import cleanup_temp_files
# cleanup_temp_files()
tqdm.pandas()

AxesArray = np.ndarray[Any, np.dtype[plt.Axes]]

plots = [
    ("hyperparameter_importance", opv.plot_param_importances),
    ("optimization_history", opv.plot_optimization_history),
    ("parallel_coordinate", opv.plot_parallel_coordinate),
    ("slice_plot", opv.plot_slice),
    ("intermediate_values", opv.plot_intermediate_values),
    ("edf", opv.plot_edf),
    ("contour", opv.plot_contour),
    ("timeline", opv.plot_timeline),
]

mplots: List[Tuple[str, Callable[..., Union[plt.Axes, AxesArray]]]] = [
    ("hyperparameter_importance", opm.plot_param_importances),
    ("optimization_history", opm.plot_optimization_history),
    ("parallel_coordinate", opm.plot_parallel_coordinate),
    ("slice_plot", opm.plot_slice),
    # ("intermediate_values", opm.plot_intermediate_values),
    ("edf", opm.plot_edf),
    ("contour", opm.plot_contour),
    ("timeline", opm.plot_timeline),
]

save_formats = ["html"]


class TrainingFile(TypedDict):
    filename: str
    filetype: str
    key: str


class FineTunedValues(TypedDict):
    value: List[str | int | float | bool]
    type: Literal["string", "integer", "float", "bool"]
    scale: Literal["linear", "log", None]


class OptunaResumeStudy(TypedDict):
    resume: bool
    id: str


@dataclass
class Args:
    model: str
    test_size: float
    bootstrap: bool
    bootstrap_nsamples: int
    parameters: Dict[str, Union[str, int, None]]
    fine_tuned_values: FineTunedValues
    pre_trained_file: str
    cv_fold: int
    cross_validation: bool
    training_file: TrainingFile
    npartitions: int
    vectors_file: str
    noise_percentage: float
    ytransformation: Optional[str]
    yscaling: Optional[str]
    embedding: str
    pca: bool
    save_pretrained_model: bool
    fine_tune_model: bool
    grid_search_method: str
    grid_search_parameters: Dict[str, int]
    parallel_computation: bool
    n_jobs: int
    parallel_computation_backend: str
    use_dask: bool
    skip_invalid_y_values: bool
    inverse_scaling: bool
    inverse_transform: bool
    learning_curve_train_sizes: Optional[List[float]]
    analyse_shapley_values: bool
    optuna_n_trials: int
    optuna_n_warmup_steps: int
    optuna_resume_study: OptunaResumeStudy
    optuna_storage_file: str
    seed: Optional[int]
    cleanlab: Optional[str]
    clean_only_train_data: bool
    index_col: str
    training_column_name_y: str
    training_column_name_X: str


def linear(x, m, c):
    return m * x + c


random_state_supported_models = ["rfr", "gbr", "gpr"]
seed = None


def calculate_leverage(X_train, X_test) -> np.ndarray:
    """Compute leverage for test samples using the hat matrix."""
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    XTX_inv = np.linalg.pinv(X_train.T @ X_train)
    hat_diag = np.einsum("ij,jk,ki->i", X_test, XTX_inv, X_test.T)
    return hat_diag


def calculate_mahalanobis_distance(X_train, X_test) -> np.ndarray:
    """Compute Mahalanobis distances for test samples from training distribution."""
    mean_vec = np.mean(X_train, axis=0)
    cov_mat = np.cov(X_train, rowvar=False)
    inv_cov_mat = np.linalg.pinv(cov_mat)
    diff = X_test - mean_vec
    md = np.einsum("ij,jk,ik->i", diff, inv_cov_mat, diff)
    return md


def get_unique_study_name(base_name: str, storage: str) -> str:
    existing_studies = optuna.study.get_all_study_summaries(storage=storage)
    existing_names = {study.study_name for study in existing_studies}

    if base_name not in existing_names:
        return base_name

    index = 1
    new_name = f"{base_name}_{index}"
    while new_name in existing_names:
        index += 1
        new_name = f"{base_name}_{index}"

    return new_name


def save_optuna_importance_plot(study: optuna.study.Study, grid_search_name: str):
    importances_fanova = get_param_importances(study)  # default method is "fanova"
    importances_mdi = get_param_importances(
        study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
    )

    logger.info("Importances from get_param_importances (fanova):")
    for param, importance in importances_fanova.items():
        logger.info(f"{param}: {importance}")

    logger.info("\nImportances from get_param_importances (MDI):")
    for param, importance in importances_mdi.items():
        logger.info(f"{param}: {importance}")

    # Save both importance methods to a CSV file
    df_importance = pd.DataFrame(
        {
            "Parameter": importances_fanova.keys(),
            "Importance (fanova)": importances_fanova.values(),
            "Importance (MDI)": [
                importances_mdi.get(param, 0) for param in importances_fanova.keys()
            ],
        }
    )
    df_importance = df_importance.sort_values("Importance (fanova)", ascending=False)

    # Save the hyperparameter importance to a CSV file
    savefile = pre_trained_loc / f"{grid_search_name}.hyperparameter_importance.csv"
    logger.info(f"Saving importance to {savefile.name}")
    df_importance.to_csv(savefile, index=False)
    logger.success(f"hyperparameter_importance saved to {savefile.name}")

    # save all optuna figures to a folder
    optuna_figures_folder = pre_trained_loc / "optuna_figures"
    if not optuna_figures_folder.exists():
        optuna_figures_folder.mkdir(parents=True)

    def save_figure(fig, filename, formats=save_formats):
        for fmt in formats:
            full_filename = optuna_figures_folder / f"{filename}.{fmt}"
            if fmt == "html":
                pio.write_html(fig, file=full_filename)
            else:
                pio.write_image(fig, file=full_filename)
            logger.info(f"Saved: {full_filename}")

    for name, plot_func in plots:
        try:
            fig = plot_func(study)
            save_figure(fig, name)
        except Exception as e:
            logger.error(f"Could not generate {name} plot: {str(e)}")

    for name, plot_func in mplots:
        try:
            with plt.style.context("seaborn-v0_8-dark"):
                ax = plot_func(study)
                fig: plt.Figure = None
                if isinstance(ax, plt.Axes):
                    fig = ax.get_figure()
                    ax.set_title("")
                elif isinstance(ax, np.ndarray):
                    ax0 = ax.flatten()[0]
                    if isinstance(ax0, plt.Axes):
                        fig = ax0.get_figure()
                if fig is None:
                    raise ValueError("Could not get figure")

                fig.set_dpi(300)
                fig.suptitle("")

                if name == "contour":
                    fig.set_size_inches((15, 12))
                elif name == "parallel_coordinate":
                    fig.set_size_inches((15, 5))
                fig.savefig(
                    optuna_figures_folder / f"{name}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
                plt.close("all")
        except Exception as e:
            logger.error(f"Could not generate {name} plot: {str(e)}")

    logger.success("All figures have been saved in the 'optuna'")


def optuna_optimize(
    args: Args,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
):
    optuna_n_trials = int(args.optuna_n_trials)
    optuna_n_warmup_steps = int(args.optuna_n_warmup_steps)

    save_loc = Paths().app_log_dir / "optuna"
    if not save_loc.exists():
        save_loc.mkdir(parents=True)

    # logfile = save_loc / "storage.db"
    optuna_storage_file = pt(args.optuna_storage_file)
    if not optuna_storage_file.parent.exists():
        optuna_storage_file.parent.mkdir(parents=True)

    storage = f"sqlite:///{str(optuna_storage_file)}"

    logger.info(f"Using {storage} for storage")

    # Define the base study name
    base_study_name = f"{loaded_training_file.stem}_{pre_trained_file.stem}"

    # Get a unique study name
    if args.optuna_resume_study["resume"]:
        study_name = base_study_name
        study_id = args.optuna_resume_study["id"]
        if study_id:
            study_name += f"_{study_id}"

        existing_studies = optuna.study.get_all_study_summaries(storage=storage)
        existing_names = {study.study_name for study in existing_studies}

        if study_name not in existing_names:
            raise ValueError(f"Study with ID: {study_id} not found")
        logger.info(f"Resuming study with ID: {study_id}")
    else:
        study_name = get_unique_study_name(base_study_name, storage)

    logger.info(f"Using study name: {study_name}")

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=optuna_n_warmup_steps),
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=args.optuna_resume_study["resume"],
    )

    static_params = {}
    for key, value in args.parameters.items():
        if key not in args.fine_tuned_values:
            static_params[key] = value
    logger.info(f"{static_params=}")

    objective: Union[SklearnModelsObjective, ExtremeBoostingModelsObjective] = None

    if current_model_name in sklearn_models_names:
        objective = SklearnModelsObjective(
            current_model_name,
            X,
            y,
            args.fine_tuned_values,
            static_params,
            args.cv_fold,
            n_jobs,
        )
    elif current_model_name in ["xgboost", "catboost", "lgbm"]:
        objective = ExtremeBoostingModelsObjective(
            current_model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            args.fine_tuned_values,
            static_params,
        )

    if objective is None:
        raise ValueError(f"Model {current_model_name} not supported")

    logger.info(f"{objective=}, {type(objective)=}")

    save_parameters(
        ".fine_tuned_parameters.json",
        args.fine_tuned_values,
        mode="fine_tuned",
        misc={
            "params": static_params,
            "optuna_n_trials": optuna_n_trials,
            "optuna_n_warmup_steps": optuna_n_warmup_steps,
            "cv_fold": args.cv_fold,
            "storage": storage,
            "study_name": study_name,
        },
    )
    logger.info("Optimizing hyperparameters using Optuna")
    study.optimize(objective, n_trials=optuna_n_trials, n_jobs=n_jobs)
    logger.info("Optuna - optimization complete")

    logger.info("Number of finished trials:", len(study.trials))
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: ", trial.value)
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    best_params = study.best_params
    best_model = models_dict[current_model_name](**best_params)

    logger.info(
        f"Best parameters: {best_params}\nBest score: {study.best_value}.\nFitting best model"
    )
    best_model.fit(X_train, y_train)
    logger.info("Fitting complete for best model from Optuna")

    if args.save_pretrained_model:
        grid_search_name = f"{pre_trained_file.stem}_grid_search"

        grid_savefile_best_model = (
            pre_trained_loc / f"{grid_search_name}_best_model.pkl"
        )
        dump(best_model, grid_savefile_best_model)
        logger.success(f"Best model saved to {grid_savefile_best_model.name}")

        grid_savefile = pre_trained_loc / f"{grid_search_name}.csv"
        df_trials = study.trials_dataframe()
        df_trials.to_csv(grid_savefile, index=False)
        logger.success(f"Trials saved to {grid_savefile.name}")

        save_optuna_importance_plot(study, grid_search_name)

    return best_model, best_params


def augment_data(
    X: np.ndarray, y: np.ndarray, n_samples: int, noise_percentage: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    augment_data a small dataset to create a larger training set.

    :X: Feature matrix
    :y: Target vector
    :n_samples: Number of samples in the bootstrapped dataset
    :noise_percentage: Scale of Gaussian noise to add to y
    :return: Bootstrapped X and y
    """

    logger.info(f"Augmenting data with {n_samples} samples")
    X_boot, y_boot = resample(
        X, y, n_samples=n_samples, replace=True, random_state=seed
    )

    logger.info(f"Adding noise percentage: {noise_percentage}")
    noise_scale = (noise_percentage / 100) * np.abs(y_boot)

    y_boot += np.random.normal(0, noise_scale)

    return X_boot, y_boot


def make_custom_kernels(kernel_dict: Dict[str, Dict[str, str]]) -> kernels.Kernel:
    logger.info("Creating custom kernel from dictionary")
    constants_kernels = None
    other_kernels = None

    for kernel_key in kernel_dict.keys():
        kernel_params = kernel_dict[kernel_key]

        for kernel_params_key, kernel_params_value in kernel_params.items():
            if "," in kernel_params_value:
                kernel_params[kernel_params_key] = tuple(
                    float(x) for x in kernel_params_value.split(",")
                )
            elif kernel_params_value != "fixed":
                kernel_params[kernel_params_key] = float(kernel_params_value)
        logger.info(f"{kernel_key=}, {kernel_params=}")

        if kernel_key == "Constant":
            constants_kernels = kernels_dict[kernel_key](**kernel_params)
        else:
            if other_kernels is None:
                other_kernels = kernels_dict[kernel_key](**kernel_params)
            else:
                other_kernels += kernels_dict[kernel_key](**kernel_params)

    kernel = constants_kernels * other_kernels
    logger.info(f"{kernel=}")
    return kernel


def get_transformed_data_for_stats(y_pred: np.ndarray, y_val: np.ndarray):
    y_true = y_val

    if inverse_scaling and yscaler:
        logger.info("Inverse transforming Y-data")
        y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true = yscaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    if inverse_transform and ytransformation:
        logger.info("Inverse transforming Y-data")
        if ytransformation == "yeo_johnson" and y_transformer:
            logger.info(
                f"Using Yeo-Johnson inverse transformation using {y_transformer=}"
            )
            y_pred = y_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true = y_transformer.inverse_transform(y_true.reshape(-1, 1)).flatten()
        else:
            logger.info("Using other inverse transformation")
            y_true = get_transformed_data(
                y_true,
                ytransformation,
                inverse=True,
                lambda_param=boxcox_lambda_param,
            )
            y_pred = get_transformed_data(
                y_pred,
                ytransformation,
                inverse=True,
                lambda_param=boxcox_lambda_param,
            )

    return y_pred, y_true


def get_stats(estimator, X_true: np.ndarray, y_val: np.ndarray):
    y_pred: np.ndarray = estimator.predict(X_true)

    y_pred, y_true = get_transformed_data_for_stats(y_pred, y_val)

    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)

    logger.info(f"R2: {r2:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    pop, _ = curve_fit(linear, y_true, y_pred)
    y_linear_fit = linear(y_true, *pop)
    y_linear_fit = np.array(y_linear_fit, dtype=float)

    return r2, mse, rmse, mae, y_true, y_pred, y_linear_fit


def compute_metrics(
    metric: Literal["r2", "mse", "rmse", "mae"], y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    if metric not in ["r2", "mse", "rmse", "mae"]:
        raise ValueError(f"Invalid metric: {metric}")

    if metric == "r2":
        return metrics.r2_score(y_true, y_pred)
    elif metric == "mse":
        return metrics.mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        return metrics.root_mean_squared_error(y_true, y_pred)
    elif metric == "mae":
        return metrics.mean_absolute_error(y_true, y_pred)


def parse_cv_scores(cross_validated_scores: dict) -> dict:
    metrics = {
        "r2": "r2",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    sets = ["train", "test"]

    cv_scores = {set_: {} for set_ in sets}
    for set_ in sets:
        for metric_key, metric in metrics.items():
            scores = cross_validated_scores[f"{set_}_{metric}"]
            if metric.startswith("neg"):
                scores = -scores

            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            ci_lower = mean - 1.96 * std
            ci_upper = mean + 1.96 * std

            sigfig_value = sigfig.round(mean, std, sep="external_brackets")

            cv_scores[set_][metric_key] = {
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "scores": scores.tolist(),
                "sigfig_value": sigfig_value,
            }

    return cv_scores


def custom_scorer(y_true, y_pred, metric, transform=False):
    if not transform:
        return compute_metrics(metric, y_true, y_pred)
    y_pred_original, y_true_original = get_transformed_data_for_stats(y_pred, y_true)
    return compute_metrics(metric, y_true_original, y_pred_original)


# Create scorer objects for use with cross_validation
def make_multi_metric_scorer(transform: bool = False) -> Dict[str, Callable]:
    def r2_score_transformed(y_true, y_pred):
        return custom_scorer(y_true, y_pred, "r2", transform)

    def neg_rmse_score_transformed(y_true, y_pred):
        return -custom_scorer(y_true, y_pred, "rmse", transform)

    def neg_mae_score_transformed(y_true, y_pred):
        return -custom_scorer(y_true, y_pred, "mae", transform)

    def neg_mse_score_transformed(y_true, y_pred):
        return -custom_scorer(y_true, y_pred, "mse", transform)

    return {
        "r2": metrics.make_scorer(r2_score_transformed),
        "neg_root_mean_squared_error": metrics.make_scorer(neg_rmse_score_transformed),
        "neg_mean_absolute_error": metrics.make_scorer(neg_mae_score_transformed),
        "neg_mean_squared_error": metrics.make_scorer(neg_mse_score_transformed),
    }


def compute_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_fold: int,
):
    logger.info("Cross-validating model")

    estimator = clone(estimator)
    cv_fold = int(cv_fold)
    transform = False
    if yscaling or ytransformation:
        transform = True
    logger.info(f"{yscaler=}, {yscaling=}, {transform=}")
    scoring = make_multi_metric_scorer(transform)
    cross_validated_scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv_fold,
        n_jobs=n_jobs,
        return_train_score=True,
        scoring=scoring,
    )
    cv_scores = parse_cv_scores(cross_validated_scores)
    logger.info(f"{cv_scores=}")

    nfold_cv_scores = {f"{cv_fold}": cv_scores}
    cv_scores_savefile = pre_trained_loc / f"{pre_trained_file.stem}.cv_scores.json"

    if cv_scores_savefile.exists():
        read_cv_scores = {}
        with open(cv_scores_savefile, "r") as f:
            read_cv_scores = json.load(f)
            read_cv_scores.update(nfold_cv_scores)

        nfold_cv_scores = read_cv_scores

    safe_json_dump(nfold_cv_scores, cv_scores_savefile)

    return cv_scores


def learn_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    sizes: List[float] = None,
    n_jobs: int = -2,
    cv=5,
):
    logger.info("Learning curve")
    scoring = "r2"
    estimator = clone(estimator)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=np.linspace(*sizes),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=seed,
    )

    max_train_size_for_cv = int(y.size - y.size / cv)
    computed_train_sizes: np.ndarray = max_train_size_for_cv * np.linspace(*sizes)
    computed_train_sizes = computed_train_sizes.astype(int)

    learning_curve_data: LearningCurveData = {}
    for train_size, cv_train_scores, cv_test_scores in zip(
        train_sizes, train_scores, test_scores
    ):
        test_mean = np.mean(cv_test_scores)
        test_std = np.std(cv_test_scores, ddof=1)
        train_mean = np.mean(cv_train_scores)
        train_std = np.std(cv_train_scores, ddof=1)
        train_sigfig_value = sigfig.round(
            train_mean, train_std, sep="external_brackets"
        )
        test_sigfig_value = sigfig.round(test_mean, test_std, sep="external_brackets")

        learning_curve_data[f"{train_size}"] = {
            "test": {
                "mean": test_mean,
                "std": test_std,
                "scores": cv_test_scores.tolist(),
                "ci_lower": test_mean - 1.96 * test_std,
                "ci_upper": test_mean + 1.96 * test_std,
                "sigfig_value": test_sigfig_value,
            },
            "train": {
                "mean": train_mean,
                "std": train_std,
                "scores": cv_train_scores.tolist(),
                "ci_lower": train_mean - 1.96 * train_std,
                "ci_upper": train_mean + 1.96 * train_std,
                "sigfig_value": train_sigfig_value,
            },
        }

        logger.info(f"{train_size} samples were used to train the model")
        logger.info(f"The average train accuracy is {train_mean:.2f}")
        logger.info(f"The average test accuracy is {test_mean:.2f}")

    learning_curve_savefile = (
        pre_trained_loc / f"{pre_trained_file.stem}.learning_curve.json"
    )
    save_json: LearningCurve = {
        "data": learning_curve_data,
        "train_sizes": train_sizes.tolist(),
        "sizes": sizes,
        "CV": cv,
        "scoring": scoring,
    }

    safe_json_dump(save_json, learning_curve_savefile)
    fig_dir = pre_trained_loc / "figures"
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)
    savefig_file = fig_dir / f"{pre_trained_file.stem}.learning_curve.pdf"
    learning_curve_plot(save_json, savefig_file)
    return


def get_shap_explainer(model_name, estimator, X):
    """
    Select and return the appropriate SHAP explainer based on the model type.

    Args:
    model_name (str): The name of the model.
    estimator: The trained model object.
    X: The feature matrix used for explanation.

    Returns:
    shap.Explainer: The appropriate SHAP explainer object.
    """
    if model_name in ["gpr", "svr"]:
        raise ValueError(f"SHAP values not supported for {model_name}")
    elif model_name in ["rfr", "gbr", "xgboost", "lgbm", "catboost"]:
        logger.info("Using TreeExplainer for SHAP values")
        return shap.TreeExplainer(estimator, X)
    elif model_name in ["linear_regression", "ridge", "lasso", "elastic_net"]:
        logger.info("Using LinearExplainer for SHAP values")
        return shap.LinearExplainer(estimator, X)
    elif model_name == "knn":
        logger.info("Using KernelExplainer for KNN SHAP values")

        # KNN doesn't have a specific explainer, so we use KernelExplainer
        # We need to define a prediction function that returns probabilities
        def predict_proba(X):
            return (
                estimator.predict_proba(X)
                if hasattr(estimator, "predict_proba")
                else estimator.predict(X)
            )

        return shap.KernelExplainer(predict_proba, X)
    else:
        logger.info(f"Using default Explainer for {model_name}")
        return shap.Explainer(estimator, X)


def analyse_shap_values(
    model_name: str,
    estimator,
    X: np.ndarray,
):
    time_start = perf_counter()
    logger.info("Analyzing SHAP values")

    explainer = get_shap_explainer(model_name, estimator, X)

    # explainer = shap.TreeExplainer(estimator, X)
    shap_values = explainer(X, check_additivity=False)

    plt.close("all")
    # plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    fig_dir = pre_trained_loc / "figures"
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)
    savefig_file = fig_dir / f"{pre_trained_file.stem}.shapley.pdf"
    plt.savefig(savefig_file, bbox_inches="tight")
    plt.close("all")
    # shap.summary_plot(shap_values, X)

    # Convert SHAP values to a numpy array
    shap_values_array = shap_values.values
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Calculate mean absolute SHAP values for each feature
    # mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    # Create a dictionary with all necessary data
    data = {
        "feature_names": explainer.feature_names or feature_names,
        "shap_values": shap_values_array.tolist(),
        # "feature_values": X.tolist(),
        # "mean_abs_shap": mean_abs_shap.tolist(),
    }

    # log data shapes
    # logger.info(f"{shap_values_array.shape=}, {mean_abs_shap.shape=}")

    shapley_savefile = pre_trained_loc / f"{pre_trained_file.stem}.shapley.json"
    safe_json_dump(data, shapley_savefile)
    logger.info(f"SHAP values saved to {shapley_savefile}.")
    logger.info(f"Time taken: {perf_counter() - time_start:.2f} seconds")
    return


pre_trained_file: pt = None
pre_trained_loc: pt = None
current_model_name: str = None


def custom_nspace(start: float, stop: float, num: int, log=True) -> np.ndarray:
    if start > stop:
        raise ValueError("Start value cannot be greater than stop value")
    if num < 1:
        raise ValueError("Number of samples must be greater than 1")

    if not log:
        return np.linspace(start, stop, num=num)
    if start == 0:
        raise ValueError("Cannot use log scale with 0")
    if stop == 0:
        raise ValueError("Cannot use log scale with 0")
    if start < 0:
        raise ValueError("Cannot use log scale with negative values")
    if stop < 0:
        raise ValueError("Cannot use log scale with negative values")

    start = np.log10(start)
    stop = np.log10(stop)
    return 10 ** np.linspace(start, stop, num=num)


def get_param_grid(fine_tuned_values: FineTunedValues) -> Dict[str, list]:
    param_grid: Dict[str, List[str] | np.ndarray] = {}

    for key, value in fine_tuned_values.items():
        if value["type"] == "string":
            param_grid[key] = value["value"]
        elif value["type"] == "bool":
            param_grid[key] = [True, False]
        elif value["type"] == "integer":
            num = 5
            start = int(value["value"][0])
            stop = int(value["value"][1])
            if len(value["value"]) > 2:
                num = int(value["value"][2])

            param_grid[key] = custom_nspace(
                start, stop, num=num, log=value["scale"] == "log"
            )
            param_grid[key] = np.asarray(param_grid[key], dtype=int)

        elif value["type"] == "float":
            num = 5
            start = float(value["value"][0])
            stop = float(value["value"][1])
            if len(value["value"]) > 2:
                num = int(value["value"][2])

            param_grid[key] = custom_nspace(
                start, stop, num=num, log=value["scale"] == "log"
            )
            param_grid[key] = np.asarray(param_grid[key], dtype=float)

        if isinstance(param_grid[key], np.ndarray):
            param_grid[key] = param_grid[key].tolist()

        param_grid[key] = sorted(set(param_grid[key]))

    logger.info(f"{param_grid=}")
    return param_grid


def fine_tune_estimator(args: Args, X: np.ndarray, y: np.ndarray):
    logger.info("Fine-tuning model")
    logger.info(f"{args.fine_tuned_values=}")

    param_grid = get_param_grid(args.fine_tuned_values)
    misc = {
        "grid_search_method": args.grid_search_method,
        "cv_fold": args.cv_fold,
    }
    save_parameters(
        ".fine_tuned_parameters.json",
        args.fine_tuned_values,
        mode="fine_tuned",
        misc=misc,
    )

    save_parameters(
        ".param_grid.json",
        param_grid,
        mode="grid",
        misc=misc,
    )

    # raise NotImplementedError("Fine-tuning not implemented yet")

    opts = {k: v for k, v in args.parameters.items() if k not in param_grid.keys()}

    if args.parallel_computation and args.model in n_jobs_keyword_available_models:
        opts["n_jobs"] = n_jobs

    initial_estimator = models_dict[args.model](**opts)

    logger.info("Running grid search")
    # Grid-search
    GridCV = grid_search_dict[args.grid_search_method]["function"]
    GridCV_parameters = {}
    for param in grid_search_dict[args.grid_search_method]["parameters"]:
        if param in args.grid_search_parameters:
            GridCV_parameters[param] = args.grid_search_parameters[param]

    if args.parallel_computation:
        GridCV_parameters["n_jobs"] = n_jobs

    logger.info(f"{GridCV=}, {GridCV_parameters=}")
    logger.info(f"{param_grid=}")
    grid_search = GridCV(
        initial_estimator,
        param_grid,
        cv=args.cv_fold,
        **GridCV_parameters,
    )
    logger.info("Fitting grid search")

    # run grid search
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    logger.info("Grid search complete")
    logger.info(f"Best score: {grid_search.best_score_}")
    logger.info(f"Best parameters: {grid_search.best_params_}")

    # save grid search
    if args.save_pretrained_model:
        grid_savefile = pre_trained_file.with_name(
            f"{pre_trained_file.stem}_grid_search"
        ).with_suffix(".pkl")
        dump(grid_search, grid_savefile)

        grid_savefile_best_model = pre_trained_file.with_name(
            f"{pre_trained_file.stem}_grid_search_best_model"
        ).with_suffix(".pkl")
        dump(best_model, grid_savefile_best_model)

        df = pd.DataFrame(grid_search.cv_results_)
        df = df.sort_values(by="rank_test_score")
        df.to_csv(grid_savefile.with_suffix(".csv"))

        logger.info(f"Grid search saved to {grid_savefile}")

    return best_model, grid_search.best_params_


def save_parameters(
    suffix: str,
    parameters: Dict[str, Union[str, int, float, None]],
    mode: Literal["fine_tuned", "normal", "grid"] = "normal",
    misc: Dict[str, Union[str, int, float, None]] = None,
):
    try:
        parameters_file = pre_trained_file.with_suffix(suffix)
        timestamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        parameters_dict = {
            "values": parameters,
            "model": current_model_name,
            "timestamp": timestamp,
            "mode": mode,
        }
        if misc:
            parameters_dict.update(misc)

        safe_json_dump(parameters_dict, parameters_file)
        return timestamp

    except Exception as e:
        logger.error(f"Error saving parameters: {e}")
        logger.error("Checking for key, value in parameters.items()")
        logger.error(f"{parameters=}")
        for key, value in parameters.items():
            logger.error(f"{key=}, {value=}, {type(value)=}")


def compute(args: Args, X: np.ndarray, y: np.ndarray):
    global pre_trained_file, pre_trained_loc, current_model_name, yscaler, seed

    current_model_name = args.model
    start_time = perf_counter()

    arguments_file = pre_trained_loc / f"{pre_trained_file.stem}.arguments.json"
    safe_json_dump(args.__dict__, arguments_file)

    if args.cleanlab and not args.clean_only_train_data:
        logger.info("Cleaning all data")
        X, y = clean_data(X, y, args.cleanlab, save=True)

        metadata_file = processed_vectors_file_dir / "metadata.json"
        metadata = json.load(open(metadata_file, "r"))
        metadata["cleaned_length"] = X.shape[0]
        safe_json_dump(metadata, metadata_file)

    test_size = float(args.test_size)

    if test_size > 0:
        logger.info("Splitting data for training and testing")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=seed
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    if (
        args.model in random_state_supported_models
        and "random_state" not in args.parameters
        and seed is not None
    ):
        args.parameters["random_state"] = seed

    kernel = None
    if args.model == "gpr":
        logger.info("Using Gaussian Process Regressor with custom kernel")
        logger.warning("checking kernal in parameters: " + "kernel" in args.parameters)
        if "kernel" in args.parameters and args.parameters["kernel"]:
            kernel = make_custom_kernels(args.parameters["kernel"])
            args.parameters["kernel"] = kernel
            logger.info(
                f"{args.parameters['kernel']=}, {type(args.parameters['kernel'])=}"
            )

    if args.model == "catboost":
        args.parameters["train_dir"] = str(Paths().app_log_dir / "catboost_info")
        logger.info(f"catboost_info: {args.parameters['train_dir']=}")

    logger.info(f"{models_dict[args.model]=}")

    estimator = None

    if args.parallel_computation and args.model in n_jobs_keyword_available_models:
        args.parameters["n_jobs"] = n_jobs

    if args.model == "catboost":
        args.parameters["verbose"] = 0
    elif args.model == "lgbm":
        args.parameters["verbose"] = -1
    elif args.model == "xgboost":
        args.parameters["verbosity"] = 0

    # if args.cleanlab and args.clean_only_train_data:
    #     logger.info("Cleaning only training data")
    #     X_train, y_train = clean_data(X_train, y_train, args.cleanlab, save=False)

    if args.fine_tune_model:
        args.cv_fold = int(args.cv_fold)
        if args.grid_search_method == "Optuna":
            logger.info("Optimizing hyperparameters using Optuna")
            estimator, best_params = optuna_optimize(
                args, X_train, y_train, X_test, y_test, X, y
            )

        else:
            logger.info(
                "Fine-tuning model using traditional grid search method: ",
                args.grid_search_method,
            )
            estimator, best_params = fine_tune_estimator(args, X, y)

        logger.info("Using best estimator from grid search")
    else:
        logger.info("Training model without fine-tuning")
        estimator = models_dict[args.model](**args.parameters)
        logger.info("Training model")
        estimator.fit(X_train, y_train)
        logger.info("Training complete")

    if estimator is None:
        raise ValueError("Estimator is None")

    if args.save_pretrained_model:
        logger.info(f"Saving model to {pre_trained_file}")
        dump((estimator, yscaler), pre_trained_file)
        logger.success("Trained model saved")

    trained_params = estimator.get_params()
    if args.model == "catboost":
        logger.info(f"{estimator.get_all_params()=}")
        trained_params = trained_params | estimator.get_all_params()

    if args.save_pretrained_model:
        save_parameters(".parameters.user.json", args.parameters)
        save_parameters(".parameters.trained.json", trained_params)

    # raise NotImplementedError("Training not implemented yet")
    logger.info("Evaluating model for test data")
    test_stats = get_stats(estimator, X_test, y_test)
    logger.info("Evaluating model for train data")
    train_stats = get_stats(estimator, X_train, y_train)

    # --- Applicability Domain Analysis ---
    logger.info("Starting Applicability Domain Analysis")
    AD_scaler = StandardScaler()
    X_train_scaled = AD_scaler.fit_transform(X_train)
    X_test_scaled = AD_scaler.transform(X_test)

    # Leverage
    leverage_scores = calculate_leverage(X_train_scaled, X_test_scaled)
    leverage_threshold = 3 * (X_train_scaled.shape[1] / X_train_scaled.shape[0])
    outside_leverage = leverage_scores > leverage_threshold
    logger.info(
        f"Outside AD (Leverage): {np.sum(outside_leverage)} / {len(outside_leverage)}"
    )

    # Mahalanobis
    mahalanobis_scores = calculate_mahalanobis_distance(X_train_scaled, X_test_scaled)
    mahalanobis_threshold = np.percentile(mahalanobis_scores, 95)
    outside_mahalanobis = mahalanobis_scores > mahalanobis_threshold
    logger.info(
        f"Outside AD (Mahalanobis): {np.sum(outside_mahalanobis)} / {len(outside_mahalanobis)}"
    )

    # save leverage and mahalanobis scores to file
    leverage_scores_file = pre_trained_file.with_suffix(".leverage_scores.json")
    mahalanobis_scores_file = pre_trained_file.with_suffix(".mahalanobis_scores.json")
    leverage_scores_dict = {
        "scores": leverage_scores.tolist(),
        "threshold": leverage_threshold,
        "outside": outside_leverage.tolist(),
    }
    mahalanobis_scores_dict = {
        "scores": mahalanobis_scores.tolist(),
        "threshold": mahalanobis_threshold,
        "outside": outside_mahalanobis.tolist(),
    }
    safe_json_dump(leverage_scores_dict, leverage_scores_file)
    safe_json_dump(mahalanobis_scores_dict, mahalanobis_scores_file)
    logger.success("Applicability Domain Analysis complete")

    results: MLResults = {
        "test_size": test_size,
        "seed": args.seed,
        "data_shapes": {
            "X": X.shape,
            "y": y.shape,
            "X_test": X_test.shape,
            "y_test": y_test.shape,
            "X_train": X_train.shape,
            "y_train": y_train.shape,
        },
        "test_stats": {
            "r2": test_stats[0],
            "mse": test_stats[1],
            "rmse": test_stats[2],
            "mae": test_stats[3],
        },
        "train_stats": {
            "r2": train_stats[0],
            "mse": train_stats[1],
            "rmse": train_stats[2],
            "mae": train_stats[3],
        },
    }

    if args.cross_validation and test_size > 0:
        results["cv_fold"] = args.cv_fold
        cv_scores = compute_cv(estimator, X, y, args.cv_fold)
        results["cv_scores"] = cv_scores

    timestamp = None
    if args.fine_tune_model:
        results["best_params"] = best_params
        misc = {
            "cv_fold": args.cv_fold,
            "grid_search_method": args.grid_search_method,
        }
        timestamp = save_parameters(".best_params.json", best_params, misc=misc)

    if timestamp is None:
        timestamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")

    results["timestamp"] = timestamp

    end_time = perf_counter()
    logger.info(f"Training completed in {(end_time - start_time):.2f} s")
    results["time"] = f"{(end_time - start_time):.2f} s"
    safe_json_dump(results, pre_trained_file.with_suffix(".results.json"))

    dat: DataType = {
        "test": {
            "y_true": test_stats[4].tolist(),
            "y_pred": test_stats[5].tolist(),
            "y_linear_fit": test_stats[6].tolist(),
        },
        "train": {
            "y_true": train_stats[4].tolist(),
            "y_pred": train_stats[5].tolist(),
            "y_linear_fit": train_stats[6].tolist(),
        },
    }

    safe_json_dump(dat, pre_trained_file.with_suffix(".dat.json"))

    if args.save_pretrained_model:
        fig = main_plot(dat, results, args.model)
        fig_dir = pre_trained_loc / "figures"

        if not fig_dir.exists():
            fig_dir.mkdir(parents=True)

        figname = pre_trained_file.stem + ".main_plot.pdf"
        if "cv_scores" in results:
            figname = pre_trained_file.stem + ".main_plot_cv.pdf"
        fig.savefig(fig_dir / figname, bbox_inches="tight")

    if args.learning_curve_train_sizes is not None and args.cross_validation:
        logger.info("Computing learning curve")
        learn_curve(
            estimator,
            X,
            y,
            sizes=args.learning_curve_train_sizes,
            n_jobs=n_jobs,
            cv=args.cv_fold,
        )
        logger.info("Learning curve computed")

    if args.analyse_shapley_values:
        sample_size = 1000  # If X is too large, take a random sample
        if X.shape[0] > sample_size:
            idx = np.random.choice(X.shape[0], sample_size, replace=False)
            background_data = X[idx]
        else:
            background_data = X

        analyse_shap_values(args.model, estimator, background_data)

    return results


def convert_to_float(value: Union[str, float]) -> float:
    try:
        return float(value)
    except ValueError:
        if isinstance(value, str) and "-" in value:
            parts = value.split("-")
            if len(parts) == 2 and parts[0] and parts[1]:
                try:
                    return (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    pass
        if skip_invalid_y_values:
            return np.nan
        raise


n_jobs = 1
backend = "threading"
skip_invalid_y_values = False
ytransformation: str = None
y_transformer: str = None
yscaling: str = None
yscaler = None
boxcox_lambda_param = None
inverse_scaling = True
inverse_transform = True
loaded_training_file: pt = None

final_df: pd.DataFrame = None


def get_data(args: Args) -> Tuple[np.ndarray, np.ndarray]:
    global yscaler, boxcox_lambda_param, y_transformer, final_df

    processed_df_file = processed_vectors_file_dir / "processed_df.parquet"
    if processed_df_file.exists():
        logger.info(f"Loading processed data from {processed_df_file}")
        final_df = pd.read_parquet(processed_df_file)
        X = final_df.iloc[:, 2:].to_numpy()
        y = final_df["y"].to_numpy()
        logger.success(
            f"Processed data loaded from {processed_df_file}\n{X.shape=}, {y.shape=}"
        )
    else:
        logger.info("Loading data")
        X = np.load(args.vectors_file, allow_pickle=True)
        X = np.array(X, dtype=float)
        original_length = X.shape[0]

        # stack the arrays (n_samples, n_features)
        if len(X.shape) == 1:
            logger.info("Reshaping X")
            X = np.vstack(X)

        # load training data from file
        ddf: pd.DataFrame = read_as_ddf(
            args.training_file["filetype"],
            args.training_file["filename"],
            args.training_file["key"],
            use_dask=args.use_dask,
            computed=True,
        )
        ddf.set_index(args.index_col, inplace=True)
        logger.info(f"{ddf.columns=}")

        # Create DataFrame with feature columns
        feature_cols = [str(i) for i in range(X.shape[1])]
        data_df = pd.DataFrame(X, index=ddf.index, columns=feature_cols)

        # Add SMILES and y columns
        data_df.loc[:, args.training_column_name_X] = ddf[args.training_column_name_X]
        data_df.loc[:, "y"] = pd.to_numeric(
            ddf[args.training_column_name_y], errors="coerce"
        )

        # Reorder columns
        cols_order = [args.training_column_name_X, "y"] + feature_cols
        data_df = data_df[cols_order]

        # Filter data efficiently using numpy operations
        features = data_df.iloc[:, 2:].to_numpy()
        y_values = data_df["y"].to_numpy()

        # Create masks
        non_zero_mask = np.any(features != 0, axis=1)
        valid_y_mask = ~(np.isnan(y_values) | np.isinf(y_values))
        final_mask = non_zero_mask & valid_y_mask

        # Apply final filtering
        final_df = data_df[final_mask]
        final_df.to_parquet(processed_df_file, compression="snappy")
        logger.success(f"Processed data saved to {processed_df_file}")

        # Print statistics
        logger.info(f"Original number of rows: {len(data_df)}")
        logger.info(f"Rows removed due to all-zero features: {np.sum(~non_zero_mask)}")
        logger.info(f"Rows removed due to invalid y values: {np.sum(~valid_y_mask)}")
        logger.info(f"Final number of rows: {len(final_df)}")

        if np.sum(~final_mask) == 0:
            logger.info("No invalid values found in X and y")
            with open(processed_vectors_file_dir / ".all_valid", "w") as f:
                f.write("All valid values")

        X = final_df.iloc[:, 2:].to_numpy()
        y = final_df["y"].to_numpy()

        X_validated_length = original_length - np.sum(~non_zero_mask)
        final_length = len(final_df)

        metadata = {
            "original_length": original_length,
            "X_validated_length": X_validated_length,
            "final_length": final_length,
        }

        metadata_file = processed_vectors_file_dir / "metadata.json"
        safe_json_dump(metadata, metadata_file)

    y_transformer = None
    if ytransformation:
        if ytransformation == "boxcox":
            logger.info("Applying boxcox transformation")
            y, boxcox_lambda_param = get_transformed_data(
                y, ytransformation, get_other_params=True
            )
            logger.info(f"{boxcox_lambda_param=}")
        elif ytransformation == "yeo_johnson":
            logger.info("Applying yeo-johnson transformation")
            y, y_transformer = get_transformed_data(
                y, ytransformation, get_other_params=True
            )
            logger.info(f"{y_transformer=}")
        else:
            logger.info(f"Applying {ytransformation} transformation")
            y = get_transformed_data(y, ytransformation)
    else:
        logger.warning("No transformation applied")

    yscaler = None
    if yscaling:
        yscaler = Yscalers[yscaling]()
        y = yscaler.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        logger.warning("No scaling applied")

    # bootstrap data
    if args.bootstrap:
        logger.info("Bootstrapping data")
        X, y = augment_data(
            X,
            y,
            n_samples=int(args.bootstrap_nsamples),
            noise_percentage=float(args.noise_percentage),
        )

    logger.info(f"Loaded data: {X.shape=}, {y.shape=}")
    with open(processed_vectors_file_dir / ".data_loaded", "w") as f:
        f.write("Data loaded")

    return X, y


def clean_data(
    X: np.ndarray, y: np.ndarray, clean_model_name: str, save: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Cleaning data using {clean_model_name}")
    cleanlab_label_issue_file = (
        processed_vectors_file_dir / f"label_issues_{clean_model_name}.parquet"
    )

    if cleanlab_label_issue_file and cleanlab_label_issue_file.exists():
        logger.info(f"Loading label issues from {cleanlab_label_issue_file}")
        label_issues_df = pd.read_parquet(cleanlab_label_issue_file)
    else:
        logger.info("Running cleanlab to clean data")
        clean_model = models_dict[clean_model_name]()
        cl = CleanLearning(clean_model, verbose=True)
        cl.fit(X, y)

        label_issues_df = cl.get_label_issues()
        label_issues_df.index = final_df.index

        if save:
            label_issues_df.to_parquet(cleanlab_label_issue_file)

        indices_match = (label_issues_df.index == final_df.index).all()

        if not indices_match:
            logger.error("Indices do not match between final_df and label_issues_df")
            logger.info("final_df index shape:", final_df.index.shape)
            logger.info("label_issues_df index shape:", label_issues_df.index.shape)
        else:
            logger.info("Indices match between final_df and label_issues_df")

    X_cleaned = X[~label_issues_df["is_label_issue"]]
    y_cleaned = y[~label_issues_df["is_label_issue"]]

    logger.info("Cleaned data using cleanlab")
    logger.info(
        f"X_cleaned shape: {X_cleaned.shape}, y_cleaned shape: {y_cleaned.shape}"
    )

    return X_cleaned, y_cleaned


processed_vectors_file_dir: pt = None


def main(args: Args):
    global \
        n_jobs, \
        backend, \
        skip_invalid_y_values, \
        ytransformation, \
        yscaling, \
        inverse_scaling, \
        inverse_transform, \
        loaded_training_file, \
        pre_trained_file, \
        pre_trained_loc, \
        seed, \
        processed_vectors_file_dir

    if args.seed:
        seed = int(args.seed)

    pre_trained_file = pt(args.pre_trained_file.strip()).with_suffix(".pkl")
    pre_trained_loc = pre_trained_file.parent
    if not pre_trained_loc.exists():
        pre_trained_loc.mkdir(parents=True)

    logfile = pre_trained_loc / f"{pre_trained_file.stem}.log"
    logger.info(f"Logging to {logfile}")

    logger.add(
        logfile,
        rotation="10 MB",
        compression="zip",
        mode="w",
    )

    if args.model not in models_dict:
        logger.error(f"{args.model} not implemented in yet!")
        raise ValueError(f"{args.model} not implemented in yet!")

    ytransformation = args.ytransformation
    yscaling = args.yscaling
    inverse_scaling = args.inverse_scaling
    inverse_transform = args.inverse_transform
    loaded_training_file = pt(args.training_file["filename"])

    skip_invalid_y_values = args.skip_invalid_y_values
    if args.parallel_computation:
        n_jobs = int(args.n_jobs)
        if n_jobs < 1:
            n_jobs = cpu_count() + n_jobs

        backend = args.parallel_computation_backend

    logger.info(f"Training {args.model} model")
    logger.info(f"{args.training_file['filename']}")

    # save the processed X and y data
    vectors_file = pt(args.vectors_file)
    vectors_loc = vectors_file.parent
    processed_vectors_file_dir = vectors_loc / f"processed_{vectors_file.stem}"
    if not processed_vectors_file_dir.exists():
        processed_vectors_file_dir.mkdir(parents=True)
        logger.info(
            f"Created directory: {processed_vectors_file_dir} for saving processed vectors"
        )

    X, y = get_data(args)

    results = None

    if args.parallel_computation:
        with parallel_config(backend, n_jobs=n_jobs):
            logger.info(f"Using {n_jobs} jobs with {backend} backend")
            results = compute(args, X, y)
    else:
        logger.info("Running in serial mode")
        results = compute(args, X, y)

    return results
