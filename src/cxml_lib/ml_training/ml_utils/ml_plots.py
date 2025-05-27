from .ml_types import DataType, MLResults, LearningCurve
import matplotlib.pyplot as plt
from pathlib import Path as pt
from sigfig import round


def main_plot(data: DataType, results: MLResults, model: str):
    y_true_test = data["test"]["y_true"]
    y_pred_test = data["test"]["y_pred"]
    y_linear_fit_test = data["test"]["y_linear_fit"]

    y_true_train = data["train"]["y_true"]
    y_pred_train = data["train"]["y_pred"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(y_true_train, y_pred_train, color="C0", label="Train", alpha=0.5)
    ax.scatter(y_true_test, y_pred_test, color="C1", label="Test")
    ax.plot(y_true_test, y_linear_fit_test, color="k")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.legend(loc="upper right")

    metrics = ["r2", "mse", "rmse", "mae"]
    test_scores = {}
    train_scores = {}

    if "cv_scores" in results:
        for v in ["test", "train"]:
            for k in metrics:
                mean = results["cv_scores"][v][k]["mean"]
                std = results["cv_scores"][v][k]["std"]
                if v == "test":
                    # test_scores[k] = f"{mean:.2f} ± {std:.2f}"
                    test_scores[k] = round(mean, std, sep="external_brackets")
                else:
                    # train_scores[k] = f"{mean:.2f} ± {std:.2f}"
                    train_scores[k] = round(mean, std, sep="external_brackets")
    else:
        for k in metrics:
            # test_scores[k] = f'{results["test_stats"][k]:.2f}'
            # train_scores[k] = f'{results["train_stats"][k]:.2f}'
            test_scores[k] = round(results["test_stats"][k], sigfigs=2)
            train_scores[k] = round(results["train_stats"][k], sigfigs=2)

    lg_ = ""
    if "cv_fold" in results:
        lg_ = f" ({results['cv_fold']}-fold CV)"

    # Add text annotations for metrics
    textstr_train = "\n".join(
        (
            f"Train{lg_}:",
            f'R²: {train_scores["r2"]}',
            f'MSE: {train_scores["mse"]}',
            f'RMSE: {train_scores["rmse"]}',
            f'MAE: {train_scores["mae"]}',
        )
    )

    textstr_test = "\n".join(
        (
            f"Test{lg_}:",
            f'R²: {test_scores["r2"]}',
            f'MSE: {test_scores["mse"]}',
            f'RMSE: {test_scores["rmse"]}',
            f'MAE: {test_scores["mae"]}',
        )
    )

    ax.text(
        0.05,
        0.95,
        textstr_train,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="C0", alpha=0.5),
    )
    ax.text(
        0.95,
        0.30,
        textstr_test,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="C1", alpha=0.5),
    )

    ax.set_title(model.upper())
    fig.set_dpi(300)
    fig.tight_layout()

    return fig


def learning_curve_plot(learning_curve: LearningCurve, savefile: pt):
    # plot the learning curve
    fig, ax = plt.subplots(dpi=300)
    train_sizes = learning_curve["train_sizes"]
    learning_curve_data = learning_curve["data"]

    for t in ["train", "test"]:
        scores = [learning_curve_data[f"{size}"][t]["mean"] for size in train_sizes]
        ci_lower = [
            learning_curve_data[f"{size}"][t]["ci_lower"] for size in train_sizes
        ]
        ci_upper = [
            learning_curve_data[f"{size}"][t]["ci_upper"] for size in train_sizes
        ]

        ax.plot(
            train_sizes,
            scores,
            ".--",
            label=t,
            color="C0" if t == "train" else "C1",
            ms=10,
        )
        ax.fill_between(train_sizes, ci_lower, ci_upper, alpha=0.3)

    ax.set_xlabel("Number of samples")
    ax.set_ylabel(f"R² ({learning_curve['CV']}-fold CV)")
    ax.legend()
    ax.minorticks_on()
    ax.set_xbound(0, 1.1 * max(train_sizes))
    ax.set_ylim(ymax=1)
    fig.tight_layout()
    fig.savefig(savefile, dpi=300, bbox_inches="tight")

    return fig
