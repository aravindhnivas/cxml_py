from .ml_types import DataType, MLResults
import matplotlib.pyplot as plt


def main_plot(data: DataType, results: MLResults, model: str):
    y_true_test = data["test"]["y_true"]
    y_pred_test = data["test"]["y_pred"]
    y_linear_fit_test = data["test"]["y_linear_fit"]

    y_true_train = data["train"]["y_true"]
    y_pred_train = data["train"]["y_pred"]

    metrics = ["r2", "mse", "rmse", "mae"]
    test_scores = {}
    train_scores = {}

    for v in ["test", "train"]:
        for k in metrics:
            mean = results["cv_scores"][v][k]["mean"]
            std = results["cv_scores"][v][k]["std"]
            if v == "test":
                test_scores[k] = f"{mean:.2f} ± {std:.2f}"
            else:
                train_scores[k] = f"{mean:.2f} ± {std:.2f}"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(y_true_train, y_pred_train, color="C0", label="Train", alpha=0.1)
    ax.scatter(y_true_test, y_pred_test, color="C1", label="Test")
    ax.plot(y_true_test, y_linear_fit_test, color="k")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.legend(loc="upper right")

    # Add text annotations for metrics
    textstr_train = "\n".join(
        (
            f"Train ({results['cv_fold']}-fold CV):",
            f'R²: {train_scores["r2"]}',
            f'MSE: {train_scores["mse"]}',
            f'RMSE: {train_scores["rmse"]}',
            f'MAE: {train_scores["mae"]}',
        )
    )

    textstr_test = "\n".join(
        (
            f"Test ({results['cv_fold']}-fold CV):",
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
