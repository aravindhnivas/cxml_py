from dataclasses import dataclass
from pathlib import Path as pt
import numpy as np

# import pandas as pd
import pandas as pd
from scipy import stats
from cxml_lib.load_file.read_data import read_as_ddf
from cxml_lib.utils.json import safe_json_dump
from cxml_lib.logger import logger
from cxml_lib.ml_training.utils import get_transformed_data


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    use_dask: bool
    property_column: str
    save_loc: str
    bin_size: int
    auto_bin_size: bool
    savefilename: str
    auto_transform_data: bool
    ytransformation: str


boxcox_lambda_param = None


def get_skew_and_transformation(df_y: pd.Series):
    """
    Check if the data is transformed based on the skewness value.
    If the skewness is greater than 1, the data is highly skewed.
    In this case, the data can be transformed using a power transformation.
    """

    global boxcox_lambda_param

    skewness = df_y.skew()
    if skewness > 1:
        logger.info(
            f"Data is highly skewed (skewness = {skewness:.2f}). Consider transforming the data."
        )
    else:
        logger.info(f"Data is not highly skewed (skewness = {skewness:.2f}).")

    data = df_y.values
    logger.info(f"{len(data)=}")

    transformed_data = {}
    transformed_data["None"] = data

    # Apply transformations based on skewness
    if skewness > 0:
        # Positive Skew (Right Skew)
        log_transformed = get_transformed_data(data, "log1p")
        sqrt_transformed = get_transformed_data(data, "sqrt")
        reciprocal_transformed = get_transformed_data(data, "reciprocal")

        transformed_data["log1p"] = log_transformed
        transformed_data["sqrt"] = sqrt_transformed
        transformed_data["reciprocal"] = reciprocal_transformed

    elif skewness < 0:
        # Negative Skew (Left Skew)
        square_transformed = get_transformed_data(data, "square")
        exp_transformed = get_transformed_data(data, "exp")

        transformed_data["square"] = square_transformed
        transformed_data["exp"] = exp_transformed

    # Box-Cox Transformation (Works for positive data only, needs scipy)
    # Make sure data is strictly positive for Box-Cox
    if np.all(data > 0):
        boxcox_transformed, boxcox_lambda_param = get_transformed_data(
            data, "boxcox", get_other_params=True
        )
        transformed_data["boxcox"] = boxcox_transformed

    # Yeo-Johnson Transformation (Can handle zero and negative values)
    transformed_data["yeo_johnson"] = get_transformed_data(data, "yeo_johnson")
    logger.info(f"{transformed_data.keys()=}")

    # Compute skewness for each transformation
    logger.info("Skewness after transformation:")
    computed_skewness = {}
    for method, transformed in transformed_data.items():
        skew = stats.skew(transformed)
        logger.info(f"{method}: {skew:.2f}")
        computed_skewness[method] = skew

    if not computed_skewness:
        logger.info("No valid skewness transformations found.")
        return None, None, None

    best_skew_key = None
    best_skew_key = min(computed_skewness, key=lambda k: abs(computed_skewness[k]))
    logger.info(f"Best transformation: {best_skew_key}")

    savefile_skews = save_loc / "skewness_after_all_transformation.json"
    safe_json_dump(
        {"best_skew_key": best_skew_key, "skews": computed_skewness}, savefile_skews
    )
    return computed_skewness, best_skew_key, transformed_data[best_skew_key]


def calculate_bin_size(bin_edges):
    # Calculate the width of each bin
    bin_sizes = np.diff(bin_edges)
    all_equal_bins = np.allclose(bin_sizes, bin_sizes[0])
    logger.warning(f"{all_equal_bins=}")  # Should print True if all bin sizes are equal
    # Assuming uniform bin widths, take the first bin width as the bin size
    bin_size = bin_sizes[0]
    return bin_size


def get_analysis_results(df_y: pd.Series):
    # 1. Descriptive Statistics
    desc_stats = df_y.describe().to_dict()

    # 2. Histogram data
    hist, bin_edges = np.histogram(df_y, bins="auto")
    # Calculate bin size from histogram data
    bin_size = calculate_bin_size(bin_edges)
    logger.info(f"{bin_size=} for {len(hist)} bins")
    hist_data = {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_size": bin_size,
    }

    # 3. Box Plot data
    box_plot_data = {
        "min": float(df_y.min()),
        "q1": float(df_y.quantile(0.25)),
        "median": float(df_y.median()),
        "q3": float(df_y.quantile(0.75)),
        "max": float(df_y.max()),
    }

    # 4. Q-Q Plot data
    qq_data = stats.probplot(df_y, dist="norm")
    qq_plot_data = {
        "theoretical_quantiles": qq_data[0][0].tolist(),
        "sample_quantiles": qq_data[0][1].tolist(),
    }

    # Perform the Anderson-Darling test
    ad_result = stats.anderson(df_y)

    # Extract the test statistic and significance level
    ad_statistic = ad_result.statistic
    ad_significance_level = ad_result.significance_level

    # Store the results in a dictionary
    anderson_darling_test = {
        "statistic": float(ad_statistic),
        "significance_levels": ad_significance_level.tolist(),  # Convert to list for JSON serialization
        "critical_values": ad_result.critical_values.tolist(),  # Convert to list for JSON serialization
    }

    # Anderson-Darling test
    logger.info("\nAnderson-Darling test:")
    logger.info(f"Statistic: {ad_result.statistic:.4f}")
    for i in range(len(ad_result.critical_values)):
        sl, cv = ad_result.significance_level[i], ad_result.critical_values[i]
        logger.info(f"At {sl}% significance level: critical value is {cv:.4f}")

    # 6. Skewness and Kurtosis
    skewness = float(df_y.skew())
    kurtosis = float(df_y.kurtosis())

    # 7. KDE data
    kde = stats.gaussian_kde(df_y)
    x_range = np.linspace(df_y.min(), df_y.max(), 100)
    kde_data = {"x": x_range.tolist(), "y": kde(x_range).tolist(), "bin_size": kde.n}

    # Combine all data
    analysis_results = {
        "descriptive_statistics": desc_stats,
        "histogram": hist_data,
        "box_plot": box_plot_data,
        "qq_plot": qq_plot_data,
        "anderson_darling_test": anderson_darling_test,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "kde": kde_data,
        "applied_transformation": None,
        "boxcox_lambda": None,
    }

    return analysis_results


save_loc = None


def main(args: Args):
    global boxcox_lambda_param, save_loc

    boxcox_lambda_param = None

    if not args.save_loc:
        raise ValueError("Please provide a valid save location.")

    save_loc = pt(args.save_loc)
    if not save_loc.exists():
        save_loc.mkdir(parents=True)

    df = read_as_ddf(
        args.filetype,
        args.filename,
        args.key,
        use_dask=args.use_dask,
        computed=True,
    )

    # Assuming your target property is named 'property'
    if not args.property_column:
        raise ValueError("Please provide a valid property column name.")

    df_y = df[args.property_column]

    y_data_distribution_file = save_loc / args.savefilename
    y_data_distribution_file = y_data_distribution_file.with_suffix(".json")

    if not y_data_distribution_file.exists():
        safe_json_dump(get_analysis_results(df_y), y_data_distribution_file)

    y_transformed = None
    ytransformation = None

    if not args.auto_transform_data:
        ytransformation = args.ytransformation

    best_skew_key = None
    if args.auto_transform_data:
        computed_skewness, best_skew_key, y_transformed = get_skew_and_transformation(
            df_y
        )
        logger.info(f"{best_skew_key=}\n{computed_skewness=}")
        if best_skew_key:
            df_y = pd.Series(y_transformed)

    elif ytransformation:
        if ytransformation == "boxcox":
            y_transformed, boxcox_lambda_param = get_transformed_data(
                df_y.values, ytransformation, get_other_params=True
            )
        else:
            y_transformed = get_transformed_data(df_y.values, ytransformation)

        df_y = pd.Series(y_transformed)

    if best_skew_key and best_skew_key != "None":
        logger.info(f"Best transformation from auto-transform: {best_skew_key}")
        ytransformation = best_skew_key

    if not isinstance(df_y, pd.Series):
        df_y = pd.Series(df_y)

    analysis_results = get_analysis_results(df_y)
    if ytransformation:
        analysis_results["applied_transformation"] = ytransformation
        if ytransformation == "boxcox":
            analysis_results["boxcox_lambda"] = boxcox_lambda_param
    else:
        analysis_results["applied_transformation"] = None
        analysis_results["boxcox_lambda"] = None

    # Save the analysis results
    savefilename = args.savefilename
    if not savefilename.endswith(".json"):
        savefilename += ".json"

    if ytransformation:
        savefilename = savefilename.replace(".json", "")
        savefilename += f"_for_{ytransformation}_transformation.json"

    savefile = save_loc / savefilename
    safe_json_dump(analysis_results, savefile)

    # save the transformed data
    if ytransformation:
        logger.info(f"Saving transformed data to {save_loc}")
        y_transformed_data = {"data": y_transformed.tolist()}
        if ytransformation == "boxcox":
            y_transformed_data["lambda"] = boxcox_lambda_param
        savefile_y = save_loc / f"{ytransformation}_y_transformed_data.json"
        safe_json_dump(y_transformed_data, savefile_y)

    # save the original data
    savefile_y = save_loc / "original_y_data.json"
    if not savefile_y.exists():
        safe_json_dump({"data": df_y.tolist()}, savefile_y)

    return {
        "savefile": str(savefile),
        "stats": analysis_results["descriptive_statistics"],
    }
