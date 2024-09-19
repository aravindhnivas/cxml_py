import json
from dataclasses import dataclass
from pathlib import Path as pt
import numpy as np

# import pandas as pd
import pandas as pd
from scipy import stats
from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger
from umdalib.training.utils import get_transformed_data


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
    transformed_data["original"] = data

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
        boxcox_transformed, boxcox_lambda_param = get_transformed_data(data, "boxcox")
        transformed_data["boxcox"] = boxcox_transformed

    # Yeo-Johnson Transformation (Can handle zero and negative values)
    yeo_johnson_transformed = get_transformed_data(data, "yeo_johnson")

    transformed_data["yeo_johnson"] = yeo_johnson_transformed

    # Compute skewness for each transformation
    computed_skewness = {}
    for method, transformed in transformed_data.items():
        skew = stats.skew(transformed)
        logger.info(f"{method}: {skew:.2f}")
        computed_skewness[method] = skew

    # Find the key with the minimum skewness value
    lowest_skew_key = None
    if computed_skewness:
        lowest_skew_key = min(computed_skewness, key=computed_skewness.get)
        logger.info(f"Lowest skewness transformation: {lowest_skew_key}")
        return computed_skewness, lowest_skew_key, transformed_data[lowest_skew_key]

    logger.info("No valid skewness transformations found.")
    return None, None, None


def main(args: Args):
    global boxcox_lambda_param
    boxcox_lambda_param = None

    save_loc = pt(args.save_loc)
    df = read_as_ddf(
        args.filetype,
        args.filename,
        args.key,
        use_dask=args.use_dask,
        computed=True,
    )

    # Assuming your target property is named 'property'
    property_column = args.property_column
    df_y = df[property_column]

    if args.auto_transform_data:
        computed_skewness, lowest_skew_key, y_transformed = get_skew_and_transformation(
            df_y
        )
        logger.info(f"{lowest_skew_key=}\n{computed_skewness=}")
        if lowest_skew_key:
            df_y = pd.Series(y_transformed)
    elif args.ytransformation:
        if args.ytransformation == "boxcox":
            df_y, boxcox_lambda_param = get_transformed_data(df_y, args.ytransformation)
        else:
            y_values = get_transformed_data(df_y.values, args.ytransformation)
            df_y = pd.Series(y_values)

    # logger.info(f"Skewness after transformation: {skewness:.2f}")

    # 1. Descriptive Statistics
    desc_stats = df_y.describe().to_dict()

    # 2. Histogram data
    hist, bin_edges = np.histogram(df_y, bins="auto")
    hist_data = {"counts": hist.tolist(), "bin_edges": bin_edges.tolist()}

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
    kde_data = {"x": x_range.tolist(), "y": kde(x_range).tolist()}

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
    }

    if args.auto_transform_data:
        analysis_results["applied_transformation"] = lowest_skew_key
        if boxcox_lambda_param:
            analysis_results["boxcox_lambda"] = boxcox_lambda_param

    # Save to JSON file
    if not save_loc.exists():
        save_loc.mkdir(parents=True)

    savefile = save_loc / args.savefilename
    with open(savefile, "w") as f:
        json.dump(analysis_results, f, indent=2)

    logger.info(
        "Analysis complete. Results saved to 'molecular_property_analysis.json'"
    )

    return {
        "savefile": str(savefile),
        "stats": desc_stats,
    }
