import json
from dataclasses import dataclass
from pathlib import Path as pt
import numpy as np

# import pandas as pd
from scipy import stats
from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger


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


def main(args: Args):
    logger.info(f"{args.property_column=}")
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

    # 1. Descriptive Statistics
    desc_stats = df[property_column].describe().to_dict()

    # 2. Histogram data
    hist, bin_edges = np.histogram(df[property_column], bins="auto")
    hist_data = {"counts": hist.tolist(), "bin_edges": bin_edges.tolist()}

    # 3. Box Plot data
    box_plot_data = {
        "min": float(df[property_column].min()),
        "q1": float(df[property_column].quantile(0.25)),
        "median": float(df[property_column].median()),
        "q3": float(df[property_column].quantile(0.75)),
        "max": float(df[property_column].max()),
    }

    # 4. Q-Q Plot data
    qq_data = stats.probplot(df[property_column], dist="norm")
    qq_plot_data = {
        "theoretical_quantiles": qq_data[0][0].tolist(),
        "sample_quantiles": qq_data[0][1].tolist(),
    }

    # 5. Shapiro-Wilk Test for Normality
    # scipy.stats.shapiro: For N > 5000, computed p-value may not be accurate.
    # stat, p_value = stats.shapiro(df[property_column])
    # shapiro_test = {"statistic": float(stat), "p_value": float(p_value)}

    # Perform the Anderson-Darling test
    ad_result = stats.anderson(df[property_column])

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
    skewness = float(df[property_column].skew())
    kurtosis = float(df[property_column].kurtosis())

    # 7. KDE data
    kde = stats.gaussian_kde(df[property_column])
    x_range = np.linspace(df[property_column].min(), df[property_column].max(), 100)
    kde_data = {"x": x_range.tolist(), "y": kde(x_range).tolist()}

    # Combine all data
    analysis_results = {
        "descriptive_statistics": desc_stats,
        "histogram": hist_data,
        "box_plot": box_plot_data,
        "qq_plot": qq_plot_data,
        # "shapiro_wilk_test": shapiro_test,
        "anderson_darling_test": anderson_darling_test,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "kde": kde_data,
    }

    # Save to JSON file
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
