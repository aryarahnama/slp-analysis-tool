import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import ks_2samp

from config import (BINS, SMALL_FONT_SIZE, FONT_SIZE, LARGE_DATASET, MAX_BOOTSTRAP_ITERATIONS, FIGURE_SIZE)


def plot_hist(data, title, path):
    """Generate and save a histogram with descriptive statistics.

    Discrete data is displayed as a probability histogram, while continuous
    data is displayed as a density histogram with a kernel density estimate.
    Descriptive statistics are displayed alongside the histogram.

    Args:
        data: Numeric observations to plot.
        title: Title for the generated figure.
        path: File path where the figure will be saved.
    """

    sns.set_theme(style="ticks")

    fig = Figure(figsize=FIGURE_SIZE)
    FigureCanvas(fig)

    ax_hist, ax_stats = fig.subplots(1, 2, gridspec_kw={"width_ratios": [4, 1]})

    ax_stats.axis("off")

    is_discrete = all(float(x).is_integer() for x in data)

    if is_discrete:
        sns.histplot(data, discrete=True, kde=False, stat="probability", ax=ax_hist)
    else:
        sns.histplot(data, bins=BINS, kde=True, stat="density", ax=ax_hist)

    ax_hist.set_title(title, fontsize=FONT_SIZE)

    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Density" if not is_discrete else "Probability")

    stats_text = pd.Series(data).describe().to_string()

    ax_stats.text(
        0,
        0.5,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=FONT_SIZE,
        family="monospace",
        va="center",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
        ),
    )

    fig.tight_layout()
    fig.savefig(path)


def plot_overlay(data_a, data_b, label_a, label_b, title, path):
    """Generate and save an overlaid histogram for two datasets.

    Both datasets use the same histogram bins for continuous data, allowing
    their distributions to be compared directly. Discrete data is plotted
    using probability-based histograms.

    Args:
        data_a: Numeric observations for the first distribution.
        data_b: Numeric observations for the second distribution.
        label_a: Legend label for the first distribution.
        label_b: Legend label for the second distribution.
        title: Title for the generated figure.
        path: File path where the figure will be saved.
    """

    sns.set_theme(style="ticks", palette="prism")

    fig = Figure(figsize=FIGURE_SIZE)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    combined = np.concatenate([data_a, data_b])
    bins = np.histogram_bin_edges(combined, bins=BINS)
    is_discrete = np.all(np.isclose(combined, np.round(combined)))
    size = len(combined)

    if is_discrete:
        sns.histplot(
            data_a, discrete=True, stat="probability",
            color="blue", alpha=0.4, label=label_a, kde=False, ax=ax
        )
        sns.histplot(
            data_b, discrete=True, stat="probability",
            color="red", alpha=0.4, label=label_b, kde=False, ax=ax
        )
    else:
        sns.histplot(
            data_a, bins=bins, stat="density",
            color="blue", alpha=0.4, label=label_a, kde=True, ax=ax
        )
        sns.histplot(
            data_b, bins=bins, stat="density",
            color="red", alpha=0.4, label=label_b, kde=True, ax=ax
        )

    ax.legend(loc=("best" if size < LARGE_DATASET else "upper right"))
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if not is_discrete else "Probability")

    fig.tight_layout()
    fig.savefig(path)


def plot_ecdf(data, title, path):
    """Generate and save an empirical cumulative distribution function plot.

    Args:
        data: Numeric observations to plot.
        title: Title for the generated figure.
        path: File path where the figure will be saved.
    """

    sns.set_theme(style="ticks")

    fig = Figure(figsize=FIGURE_SIZE)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    sns.ecdfplot(data, ax=ax)

    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Proportion")

    fig.tight_layout()
    fig.savefig(path)


def plot_ecdf_overlay(data_a, data_b, label_a, label_b, title, path):
    """Generate and save overlaid ECDF plots with the KS statistic.

    The empirical cumulative distributions of the two datasets are plotted
    together. The location and magnitude of the two-sample Kolmogorov-Smirnov
    statistic are also displayed when the statistic can be computed.

    Args:
        data_a: Numeric observations for the first distribution.
        data_b: Numeric observations for the second distribution.
        label_a: Legend label for the first distribution.
        label_b: Legend label for the second distribution.
        title: Title for the generated figure.
        path: File path where the figure will be saved.
    """

    sns.set_theme(style="ticks", palette="prism")

    fig = Figure(figsize=FIGURE_SIZE)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    size = len(data_a) + len(data_b)

    sns.ecdfplot(data_a, color="blue", label=label_a, ax=ax)
    sns.ecdfplot(data_b, color="red", label=label_b, ax=ax)

    try:
        ks = ks_2samp(data_a, data_b)
    except Exception:
        ks = None

    if ks is not None and hasattr(ks, "statistic_location") and hasattr(ks, "statistic"):
        x_ks, ks_stat = ks.statistic_location, ks.statistic

        # ECDF values at KS location
        y_a = np.searchsorted(np.sort(data_a), x_ks, side="right") / len(data_a)
        y_b = np.searchsorted(np.sort(data_b), x_ks, side="right") / len(data_b)

        combined = np.concatenate([data_a, data_b])
        is_discrete = np.all(np.isclose(combined, np.round(combined)))
        x_value = x_ks if not is_discrete else x_ks + 0.5

        # Draw KS distance
        ax.vlines(
            x_value,
            min(y_a, y_b),
            max(y_a, y_b),
            colors="black",
            linestyles="--",
            linewidth=2,
        )

        ax.annotate(
            f"KS = {ks_stat:.3f}",
            xy=(x_value, (y_a + y_b) / 2),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=SMALL_FONT_SIZE,
            color="black",
        )

    ax.legend(loc=("best" if size < LARGE_DATASET else "upper left"))
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Proportion")

    fig.tight_layout()
    fig.savefig(path)


def plot_stats_summary(stats, title, path):
    """Generate and save a summary of distribution comparison statistics.

    The summary includes Kolmogorov-Smirnov and Mann-Whitney U test results,
    along with effect-size estimates, standard deviations, confidence
    intervals, and bootstrap iteration counts when available.

    Args:
        stats: Dictionary containing hypothesis test and effect-size results.
            If None, the figure indicates that there is insufficient data.
        title: Title for the generated figure.
        path: File path where the figure will be saved.
    """

    sns.set_theme(style="white")

    fig = Figure(figsize=FIGURE_SIZE)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.axis("off")

    if " " in title.strip():
        metric = title.strip().split(" ")[1]
    else:
        metric = "x"  # refer to the metric as just 'x' if we cannot cleanly find it

    if stats is None:
        text = "Not enough data"
    else:
        confidence = int(round(stats["confidence"] * 100))
        has_cd_std = stats["cliffs_delta_std"] is not None
        has_hl_std = stats["hodges_lehmann_std"] is not None
        has_iqr_std = stats["iqr_difference_std"] is not None
        has_skew_std = stats["skewness_difference_std"] is not None
        is_cd_std_estimate = int(stats["hl_bootstrap_iterations"]) != MAX_BOOTSTRAP_ITERATIONS
        is_hl_std_estimate = int(stats["hl_bootstrap_iterations"]) != MAX_BOOTSTRAP_ITERATIONS
        is_iqr_std_estimate = int(stats["iqr_bootstrap_iterations"]) != MAX_BOOTSTRAP_ITERATIONS
        is_skew_std_estimate = int(stats["skew_bootstrap_iterations"]) != MAX_BOOTSTRAP_ITERATIONS
        has_cd_ci = stats["lower_bound_cliffs_delta_ci"] is not None and stats["upper_bound_cliffs_delta_ci"] is not None
        has_hl_ci = stats["lower_bound_hodges_lehmann_ci"] is not None and stats["upper_bound_hodges_lehmann_ci"] is not None
        has_iqr_ci = stats["lower_bound_iqr_diff_ci"] is not None and stats["upper_bound_iqr_diff_ci"] is not None
        has_skew_ci = stats["lower_bound_skew_diff_ci"] is not None and stats["upper_bound_skew_diff_ci"] is not None

        text = (
            f"Hypothesis Tests:\n"
            f"KS Statistic: {stats['ks_stat']:.4f}    [at {metric} = {stats['ks_stat_location']:.4f}]\n"
            f"P-value: {stats['ks_p_value']:.4e}\n\n"
            f"MW Statistic: {stats['mw_stat']:.4f}\n"
            f"P-value: {stats['mw_p_value']:.4e}\n\n"
            f"Effect Size:\n"
            f"Cliff's delta: {stats['cliffs_delta']:.4f}\n"
            + (
                f"Cliff's delta Standard Deviation: {stats['cliffs_delta_std']:.4f}"
                + (f" (estimate)\n"  if is_cd_std_estimate else "\n")
                if has_cd_std else ""
            )
            + (
                f"{confidence}% CI for Cliff's delta: "
                f"[{stats['lower_bound_cliffs_delta_ci']:.4f}, "
                f"{stats['upper_bound_cliffs_delta_ci']:.4f}] "
                f"on {int(stats['cd_bootstrap_iterations'])} bootstrap samples\n\n"
                if has_cd_ci else "\n"
            )
            + f"Hodges–Lehmann estimator: {stats['hodges_lehmann']:.4f}\n"
            + (
                f"Hodges–Lehmann Standard Deviation: {stats['hodges_lehmann_std']:.4f}"
                + (f" (estimate)\n"  if is_hl_std_estimate else "\n")
                if has_hl_std else ""
            )
            + (
                f"{confidence}% CI for Hodges–Lehmann: "
                f"[{stats['lower_bound_hodges_lehmann_ci']:.4f}, "
                f"{stats['upper_bound_hodges_lehmann_ci']:.4f}] "
                f"on {int(stats['hl_bootstrap_iterations'])} bootstrap samples\n\n"
                if has_hl_ci else "\n"
            )
            + f"IQR Difference: {stats['iqr_difference']:.4f}\n"
            + (
                f"IQR Difference Standard Deviation: {stats['iqr_difference_std']:.4f}"
                + (f" (estimate)\n"  if is_iqr_std_estimate else "\n")
                if has_iqr_std else ""
            )
            + (
                f"{confidence}% CI for IQR Difference: "
                f"[{stats['lower_bound_iqr_diff_ci']:.4f}, "
                f"{stats['upper_bound_iqr_diff_ci']:.4f}] "
                f"on {int(stats['iqr_bootstrap_iterations'])} bootstrap samples\n\n"
                if has_iqr_ci else "\n"
            )
            + f"Skewness Difference: {stats['skewness_difference']:.4f}\n"
            + (
                f"Skew Difference Standard Deviation: {stats['skewness_difference_std']:.4f}"
                + (f" (estimate)\n"  if is_skew_std_estimate else "\n")
                if has_skew_std else ""
            )
            + (
                f"{confidence}% CI for Skew Difference: "
                f"[{stats['lower_bound_skew_diff_ci']:.4f}, "
                f"{stats['upper_bound_skew_diff_ci']:.4f}] "
                f"on {int(stats['skew_bootstrap_iterations'])} bootstrap samples\n\n"
                if has_skew_ci else "\n"
            )
        )

    ax.text(
        0.05, 0.95, text,
        fontsize=FONT_SIZE,
        va="top",
        family="monospace"
    )

    ax.set_title(title, fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(path)
