from scipy.stats import ks_2samp, mannwhitneyu, gaussian_kde, iqr

import numpy as np

from config import (MIN_STAT_CUTOFF, MAX_HL_ITERATIONS, DEFAULT_CONFIDENCE, MAX_BOOTSTRAP_ITERATIONS, MAX_BOOTSTRAP_ITERATIONS_LARGE, MAX_CARTESIAN_PRODUCT_SIZE_NAIVE,
                    MIN_BOOTSTRAP_ITERATIONS, MAX_CARTESIAN_PRODUCT_SIZE, CARTESIAN_PRODUCT_CUTOFF, VERY_LARGE_DATASET, EXTREMELY_LARGE_DATASET, TOTAL_DATA_CUTOFF, NUMERICAL_EPSILON)


def iqr_difference(a, b):
    """Calculate the difference in interquartile range between two samples.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        The interquartile range of a minus the interquartile range of b.
    """

    return iqr(a) - iqr(b)


def skewness(x, u=0.75):
    """Calculate Bowley skewness for a sample.

    Uses the lower quantile, median, and upper quantile defined by ``u`` to
    measure the asymmetry of the sample around its median.

    Args:
        x: Sample of observations.
        u: Upper quantile used to calculate the skewness. The corresponding
            lower quantile is ``1 - u``.

    Returns:
        The Bowley skewness of the sample, or 0 if the upper and lower
        quantiles are equal.
    """

    x = np.sort(np.asarray(x))

    q_low = np.quantile(x, 1 - u)  # lower quantile
    q_med = np.median(x)
    q_high = np.quantile(x, u)     # upper quantile

    return (q_high + q_low - 2*q_med) / (q_high - q_low) if not np.isclose(q_high - q_low, 0.0) else 0.0


def skewness_difference(a, b):
    """Calculate the difference in Bowley skewness between two samples.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        The skewness of a minus the skewness of b.
    """

    return skewness(a) - skewness(b)


def cliffs_delta_manual(a, b, m=None, n=None):
    """Calculate Cliff's delta using explicit pairwise comparisons.

    Compares every observation in the first sample with every observation in
    the second sample and calculates the difference between the proportions
    of pairs where the first observation is greater than or less than the
    second.

    Args:
        a: First sample.
        b: Second sample.
        m: Number of observations in a. Defaults to len(a).
        n: Number of observations in b. Defaults to len(b).

    Returns:
        Cliff's delta, ranging from -1 to 1. Returns 0 for an empty sample.
    """

    if m is None or n is None:
        m, n = len(a), len(b)

    N = m * n

    if N == 0:
        return 0.0

    greater = 0
    less = 0

    for x in a:
        greater += np.sum(x > b)
        less += np.sum(x < b)

    return (greater - less) / N


def cliffs_delta_variance(a, b, delta=None):
    """Estimate the variance of Cliff's delta.

    Constructs the pairwise comparison matrix between the two samples and
    uses row and column means to estimate the variance of the Cliff's delta
    statistic.

    Args:
        a: First sample.
        b: Second sample.
        delta: Precomputed Cliff's delta. If None, it is calculated from the
            pairwise comparison matrix.

    Returns:
        The estimated variance of Cliff's delta.
    """

    m, n = len(a), len(b)
    N = m * n

    # Pairwise difference matrix
    diff = np.subtract.outer(a, b).astype(np.float32)
    delta_ij = np.sign(diff)

    if delta is None:
        delta = delta_ij.mean()

    # Row/column means
    delta_r = delta_ij.mean(axis=1)
    delta_c = delta_ij.mean(axis=0)

    # Variance estimator
    term1 = (n ** 2) * np.sum((delta_r - delta) ** 2)
    term2 = (m ** 2) * np.sum((delta_c - delta) ** 2)
    term3 = np.sum((delta_ij - delta) ** 2)
    variance = (term1 + term2 - term3) / (N * (m - 1) * (n - 1))

    return variance


def cliffs_delta(a, b, is_manual=False, compute_variance=False):
    """Calculate Cliff's delta and optionally estimate its variance.

    Uses the Mann–Whitney U statistic to calculate Cliff's delta by default,
    avoiding explicit construction of the full pairwise comparison matrix.
    The manual implementation can be selected when explicit pairwise
    comparisons are desired. For variance estimation, the full pairwise
    calculation is used for smaller datasets and a subsample is used for
    larger datasets.

    Args:
        a: First sample.
        b: Second sample.
        is_manual: Whether to calculate Cliff's delta using the manual
            pairwise-comparison implementation.
        compute_variance: Whether to also calculate and return a variance
            estimate.

    Returns:
        Cliff's delta if compute_variance is False. Otherwise, returns a
        tuple containing Cliff's delta and its estimated variance.
    """

    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    N = m * n

    if N == 0:
        return (0.0, 0.0) if compute_variance else 0.0

    if is_manual:
        return cliffs_delta_manual(a, b, m=m, n=n)

    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    delta = (2 * u) / N - 1

    if not compute_variance:
        return delta

    min_variance = (1 - delta ** 2) / (N - 1)

    if N < MAX_CARTESIAN_PRODUCT_SIZE_NAIVE:
        variance = cliffs_delta_variance(a, b, delta)
    else:
        scale = min(1.0, np.sqrt(MAX_CARTESIAN_PRODUCT_SIZE_NAIVE / N))
        m_sub = max(1, int(m * scale))
        n_sub = max(1, int(n * scale))

        rng = np.random.default_rng()
        a_sub = rng.choice(a, size=m_sub, replace=False)
        b_sub = rng.choice(b, size=n_sub, replace=False)

        variance = cliffs_delta_variance(a_sub, b_sub)

    variance = max(variance, min_variance)

    return delta, variance


def hodges_lehmann_naive(a, b, compute_variance=False):
    """Calculate the Hodges–Lehmann estimator using all pairwise differences.

    Constructs the Cartesian product of the two samples, calculates every
    pairwise difference, and takes the median as the Hodges–Lehmann estimate.
    This implementation is straightforward but requires memory proportional
    to the product of the two sample sizes.

    Args:
        a: First sample.
        b: Second sample.
        compute_variance: Whether to also estimate the variance using a
            kernel density estimate around the Hodges–Lehmann statistic.

    Returns:
        The Hodges–Lehmann estimate, or a tuple containing the estimate and
        its variance when compute_variance is True.
    """

    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    N = m * n

    if N == 0:
        return (0.0, 0.0) if compute_variance else 0.0

    diffs = np.subtract.outer(a, b).astype(np.float32).ravel()
    hl = np.median(diffs)

    if not compute_variance:
        return hl

    diff_std = np.std(diffs)

    if np.isclose(diff_std, 0.0):
        variance = 0.0
    else:
        kde = gaussian_kde(diffs)
        f_theta = kde.evaluate([hl])[0]

        variance = 1.0 / (4 * N * (f_theta ** 2)) if not np.isclose(f_theta, 0.0) else 0.0

    return hl, variance


def hodges_lehmann_efficient(a, b, compute_variance=False, bandwidth=None):
    """Calculate the Hodges–Lehmann estimator without constructing all differences.

    Uses sorted samples and binary-search-style counting of pairwise
    differences to efficiently locate the median pairwise difference. This
    avoids explicitly storing the full Cartesian product and is therefore
    suitable for substantially larger datasets.

    Args:
        a: First sample.
        b: Second sample.
        compute_variance: Whether to estimate the variance of the estimator.
        bandwidth: Bandwidth used for local density estimation. If None, a
            bandwidth is estimated from the sample variances and data size.

    Returns:
        The Hodges–Lehmann estimate, or a tuple containing the estimate and
        its variance when compute_variance is True.
    """

    a, b = np.asarray(a), np.asarray(b)

    combined = np.concatenate([a, b])
    is_discrete = np.all(np.isclose(combined, np.round(combined)))

    if is_discrete:
        a, b = np.sort(np.round(a).astype(np.int32)), np.sort(np.round(b).astype(np.int32))
    else:
        a, b = np.sort(a.astype(np.float32)), np.sort(b.astype(np.float32))

    swapped = False

    # Iterate over smaller array if possible
    if len(a) > len(b):
        a, b = b, a
        swapped = True

    m, n = len(a), len(b)
    N = m * n

    k1 = (N - 1) // 2
    k2 = N // 2

    low = a[0] - b[-1]
    high = a[-1] - b[0]

    def count_leq(x):
        count = 0
        j = 0

        for ai in a:
            threshold = ai - x

            while j < n and b[j] < threshold:
                j += 1

            count += (n - j)

        return count

    def kth_difference(k):
        lo, hi = low, high

        if is_discrete:
            while lo < hi:
                mid = (lo + hi) // 2

                if count_leq(mid) <= k:
                    lo = mid + 1
                else:
                    hi = mid

            return lo
        else:
            for _ in range(MAX_HL_ITERATIONS):
                mid = (lo + hi) / 2

                if count_leq(mid) <= k:
                    lo = np.nextafter(mid, hi)
                else:
                    hi = mid

            return hi

    # Final HL estimate
    x1 = kth_difference(k1)
    x2 = kth_difference(k2)
    hl = (x1 + x2) / 2

    if swapped:
        hl = -hl

    if not compute_variance:
        return hl

    # Density estimation near HL point
    if bandwidth is None:
        scale_const = 0.9

        sigma = np.sqrt(np.var(a) + np.var(b))

        bandwidth = scale_const * sigma * (N ** (-1/5))

        bandwidth = max(bandwidth, NUMERICAL_EPSILON)

    cdf_lo, cdf_hi = count_leq(hl - bandwidth) / N, count_leq(hl + bandwidth) / N
    f_theta = (cdf_hi - cdf_lo) / (2 * bandwidth)
    variance = 1.0 / (4 * N * (f_theta ** 2)) if not np.isclose(f_theta, 0.0) else 0.0

    return hl, variance


def hodges_lehmann(a, b, compute_variance=False):
    """Calculate the Hodges–Lehmann estimator using an adaptive implementation.

    Selects the naive implementation for smaller Cartesian products and the
    memory-efficient implementation for larger products.

    Args:
        a: First sample.
        b: Second sample.
        compute_variance: Whether to also estimate and return the variance.

    Returns:
        The Hodges–Lehmann estimate, or a tuple containing the estimate and
        its variance when compute_variance is True.
    """

    return hodges_lehmann_naive(a, b, compute_variance=compute_variance) if (len(a) * len(b)) < MAX_CARTESIAN_PRODUCT_SIZE_NAIVE else hodges_lehmann_efficient(a, b, compute_variance=compute_variance)


def determine_bootstrap_iterations(a, b):
    """Determine bootstrap iteration counts based on dataset size.

    Selects separate bootstrap iteration counts for Cliff's delta,
    Hodges–Lehmann, IQR difference, and skewness difference. The number of
    iterations is reduced for larger datasets to limit computational cost,
    with some statistics disabled entirely when the datasets exceed the
    configured size thresholds.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        A tuple containing the bootstrap iteration counts for Cliff's delta,
        Hodges–Lehmann, IQR difference, and skewness difference, respectively.
        A count of zero indicates that bootstrapping is disabled.
    """

    cartesian_product_size = len(a) * len(b)
    larger_dataset_size = max(len(a), len(b))
    total_data_size = len(a) + len(b)

    if larger_dataset_size < VERY_LARGE_DATASET:
        cd_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS
        iqr_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS
        skew_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS
    elif larger_dataset_size < EXTREMELY_LARGE_DATASET:
        cd_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS_LARGE
        iqr_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS_LARGE
        skew_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS_LARGE
    elif total_data_size < TOTAL_DATA_CUTOFF:
        cd_bootstrap_iterations = MIN_BOOTSTRAP_ITERATIONS
        iqr_bootstrap_iterations = MIN_BOOTSTRAP_ITERATIONS
        skew_bootstrap_iterations = MIN_BOOTSTRAP_ITERATIONS
    else:
        cd_bootstrap_iterations = 0
        iqr_bootstrap_iterations = 0
        skew_bootstrap_iterations = 0

    if cartesian_product_size < MAX_CARTESIAN_PRODUCT_SIZE_NAIVE:
        hl_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS
    elif cartesian_product_size < MAX_CARTESIAN_PRODUCT_SIZE:
        hl_bootstrap_iterations = MAX_BOOTSTRAP_ITERATIONS_LARGE
    elif cartesian_product_size < CARTESIAN_PRODUCT_CUTOFF:
        hl_bootstrap_iterations = MIN_BOOTSTRAP_ITERATIONS
    else:
        hl_bootstrap_iterations = 0

    return cd_bootstrap_iterations, hl_bootstrap_iterations, iqr_bootstrap_iterations, skew_bootstrap_iterations


def generate_ci(a, b, stat_func, confidence=DEFAULT_CONFIDENCE, n_bootstrap=MAX_BOOTSTRAP_ITERATIONS, paired=False, random_state=None):
    """Generate a bootstrap percentile confidence interval for a statistic.

    Resamples the input samples with replacement and evaluates the supplied
    statistic on each bootstrap sample. Supports both independent and paired
    bootstrap resampling.

    Args:
        a: First sample.
        b: Second sample.
        stat_func: Function that accepts two samples and returns the statistic
            to be bootstrapped.
        confidence: Confidence level for the percentile interval.
        n_bootstrap: Number of bootstrap resamples to generate.
        paired: Whether the samples should be resampled using the same indices.
        random_state: Random seed or NumPy random generator used for
            reproducible resampling.

    Returns:
        A dictionary containing the lower and upper confidence interval bounds
        under ``ci_low`` and ``ci_high``, along with the bootstrap statistic
        distribution under ``bootstrap_distribution``. Returns a dictionary
        containing None values if the inputs are invalid or an error occurs.
    """

    empty_dict = {"ci_low": None, "ci_high": None, "bootstrap_distribution": None}

    try:
        a = np.asarray(a)
        b = np.asarray(b)

        if len(a) == 0 or len(b) == 0:
            raise ValueError("Input arrays cannot be empty")

        if (confidence <= 0 or confidence >= 1) or n_bootstrap <= 0:
            return empty_dict

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        boot_stats = np.empty(n_bootstrap, dtype=np.float32)

        if paired:
            if len(a) != len(b):
                raise ValueError("Paired bootstrap requires equal-length arrays")

            n = len(a)

            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)

                a_resampled = a[idx]
                b_resampled = b[idx]

                boot_stats[i] = stat_func(a_resampled, b_resampled)
        else:
            n_a = len(a)
            n_b = len(b)

            for i in range(n_bootstrap):
                idx_a = rng.integers(0, n_a, size=n_a)
                idx_b = rng.integers(0, n_b, size=n_b)

                a_resampled = a[idx_a]
                b_resampled = b[idx_b]

                boot_stats[i] = stat_func(a_resampled, b_resampled)

        # Percentile interval
        alpha = 1.0 - confidence
        ci_low = np.quantile(boot_stats, alpha / 2)
        ci_high = np.quantile(boot_stats, 1 - alpha / 2)

        return {
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_distribution": boot_stats,
        }
    except Exception:
        return empty_dict


def compute_distribution_stats(a, b, confidence=DEFAULT_CONFIDENCE):
    """Compute distribution tests, effect sizes, and confidence intervals.

    Performs two non-parametric hypothesis tests and calculates several
    distributional effect-size measures. The hypothesis tests include the
    two-sample Kolmogorov–Smirnov test and Mann–Whitney U test. Effect sizes
    include Cliff's delta, the Hodges–Lehmann estimator, the difference in
    interquartile ranges, and the difference in Bowley skewness.

    Bootstrap confidence intervals and variance estimates are calculated for
    the effect sizes when the input dataset sizes permit the required
    computation. Bootstrap iteration counts are adaptively selected based on
    the sizes of the input samples.

    Args:
        a: First sample.
        b: Second sample.
        confidence: Confidence level used for bootstrap confidence intervals.

    Returns:
        A dictionary containing hypothesis-test statistics, p-values, effect
        sizes, estimated standard deviations, confidence interval bounds,
        and the number of bootstrap iterations used for each effect size.
        Returns None if either sample contains fewer observations than the
        minimum statistical cutoff.
    """

    if min(len(a), len(b)) < MIN_STAT_CUTOFF:
        return None

    # Kolmogorov-Smirnov test
    ks = ks_2samp(a, b, alternative="two-sided")
    ks_stat, ks_p, ks_loc = ks.statistic, ks.pvalue, ks.statistic_location

    # Mann–Whitney U test
    mw = mannwhitneyu(a, b, alternative="two-sided")
    mw_stat, mw_p = mw.statistic, mw.pvalue

    # Effect size
    cd_bootstrap, hl_bootstrap, iqr_bootstrap, skew_bootstrap = determine_bootstrap_iterations(a, b)
    has_cd_bootstrap, has_hl_bootstrap = cd_bootstrap != 0, hl_bootstrap != 0

    observed_cd, cd_var_computed = cliffs_delta(a, b, compute_variance=True)
    cd_ci = generate_ci(a, b, cliffs_delta,
                        confidence=confidence, n_bootstrap=cd_bootstrap)
    cd_var_boot = np.var(cd_ci["bootstrap_distribution"],
                     ddof=1) if cd_ci["bootstrap_distribution"] is not None else None
    has_var_boot = has_cd_bootstrap and (cd_var_boot is not None)
    cd_var = max(cd_var_computed, cd_var_boot if has_var_boot else 0)

    observed_hl, hl_var_computed = hodges_lehmann(a, b, compute_variance=True)
    hl_ci = generate_ci(a, b, hodges_lehmann,
                        confidence=confidence, n_bootstrap=hl_bootstrap)
    hl_var_boot = np.var(hl_ci["bootstrap_distribution"],
                    ddof=1) if hl_ci["bootstrap_distribution"] is not None else None
    has_hl_boot = has_hl_bootstrap and (hl_var_boot is not None)
    hl_var = max(hl_var_computed, hl_var_boot if has_hl_boot else 0)

    observed_iqr_diff = iqr_difference(a, b)
    iqr_ci = generate_ci(a, b, iqr_difference,
                         confidence=confidence, n_bootstrap=iqr_bootstrap)
    iqr_var = np.var(iqr_ci["bootstrap_distribution"],
                     ddof=1) if iqr_ci["bootstrap_distribution"] is not None else None

    observed_skew_diff = skewness_difference(a, b)
    skew_ci = generate_ci(a, b, skewness_difference,
                          confidence=confidence, n_bootstrap=skew_bootstrap)
    skew_var = np.var(skew_ci["bootstrap_distribution"],
                      ddof=1) if skew_ci["bootstrap_distribution"] is not None else None

    cd_std = np.sqrt(cd_var) if cd_var is not None else None
    hl_std = np.sqrt(hl_var) if hl_var is not None else None
    iqr_std = np.sqrt(iqr_var) if iqr_var is not None else None
    skew_std = np.sqrt(skew_var) if skew_var is not None else None

    return {
        "ks_stat": ks_stat,
        "ks_p_value": ks_p,
        "ks_stat_location": ks_loc,
        "mw_stat": mw_stat,
        "mw_p_value": mw_p,
        "confidence": confidence,
        "cliffs_delta": observed_cd,
        "cliffs_delta_std": cd_std,
        "lower_bound_cliffs_delta_ci": cd_ci["ci_low"],
        "upper_bound_cliffs_delta_ci": cd_ci["ci_high"],
        "cd_bootstrap_iterations": cd_bootstrap,
        "hodges_lehmann": observed_hl,
        "hodges_lehmann_std": hl_std,
        "lower_bound_hodges_lehmann_ci": hl_ci["ci_low"],
        "upper_bound_hodges_lehmann_ci": hl_ci["ci_high"],
        "hl_bootstrap_iterations": hl_bootstrap,
        "iqr_difference": observed_iqr_diff,
        "iqr_difference_std": iqr_std,
        "lower_bound_iqr_diff_ci": iqr_ci["ci_low"],
        "upper_bound_iqr_diff_ci": iqr_ci["ci_high"],
        "iqr_bootstrap_iterations": iqr_bootstrap,
        "skewness_difference": observed_skew_diff,
        "skewness_difference_std": skew_std,
        "lower_bound_skew_diff_ci": skew_ci["ci_low"],
        "upper_bound_skew_diff_ci": skew_ci["ci_high"],
        "skew_bootstrap_iterations": skew_bootstrap,
    }
