"""Bootstrap confidence intervals and paired statistical significance testing."""

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculates percentile bootstrap confidence interval for a metric vector.

    Args:
        values: 1D array of per-user metric values.
        n_bootstraps: Number of bootstrap iterations.
        confidence_level: Desired confidence level (e.g. 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(np.mean(values))
    rng = np.random.default_rng(seed)
    n = len(values)

    boot_means = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    return point_estimate, ci_lower, ci_upper


def paired_significance_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, float]:
    """Performs paired t-test and Wilcoxon signed-rank test between two models across users.

    Args:
        scores_a: Per-user metric array for Model A.
        scores_b: Per-user metric array for Model B.

    Returns:
        Dictionary with diff_mean, t_stat, t_pvalue, wilcoxon_stat, wilcoxon_pvalue.
    """
    diff = scores_b - scores_a
    mean_diff = float(np.mean(diff))

    # Paired Student's t-test
    t_stat, t_pvalue = stats.ttest_rel(scores_b, scores_a)

    # Wilcoxon signed-rank test (handles non-normality)
    # Check if differences are all zero
    if np.all(diff == 0):
        w_stat, w_pvalue = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(scores_b, scores_a)
            w_stat, w_pvalue = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pvalue = 0.0, 1.0

    return {
        "mean_diff": mean_diff,
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "t_pvalue": float(t_pvalue) if not np.isnan(t_pvalue) else 1.0,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pvalue,
        "is_significant_p05": bool(t_pvalue < 0.05),
    }
