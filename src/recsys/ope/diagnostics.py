"""Diagnostic tools for off-policy evaluation: ESS, weight extremes, overlap checks."""


import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """Calculates Effective Sample Size (ESS) = (sum(w))^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w)
    sq_sum = np.sum(w ** 2)
    if sq_sum <= 0:
        return 0.0
    return float((w_sum ** 2) / sq_sum)


def check_support_overlap(
    eval_probs: np.ndarray,
    logging_probs: np.ndarray,
    tol: float = 1e-12,
) -> tuple[int, float]:
    """Checks for common support violations where eval policy selects action with zero logging propensity.

    Returns:
        Tuple of (violation_count, violation_percentage).
    """
    ep = np.asarray(eval_probs, dtype=np.float64)
    lp = np.asarray(logging_probs, dtype=np.float64)

    violations = np.sum((ep > tol) & (lp <= tol))
    pct = (float(violations) / len(ep)) * 100.0 if len(ep) > 0 else 0.0
    return int(violations), pct


def summarize_importance_weights(weights: np.ndarray) -> dict[str, float]:
    """Summarizes importance weight diagnostics."""
    w = np.asarray(weights, dtype=np.float64)
    if len(w) == 0:
        return {}

    return {
        "mean_weight": float(np.mean(w)),
        "std_weight": float(np.std(w)),
        "max_weight": float(np.max(w)),
        "min_weight": float(np.min(w)),
        "median_weight": float(np.median(w)),
        "p95_weight": float(np.percentile(w, 95)),
        "p99_weight": float(np.percentile(w, 99)),
        "ess": effective_sample_size(w),
        "ess_ratio": float(effective_sample_size(w) / len(w)),
    }
