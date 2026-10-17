"""Diversity, coverage, and distribution metrics for recommender systems."""

from collections.abc import Sequence

import numpy as np


def catalog_coverage(recommendations: dict[int, Sequence[int]], num_total_items: int) -> float:
    """Calculates the proportion of distinct items recommended across all users."""
    if num_total_items <= 0:
        return 0.0
    unique_rec_items = set()
    for preds in recommendations.values():
        unique_rec_items.update(preds)
    return float(len(unique_rec_items) / num_total_items)


def gini_index(item_frequencies: Sequence[float]) -> float:
    """Calculates Gini Index of recommendation distribution (0 = perfectly equal, 1 = concentrated).

    Formula:
        G = (2 * sum_{i=1}^n i * y_i) / (n * sum_{i=1}^n y_i) - (n + 1) / n
    where y is sorted in ascending order.
    """
    y = np.sort(np.asarray(item_frequencies, dtype=np.float64))
    n = len(y)
    if n == 0 or np.sum(y) == 0:
        return 0.0
    cumulative_rank = np.arange(1, n + 1)
    return float((2.0 * np.sum(cumulative_rank * y)) / (n * np.sum(y)) - (n + 1.0) / n)


def average_recommendation_popularity(
    recommendations: dict[int, Sequence[int]],
    item_train_counts: np.ndarray,
) -> float:
    """Calculates average popularity (training interaction count) of recommended items."""
    total_count = 0.0
    num_recommendations = 0
    for preds in recommendations.values():
        for item in preds:
            if 0 <= item < len(item_train_counts):
                total_count += item_train_counts[item]
                num_recommendations += 1
    return float(total_count / num_recommendations) if num_recommendations > 0 else 0.0


def long_tail_share(
    recommendations: dict[int, Sequence[int]],
    tail_item_set: set[int],
) -> float:
    """Calculates the proportion of recommended items belonging to the long-tail set."""
    tail_hits = 0
    total_recs = 0
    for preds in recommendations.values():
        for item in preds:
            total_recs += 1
            if item in tail_item_set:
                tail_hits += 1
    return float(tail_hits / total_recs) if total_recs > 0 else 0.0
