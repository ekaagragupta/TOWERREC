"""Vectorized offline ranking metrics: Recall@K, Precision@K, NDCG@K, MAP@K, HitRate@K, MRR."""

from collections.abc import Sequence

import numpy as np


def recall_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Calculates Recall@K for a single user."""
    if not actual or k <= 0:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]
    hits = sum(1 for item in pred_k if item in actual_set)
    return hits / float(len(actual_set))


def precision_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Calculates Precision@K for a single user."""
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]
    hits = sum(1 for item in pred_k if item in actual_set)
    return hits / float(k)


def hit_rate_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Calculates HitRate@K (1 if any hit in top K, else 0)."""
    actual_set = set(actual)
    pred_k = predicted[:k]
    for item in pred_k:
        if item in actual_set:
            return 1.0
    return 0.0


def mrr_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Calculates Mean Reciprocal Rank (MRR) at K."""
    actual_set = set(actual)
    pred_k = predicted[:k]
    for rank, item in enumerate(pred_k, start=1):
        if item in actual_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain (NDCG) at K with binary relevance."""
    if not actual or k <= 0:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]

    dcg = 0.0
    for rank, item in enumerate(pred_k, start=1):
        if item in actual_set:
            dcg += 1.0 / np.log2(rank + 1.0)

    # Ideal DCG (all hits up to min(k, len(actual)))
    n_ideal = min(k, len(actual_set))
    idcg = sum(1.0 / np.log2(r + 1.0) for r in range(1, n_ideal + 1))

    return float(dcg / idcg) if idcg > 0.0 else 0.0


def map_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Calculates Mean Average Precision (MAP) at K."""
    if not actual or k <= 0:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]

    score = 0.0
    hits = 0
    for rank, item in enumerate(pred_k, start=1):
        if item in actual_set:
            hits += 1
            score += hits / float(rank)

    n_ideal = min(k, len(actual_set))
    return float(score / n_ideal) if n_ideal > 0 else 0.0


def evaluate_user_recommendations(
    ground_truth: dict[int, Sequence[int]],
    recommendations: dict[int, Sequence[int]],
    k_list: Sequence[int] = (10, 20, 50),
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluates recommendations across all users for multiple cutoffs K.

    Args:
        ground_truth: Mapping from user_id to ground truth target item list.
        recommendations: Mapping from user_id to predicted item ranked list.
        k_list: Tuple or list of cutoff thresholds K.

    Returns:
        Tuple of (mean_metrics_dict, per_user_metrics_dict).
    """
    users = [u for u in ground_truth if u in recommendations and len(ground_truth[u]) > 0]
    n_users = len(users)

    if n_users == 0:
        return {}, {}

    per_user_metrics: dict[str, np.ndarray] = {}
    mean_metrics: dict[str, float] = {}

    for k in k_list:
        rec_arr = np.zeros(n_users, dtype=np.float64)
        prec_arr = np.zeros(n_users, dtype=np.float64)
        ndcg_arr = np.zeros(n_users, dtype=np.float64)
        map_arr = np.zeros(n_users, dtype=np.float64)
        hit_arr = np.zeros(n_users, dtype=np.float64)
        mrr_arr = np.zeros(n_users, dtype=np.float64)

        for idx, u in enumerate(users):
            act = ground_truth[u]
            pred = recommendations[u]
            rec_arr[idx] = recall_at_k(act, pred, k)
            prec_arr[idx] = precision_at_k(act, pred, k)
            ndcg_arr[idx] = ndcg_at_k(act, pred, k)
            map_arr[idx] = map_at_k(act, pred, k)
            hit_arr[idx] = hit_rate_at_k(act, pred, k)
            mrr_arr[idx] = mrr_at_k(act, pred, k)

        per_user_metrics[f"recall@{k}"] = rec_arr
        per_user_metrics[f"precision@{k}"] = prec_arr
        per_user_metrics[f"ndcg@{k}"] = ndcg_arr
        per_user_metrics[f"map@{k}"] = map_arr
        per_user_metrics[f"hit_rate@{k}"] = hit_arr
        per_user_metrics[f"mrr@{k}"] = mrr_arr

        mean_metrics[f"recall@{k}"] = float(np.mean(rec_arr))
        mean_metrics[f"precision@{k}"] = float(np.mean(prec_arr))
        mean_metrics[f"ndcg@{k}"] = float(np.mean(ndcg_arr))
        mean_metrics[f"map@{k}"] = float(np.mean(map_arr))
        mean_metrics[f"hit_rate@{k}"] = float(np.mean(hit_arr))
        mean_metrics[f"mrr@{k}"] = float(np.mean(mrr_arr))

    return mean_metrics, per_user_metrics
