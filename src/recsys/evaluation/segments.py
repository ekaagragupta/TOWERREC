"""User activity and item popularity segmentation for sliced metric analysis."""

from collections.abc import Sequence

import numpy as np


def compute_user_activity_segments(
    user_train_counts: np.ndarray,
    eval_user_indices: Sequence[int],
) -> dict[str, list[int]]:
    """Partitions evaluation users into tertiles: cold (bottom 33%), warm (middle 33%), heavy (top 33%).

    Args:
        user_train_counts: Array of train interaction counts indexed by user_idx.
        eval_user_indices: List of users present in the evaluation split.

    Returns:
        Dictionary mapping 'cold', 'warm', 'heavy' to lists of user indices.
    """
    counts = [user_train_counts[u] for u in eval_user_indices]
    if len(counts) == 0:
        return {"cold": [], "warm": [], "heavy": []}

    q33, q67 = np.percentile(counts, [33.33, 66.67])

    segments: dict[str, list[int]] = {"cold": [], "warm": [], "heavy": []}
    for u in eval_user_indices:
        c = user_train_counts[u]
        if c <= q33:
            segments["cold"].append(u)
        elif c <= q67:
            segments["warm"].append(u)
        else:
            segments["heavy"].append(u)

    return segments


def compute_item_popularity_segments(
    item_train_counts: np.ndarray,
) -> tuple[set[int], set[int], set[int]]:
    """Partitions items into Head (top 20% by count), Torso (next 30%), Tail (bottom 50%).

    Returns:
        Tuple of (head_items_set, torso_items_set, tail_items_set).
    """
    sorted_items = np.argsort(-item_train_counts)  # descending
    n = len(sorted_items)

    head_cutoff = int(0.20 * n)
    torso_cutoff = int(0.50 * n)

    head_items = set(sorted_items[:head_cutoff])
    torso_items = set(sorted_items[head_cutoff:torso_cutoff])
    tail_items = set(sorted_items[torso_cutoff:])

    return head_items, torso_items, tail_items
