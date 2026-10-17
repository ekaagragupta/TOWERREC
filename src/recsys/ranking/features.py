"""Feature engineering for Stage 2 re-ranker with complete documentation."""

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "two_tower_score",
    "retrieval_rank",
    "item_popularity",
    "item_recency_age",
    "user_activity_level",
    "user_genre_affinity",
    "genre_match_count",
    "user_item_cooccurrence",
    "hour_of_day",
    "day_of_week",
]


def build_item_cooccurrence_matrix(
    train_df: pd.DataFrame,
    num_items: int,
    top_k_cooccur: int = 50,
) -> dict[int, dict[int, float]]:
    """Builds an item-item co-occurrence frequency map from training user sessions."""
    cooccur: dict[int, dict[int, int]] = {}

    user_groups = train_df.groupby("user_idx")["item_idx"].apply(list)
    for items in user_groups:
        if len(items) < 2:
            continue
        # Use recent items to avoid quadratic explosion
        recent_items = items[-20:]
        n = len(recent_items)
        for i in range(n):
            i1 = recent_items[i]
            if i1 not in cooccur:
                cooccur[i1] = {}
            for j in range(i + 1, n):
                i2 = recent_items[j]
                cooccur[i1][i2] = cooccur[i1].get(i2, 0) + 1
                if i2 not in cooccur:
                    cooccur[i2] = {}
                cooccur[i2][i1] = cooccur[i2].get(i1, 0) + 1

    # Normalize to frequencies
    norm_cooccur: dict[int, dict[int, float]] = {}
    for i1, neighbors in cooccur.items():
        total = sum(neighbors.values())
        if total > 0:
            sorted_n = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:top_k_cooccur]
            norm_cooccur[i1] = {i2: float(cnt / total) for i2, cnt in sorted_n}

    return norm_cooccur


def extract_ranking_features(
    candidate_df: pd.DataFrame,
    stats: dict[str, any],
    genre_matrix: np.ndarray,
    year_array: np.ndarray,
    user_last_item_map: dict[int, int],
    cooccur_map: dict[int, dict[int, float]],
    include_two_tower_score: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Extracts ranking features, ground truth labels, query group sizes, and feature names.

    Args:
        candidate_df: DataFrame with [user_idx, item_idx, label, two_tower_score, retrieval_rank, timestamp].
        stats: Training statistics (item_counts, user_counts, user_genre_profiles).
        genre_matrix: Item genre multi-hot matrix.
        year_array: Normalized item release year array.
        user_last_item_map: Mapping from user_idx to their most recently interacted item.
        cooccur_map: Item-item co-occurrence map.
        include_two_tower_score: Whether to include the two-tower retrieval score (for ablation).

    Returns:
        Tuple of (feature_matrix_X, labels_y, group_sizes, active_feature_names).
    """
    # Group strictly by user_idx to form query groups
    sorted_df = candidate_df.sort_values(by="user_idx").reset_index(drop=True)

    group_sizes = sorted_df["user_idx"].value_counts(sort=False)[sorted_df["user_idx"].unique()].tolist()
    n_samples = len(sorted_df)

    item_counts = stats["item_counts"]
    user_counts = stats["user_counts"]
    user_genre_profiles = stats["user_genre_profiles"]

    feature_cols = []
    active_names = []

    # 1. Two-tower score
    if include_two_tower_score:
        feature_cols.append(sorted_df["two_tower_score"].values.astype(np.float32))
        active_names.append("two_tower_score")

    # 2. Retrieval rank (1 / rank for monotonic linear scaling)
    feature_cols.append((1.0 / sorted_df["retrieval_rank"].values).astype(np.float32))
    active_names.append("retrieval_rank")

    # 3. Item popularity (log1p)
    cand_items = sorted_df["item_idx"].values
    cand_users = sorted_df["user_idx"].values

    item_pop = np.log1p(item_counts[cand_items]).astype(np.float32)
    feature_cols.append(item_pop)
    active_names.append("item_popularity")

    # 4. Item recency / release age
    item_age = year_array[cand_items].astype(np.float32)
    feature_cols.append(item_age)
    active_names.append("item_recency_age")

    # 5. User activity level (log1p)
    user_act = np.log1p(user_counts[cand_users]).astype(np.float32)
    feature_cols.append(user_act)
    active_names.append("user_activity_level")

    # 6. User-genre affinity & 7. Genre match count
    genre_affinities = np.zeros(n_samples, dtype=np.float32)
    genre_matches = np.zeros(n_samples, dtype=np.float32)

    for idx in range(n_samples):
        u = cand_users[idx]
        i = cand_items[idx]
        u_profile = user_genre_profiles[u]
        i_genres = genre_matrix[i]

        # Dot product
        genre_affinities[idx] = np.dot(u_profile, i_genres)
        genre_matches[idx] = np.sum((u_profile > 0) & (i_genres > 0))

    feature_cols.append(genre_affinities)
    active_names.append("user_genre_affinity")

    feature_cols.append(genre_matches)
    active_names.append("genre_match_count")

    # 8. User-item co-occurrence with user's last interacted item
    cooccur_scores = np.zeros(n_samples, dtype=np.float32)
    for idx in range(n_samples):
        u = cand_users[idx]
        i = cand_items[idx]
        last_item = user_last_item_map.get(u, -1)
        if last_item != -1 and last_item in cooccur_map:
            cooccur_scores[idx] = cooccur_map[last_item].get(i, 0.0)

    feature_cols.append(cooccur_scores)
    active_names.append("user_item_cooccurrence")

    # 9 & 10. Temporal context (hour and day of week)
    timestamps = sorted_df["timestamp"].values
    dt_idx = pd.to_datetime(timestamps, unit="s", utc=True)
    hours = (np.asarray(dt_idx.hour) / 24.0).astype(np.float32)
    dows = (np.asarray(dt_idx.dayofweek) / 7.0).astype(np.float32)

    feature_cols.append(hours)
    active_names.append("hour_of_day")

    feature_cols.append(dows)
    active_names.append("day_of_week")

    X = np.column_stack(feature_cols)
    y = sorted_df["label"].values.astype(np.float32)

    return X, y, group_sizes, active_names
