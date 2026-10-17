"""Feature extraction and sequence building with strict temporal leakage prevention."""

import re

import numpy as np
import pandas as pd

from recsys.utils.logger import get_logger

logger = get_logger("recsys.data.features")


def extract_movie_features(
    movies_df: pd.DataFrame,
    item_to_idx: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extracts genre multi-hot matrix and year buckets for known items.

    Args:
        movies_df: DataFrame with movieId, title, genres.
        item_to_idx: Mapping from movieId to 0-indexed contiguous item index.

    Returns:
        Tuple of (genre_matrix, year_array, genre_names).
    """
    num_items = len(item_to_idx)

    # 1. Collect all distinct genres
    all_genres = set()
    for g_str in movies_df["genres"].dropna():
        for g in g_str.split("|"):
            if g and g != "(no genres listed)":
                all_genres.add(g)
    genre_names = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(genre_names)}
    num_genres = len(genre_names)

    genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
    year_array = np.zeros(num_items, dtype=np.float32)

    # Regex for 4-digit year in title: e.g., "Toy Story (1995)"
    year_pattern = re.compile(r"\((\d{4})\)")
    years_found = []

    movie_dict = movies_df.set_index("movieId").to_dict(orient="index")

    for raw_id, idx in item_to_idx.items():
        if raw_id in movie_dict:
            row = movie_dict[raw_id]
            # Genres
            g_str = str(row.get("genres", ""))
            for g in g_str.split("|"):
                if g in genre_to_idx:
                    genre_matrix[idx, genre_to_idx[g]] = 1.0

            # Year
            title = str(row.get("title", ""))
            match = year_pattern.search(title)
            if match:
                y = int(match.group(1))
                if 1900 <= y <= 2030:
                    years_found.append(y)
                    year_array[idx] = y
                else:
                    year_array[idx] = 2000.0
            else:
                year_array[idx] = 2000.0
        else:
            year_array[idx] = 2000.0

    median_year = float(np.median(years_found)) if years_found else 2000.0
    year_array[year_array == 0] = median_year
    # Normalize year between [0, 1] for network stability
    min_year, max_year = 1900.0, 2030.0
    norm_year_array = (year_array - min_year) / (max_year - min_year)

    logger.info("Extracted features for %d items: %d genres, median year %.1f", num_items, num_genres, median_year)
    return genre_matrix, norm_year_array.astype(np.float32), genre_names


def build_user_history_sequences(
    df: pd.DataFrame,
    max_history_len: int = 20,
    recency_decay_lambda: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constructs user historical interaction sequences strictly prior to each event timestamp.

    Guarantees no future leakage: for interaction at index t, history only contains
    items interacted at timestamps t' < t.

    Args:
        df: DataFrame sorted by timestamp ascending, with columns [user_idx, item_idx, timestamp].
        max_history_len: Maximum number of previous items to retain.
        recency_decay_lambda: Exponential decay factor per hour/second for recency weighting.

    Returns:
        Tuple of (hist_items, hist_weights, hist_lens) arrays with shape (N, max_history_len).
    """
    # Group interactions by user
    df_sorted = df.sort_values(by=["user_idx", "timestamp"], ascending=True)

    n_samples = len(df_sorted)
    hist_items = np.zeros((n_samples, max_history_len), dtype=np.int64)
    hist_weights = np.zeros((n_samples, max_history_len), dtype=np.float32)
    hist_lens = np.zeros(n_samples, dtype=np.int64)

    user_histories: dict[int, list[tuple[int, float]]] = {}

    user_indices = df_sorted["user_idx"].values
    item_indices = df_sorted["item_idx"].values
    timestamps = df_sorted["timestamp"].values

    for i in range(n_samples):
        u = user_indices[i]
        curr_item = item_indices[i]
        curr_t = float(timestamps[i])

        if u not in user_histories:
            user_histories[u] = []

        history = user_histories[u]
        h_len = min(len(history), max_history_len)
        hist_lens[i] = h_len

        if h_len > 0:
            recent_slice = history[-h_len:]
            for pos, (past_item, past_t) in enumerate(recent_slice):
                delta_t = max(0.0, curr_t - past_t) / 3600.0  # elapsed hours
                weight = np.exp(-recency_decay_lambda * delta_t)
                hist_items[i, pos] = past_item
                hist_weights[i, pos] = weight

            # Normalize weights to sum to 1
            w_sum = hist_weights[i, :h_len].sum()
            if w_sum > 0:
                hist_weights[i, :h_len] /= w_sum
            else:
                hist_weights[i, :h_len] = 1.0 / h_len

        # Append current interaction to user's history for future events
        user_histories[u].append((curr_item, curr_t))

    return hist_items, hist_weights, hist_lens


def compute_training_statistics(
    train_df: pd.DataFrame,
    genre_matrix: np.ndarray,
    num_users: int,
    num_items: int,
) -> dict[str, any]:
    """Computes global and per-entity statistics strictly on the training set to prevent leakage.

    Args:
        train_df: Training interactions DataFrame.
        genre_matrix: Item genre feature matrix (num_items, num_genres).
        num_users: Total user count.
        num_items: Total item count.

    Returns:
        Dictionary of stats: item_popularity, user_activity, user_genre_profiles, item_frequencies.
    """
    # 1. Item popularity (counts in train set)
    item_counts = np.zeros(num_items, dtype=np.float32)
    counts_series = train_df["item_idx"].value_counts()
    item_counts[counts_series.index.values] = counts_series.values

    # In-batch logQ sampling probabilities: p_j = freq(j) / total_interactions
    total_interactions = len(train_df)
    item_frequencies = np.maximum(item_counts, 1.0) / total_interactions

    # 2. User activity level (counts in train set)
    user_counts = np.zeros(num_users, dtype=np.float32)
    u_counts_series = train_df["user_idx"].value_counts()
    user_counts[u_counts_series.index.values] = u_counts_series.values

    # 3. User-genre affinity profiles (weighted average of genres for items user interacted with)
    num_genres = genre_matrix.shape[1]
    user_genre_profiles = np.zeros((num_users, num_genres), dtype=np.float32)

    user_grouped = train_df.groupby("user_idx")["item_idx"].apply(list)
    for u_idx, items in user_grouped.items():
        if items:
            profile = genre_matrix[items].mean(axis=0)
            user_genre_profiles[u_idx] = profile

    return {
        "item_counts": item_counts,
        "item_frequencies": item_frequencies,
        "user_counts": user_counts,
        "user_genre_profiles": user_genre_profiles,
        "total_train_interactions": total_interactions,
    }
