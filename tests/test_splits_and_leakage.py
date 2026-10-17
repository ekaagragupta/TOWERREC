"""Unit and leakage tests asserting strict temporal ordering and zero feature leakage."""

import pandas as pd
import pytest

from recsys.data.download import generate_synthetic_movielens
from recsys.data.features import (
    build_user_history_sequences,
    compute_training_statistics,
    extract_movie_features,
)
from recsys.data.split import time_based_split


@pytest.fixture
def synthetic_data(tmp_path):
    ratings_path, movies_path = generate_synthetic_movielens(
        target_dir=tmp_path / "ml_test",
        n_users=50,
        n_items=100,
        n_ratings=2000,
    )
    df_ratings = pd.read_csv(ratings_path)
    df_movies = pd.read_csv(movies_path)
    return df_ratings, df_movies


def test_time_based_split_ordering(synthetic_data):
    df_ratings, _ = synthetic_data
    train_df, val_df, test_df, metadata = time_based_split(
        df_ratings,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        min_user_interactions=2,
        min_item_interactions=2,
    )

    # 1. Assert no test timestamp precedes any train or val timestamp
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), (
        f"Train max timestamp {train_df['timestamp'].max()} >= Val min timestamp {val_df['timestamp'].min()}"
    )
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), (
        f"Val max timestamp {val_df['timestamp'].max()} >= Test min timestamp {test_df['timestamp'].min()}"
    )

    # 2. Strict global inequality: max(train) < min(test)
    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_no_history_leakage(synthetic_data):
    df_ratings, _ = synthetic_data
    train_df, _, _, _ = time_based_split(
        df_ratings,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # Build history sequences
    hist_items, hist_weights, hist_lens = build_user_history_sequences(
        train_df,
        max_history_len=10,
        recency_decay_lambda=0.001,
    )

    df_sorted = train_df.sort_values(by=["user_idx", "timestamp"], ascending=True).reset_index(drop=True)
    user_indices = df_sorted["user_idx"].values
    item_indices = df_sorted["item_idx"].values
    timestamps = df_sorted["timestamp"].values

    for i in range(len(df_sorted)):
        curr_u = user_indices[i]
        curr_item = item_indices[i]
        curr_ts = timestamps[i]
        h_len = hist_lens[i]

        if h_len > 0:
            past_items = hist_items[i, :h_len]
            # Verify past items for this user indeed occurred strictly before curr_ts
            user_past_events = df_sorted[
                (df_sorted["user_idx"] == curr_u) & (df_sorted["timestamp"] < curr_ts)
            ]["item_idx"].values

            for p_item in past_items:
                assert p_item in user_past_events, f"Item {p_item} in history was not interacted before {curr_ts}"

            # Verify current target item is NOT in history if it was never interacted before
            if curr_item not in user_past_events:
                assert curr_item not in past_items


def test_no_train_feature_contamination(synthetic_data):
    df_ratings, df_movies = synthetic_data
    train_df, val_df, test_df, metadata = time_based_split(
        df_ratings,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    genre_matrix, _, _ = extract_movie_features(df_movies, metadata["item_to_idx"])
    stats = compute_training_statistics(
        train_df,
        genre_matrix,
        metadata["num_users"],
        metadata["num_items"],
    )

    # Assert item counts only reflect train interactions
    assert stats["item_counts"].sum() == len(train_df)
    assert stats["user_counts"].sum() == len(train_df)

    # Pick an item that appears in test but has 0 counts in train (if any)
    train_items = set(train_df["item_idx"].unique())
    all_known_items = set(range(metadata["num_items"]))
    unseen_in_train = all_known_items - train_items
    for item in unseen_in_train:
        assert stats["item_counts"][item] == 0.0
