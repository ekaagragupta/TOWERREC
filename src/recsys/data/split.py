"""Strict time-based dataset splitting and leakage-free index mapping."""

import pandas as pd

from recsys.utils.logger import get_logger

logger = get_logger("recsys.data.split")


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_user_interactions: int = 5,
    min_item_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, any]]:
    """Partitions interactions into Train, Validation, and Test strictly by timestamp.

    Guarantees:
        max(train.timestamp) < min(val.timestamp) <= max(val.timestamp) < min(test.timestamp)

    Args:
        df: DataFrame containing userId, movieId, rating, timestamp.
        train_ratio: Proportion of interactions in training split.
        val_ratio: Proportion of interactions in validation split.
        test_ratio: Proportion of interactions in test split.
        min_user_interactions: Minimum interactions for a user to be retained.
        min_item_interactions: Minimum interactions for an item to be retained.

    Returns:
        Tuple of (train_df, val_df, test_df, metadata_dict).
    """
    logger.info("Total raw interactions: %d", len(df))

    # 1. Iterative k-core filtering if requested
    if min_user_interactions > 1 or min_item_interactions > 1:
        prev_len = 0
        while len(df) != prev_len:
            prev_len = len(df)
            user_counts = df["userId"].value_counts()
            item_counts = df["movieId"].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            valid_items = item_counts[item_counts >= min_item_interactions].index
            df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_items)]
        logger.info("Interactions after k-core filtering: %d", len(df))

    # 2. Strict chronological sort
    df = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)

    n_total = len(df)
    train_end_idx = int(n_total * train_ratio)
    val_end_idx = int(n_total * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    # Determine exact timestamp cutoffs
    t_val_cutoff = train_df["timestamp"].max()
    t_test_cutoff = val_df["timestamp"].max()

    # Re-verify boundaries
    val_df = val_df[val_df["timestamp"] > t_val_cutoff].copy()
    test_df = test_df[test_df["timestamp"] > t_test_cutoff].copy()

    # 3. Create canonical user and item ID mapping strictly from training set (cold items handled)
    unique_users = sorted(train_df["userId"].unique())
    unique_items = sorted(train_df["movieId"].unique())

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {mid: idx for idx, mid in enumerate(unique_items)}

    # Map IDs
    train_df["user_idx"] = train_df["userId"].map(user_to_idx)
    train_df["item_idx"] = train_df["movieId"].map(item_to_idx)

    val_df["user_idx"] = val_df["userId"].map(user_to_idx)
    val_df["item_idx"] = val_df["movieId"].map(item_to_idx)

    test_df["user_idx"] = test_df["userId"].map(user_to_idx)
    test_df["item_idx"] = test_df["movieId"].map(item_to_idx)

    # Filter out users/items not in train for evaluation of known representations, or retain for cold-start
    val_eval = val_df.dropna(subset=["user_idx", "item_idx"]).copy()
    val_eval["user_idx"] = val_eval["user_idx"].astype(int)
    val_eval["item_idx"] = val_eval["item_idx"].astype(int)

    test_eval = test_df.dropna(subset=["user_idx", "item_idx"]).copy()
    test_eval["user_idx"] = test_eval["user_idx"].astype(int)
    test_eval["item_idx"] = test_eval["item_idx"].astype(int)

    metadata = {
        "num_users": len(unique_users),
        "num_items": len(unique_items),
        "train_interactions": len(train_df),
        "val_interactions": len(val_eval),
        "test_interactions": len(test_eval),
        "t_val_cutoff": int(t_val_cutoff),
        "t_test_cutoff": int(t_test_cutoff),
        "train_max_ts": int(train_df["timestamp"].max()),
        "val_min_ts": int(val_eval["timestamp"].min()),
        "val_max_ts": int(val_eval["timestamp"].max()),
        "test_min_ts": int(test_eval["timestamp"].min()),
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
    }

    logger.info(
        "Split complete: Train=%d (max_ts=%d), Val=%d (min_ts=%d), Test=%d (min_ts=%d)",
        len(train_df),
        metadata["train_max_ts"],
        len(val_eval),
        metadata["val_min_ts"],
        len(test_eval),
        metadata["test_min_ts"],
    )

    return train_df, val_eval, test_eval, metadata
