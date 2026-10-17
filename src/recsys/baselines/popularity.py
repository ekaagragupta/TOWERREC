"""Global popularity baseline recommender."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


class PopularityRecommender:
    """Ranks candidates by global historical interaction counts in the training set."""

    def __init__(self, time_decay: bool = False, decay_lambda: float = 0.0001):
        self.time_decay = time_decay
        self.decay_lambda = decay_lambda
        self.ranked_items: np.ndarray = np.array([], dtype=np.int64)
        self.item_scores: np.ndarray = np.array([], dtype=np.float32)

    def fit(self, train_df: pd.DataFrame, num_items: int) -> "PopularityRecommender":
        """Fits popularity scores from training interactions."""
        scores = np.zeros(num_items, dtype=np.float32)

        if self.time_decay and "timestamp" in train_df.columns:
            max_t = train_df["timestamp"].max()
            delta_hours = (max_t - train_df["timestamp"].values) / 3600.0
            weights = np.exp(-self.decay_lambda * delta_hours)
            for item_idx, w in zip(train_df["item_idx"].values, weights):
                scores[item_idx] += w
        else:
            counts = train_df["item_idx"].value_counts()
            scores[counts.index.values] = counts.values.astype(np.float32)

        self.item_scores = scores
        self.ranked_items = np.argsort(-scores)  # descending
        return self

    def recommend(
        self,
        user_ids: Sequence[int],
        k: int = 10,
        user_histories: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        """Recommends top-k popular items, optionally filtering items already in user history."""
        recommendations: dict[int, list[int]] = {}

        for u in user_ids:
            seen = user_histories.get(u, set()) if user_histories else set()
            recs = []
            for item in self.ranked_items:
                if item not in seen:
                    recs.append(int(item))
                if len(recs) == k:
                    break
            recommendations[u] = recs

        return recommendations
