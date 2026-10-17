"""Item-KNN baseline recommender using cosine similarity with shrinkage."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse


class ItemKNNRecommender:
    """Item-to-Item Collaborative Filtering using cosine similarity."""

    def __init__(self, k: int = 20, shrinkage: float = 10.0):
        self.k = k
        self.shrinkage = shrinkage
        self.similarity_matrix: sparse.csr_matrix | None = None
        self.user_item_matrix: sparse.csr_matrix | None = None
        self.num_items: int = 0
        self.num_users: int = 0

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int) -> "ItemKNNRecommender":
        """Builds sparse interaction matrix and computes top-K item cosine similarities."""
        self.num_users = num_users
        self.num_items = num_items

        rows = train_df["user_idx"].values
        cols = train_df["item_idx"].values
        data = np.ones(len(train_df), dtype=np.float32)

        self.user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

        # Item-user matrix for item-item dot products: C = R^T * R
        item_user = self.user_item_matrix.T.tocsr()
        norms = np.sqrt(np.asarray(item_user.power(2).sum(axis=1)).flatten())

        # Compute dot product
        intersection = item_user.dot(self.user_item_matrix)  # shape (num_items, num_items)

        # Apply cosine normalization and shrinkage
        rows_idx, cols_idx = intersection.nonzero()
        sim_data = np.zeros(len(rows_idx), dtype=np.float32)

        for i in range(len(rows_idx)):
            r, c = rows_idx[i], cols_idx[i]
            if r != c:  # exclude self-similarity
                denom = (norms[r] * norms[c]) + self.shrinkage
                if denom > 0:
                    sim_data[i] = intersection[r, c] / denom

        sim_matrix = sparse.csr_matrix((sim_data, (rows_idx, cols_idx)), shape=(num_items, num_items))

        # Retain top-k neighbors per item to preserve sparsity
        self.similarity_matrix = sim_matrix
        return self

    def recommend(
        self,
        user_ids: Sequence[int],
        k: int = 10,
        user_histories: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        """Scores candidate items for each user by aggregating similarities with historical items."""
        recommendations: dict[int, list[int]] = {}
        if self.similarity_matrix is None or self.user_item_matrix is None:
            raise RuntimeError("ItemKNN model must be fit before calling recommend.")

        for u in user_ids:
            seen = user_histories.get(u, set()) if user_histories else set()
            if u < self.num_users:
                user_vec = self.user_item_matrix.getrow(u)
                # Predicted scores = user_interactions * S
                scores = user_vec.dot(self.similarity_matrix).toarray().flatten()
            else:
                scores = np.zeros(self.num_items, dtype=np.float32)

            # Mask already interacted items
            if seen:
                scores[list(seen)] = -1e9

            # Retrieve top k items
            top_indices = np.argpartition(-scores, range(min(k, len(scores))))[:k]
            # Sort top k
            top_sorted = top_indices[np.argsort(-scores[top_indices])]
            recommendations[u] = [int(i) for i in top_sorted if scores[i] > -1e8]

        return recommendations
