"""PyTorch Dataset classes for Two-Tower retrieval training and evaluation."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerTrainDataset(Dataset):
    """Dataset yielding user histories, item features, and item sampling frequencies."""

    def __init__(
        self,
        df: pd.DataFrame,
        hist_items: np.ndarray,
        hist_weights: np.ndarray,
        genre_matrix: np.ndarray,
        year_array: np.ndarray,
        item_frequencies: np.ndarray,
    ):
        self.user_ids = df["user_idx"].values.astype(np.int64)
        self.item_ids = df["item_idx"].values.astype(np.int64)
        # Shift history item indices by +1 for padding index 0
        self.hist_items = hist_items + 1
        self.hist_weights = hist_weights.astype(np.float32)
        self.genre_matrix = genre_matrix.astype(np.float32)
        self.year_array = year_array.astype(np.float32)
        self.item_frequencies = item_frequencies.astype(np.float32)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.user_ids[idx]
        i = self.item_ids[idx]
        h_items = self.hist_items[idx]
        h_weights = self.hist_weights[idx]
        genre = self.genre_matrix[i]
        year = np.array([self.year_array[i]], dtype=np.float32)
        freq = self.item_frequencies[i]

        return (
            torch.tensor(u, dtype=torch.long),
            torch.from_numpy(h_items).long(),
            torch.from_numpy(h_weights).float(),
            torch.tensor(i, dtype=torch.long),
            torch.from_numpy(genre).float(),
            torch.from_numpy(year).float(),
            torch.tensor(freq, dtype=torch.float32),
        )
