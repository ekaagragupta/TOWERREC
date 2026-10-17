"""PyTorch BPR Matrix Factorization baseline recommender."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from recsys.utils.logger import get_logger

logger = get_logger("recsys.baselines.bpr_mf")


class BPRDataset(Dataset):
    """Generates (user, positive_item, negative_item) triplets for BPR training."""

    def __init__(self, user_items_map: dict[int, set[int]], num_items: int, n_samples: int = 100000):
        self.users = list(user_items_map.keys())
        self.user_items_map = user_items_map
        self.num_items = num_items
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        u = self.users[np.random.randint(0, len(self.users))]
        pos_items = list(self.user_items_map[u])
        pos_i = pos_items[np.random.randint(0, len(pos_items))]

        # Negative sampling (uniform rejection)
        neg_i = np.random.randint(0, self.num_items)
        while neg_i in self.user_items_map[u]:
            neg_i = np.random.randint(0, self.num_items)

        return torch.tensor(u, dtype=torch.long), torch.tensor(pos_i, dtype=torch.long), torch.tensor(neg_i, dtype=torch.long)


class BPRMFModel(nn.Module):
    """Matrix Factorization model trained with Bayesian Personalized Ranking loss."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_biases = nn.Embedding(num_items, 1)

        # Initialize embeddings with small standard normal
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(u)
        item_emb = self.item_embeddings(i)
        bias = self.item_biases(i).squeeze(-1)
        return (user_emb * item_emb).sum(dim=-1) + bias

    def predict_all_for_user(self, u: int) -> np.ndarray:
        """Computes scores for a single user across all items."""
        with torch.no_grad():
            u_t = torch.tensor([u], dtype=torch.long)
            user_emb = self.user_embeddings(u_t)  # (1, D)
            item_embs = self.item_embeddings.weight  # (num_items, D)
            biases = self.item_biases.weight.squeeze(-1)  # (num_items,)
            scores = (user_emb @ item_embs.T).squeeze(0) + biases
            return scores.cpu().numpy()


class BPRMFRecommender:
    """Wrapper for training and scoring PyTorch BPR-MF on CPU."""

    def __init__(
        self,
        embedding_dim: int = 32,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        num_epochs: int = 15,
        batch_size: int = 256,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model: BPRMFModel | None = None
        self.num_users: int = 0
        self.num_items: int = 0

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int) -> "BPRMFRecommender":
        """Fits BPR-MF using SGD on (user, positive, negative) triples."""
        torch.manual_seed(self.seed)
        self.num_users = num_users
        self.num_items = num_items
        self.model = BPRMFModel(num_users, num_items, self.embedding_dim)

        user_items_map = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        n_samples = max(len(train_df) * 2, 50000)
        dataset = BPRDataset(user_items_map, num_items, n_samples=n_samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        logger.info("Training BPR-MF (%d users, %d items, %d epochs)...", num_users, num_items, self.num_epochs)
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0.0
            for u_batch, pos_batch, neg_batch in dataloader:
                optimizer.zero_grad()
                pos_scores = self.model(u_batch, pos_batch)
                neg_scores = self.model(u_batch, neg_batch)
                # BPR loss: - ln(sigmoid(pos_scores - neg_scores)) = softplus(-(pos - neg))
                loss = torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(u_batch)

            avg_loss = total_loss / len(dataset)
            if epoch % 5 == 0 or epoch == self.num_epochs:
                logger.info("BPR-MF Epoch %d/%d - Loss: %.4f", epoch, self.num_epochs, avg_loss)

        self.model.eval()
        return self

    def recommend(
        self,
        user_ids: Sequence[int],
        k: int = 10,
        user_histories: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        """Generates recommendations for user_ids."""
        recommendations: dict[int, list[int]] = {}
        if self.model is None:
            raise RuntimeError("Model must be trained before calling recommend.")

        for u in user_ids:
            seen = user_histories.get(u, set()) if user_histories else set()
            if u < self.num_users:
                scores = self.model.predict_all_for_user(u)
            else:
                scores = np.zeros(self.num_items, dtype=np.float32)

            if seen:
                scores[list(seen)] = -1e9

            top_indices = np.argpartition(-scores, range(min(k, len(scores))))[:k]
            top_sorted = top_indices[np.argsort(-scores[top_indices])]
            recommendations[u] = [int(i) for i in top_sorted]

        return recommendations
