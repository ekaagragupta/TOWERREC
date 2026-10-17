"""User and Item Tower neural architectures for candidate retrieval."""

import torch
import torch.nn.functional as F
from torch import nn


class UserTower(nn.Module):
    """User Tower encoding user ID and recency-weighted history interactions."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        use_history: bool = True,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_history = use_history

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)

        if use_history:
            # Item history embedding table with 0 reserved for padding
            self.item_history_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
            nn.init.normal_(self.item_history_embedding.weight, std=0.05)
            in_dim = embedding_dim * 2
        else:
            in_dim = embedding_dim

        layers: list[nn.Module] = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim

        # Final projection to embedding_dim
        layers.append(nn.Linear(curr_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        hist_items: torch.Tensor | None = None,
        hist_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encodes user inputs into L2-normalized representations.

        Args:
            user_ids: Tensor of user indices (B,).
            hist_items: Tensor of past item indices (B, seq_len), +1 shifted for padding.
            hist_weights: Tensor of normalized recency weights (B, seq_len).

        Returns:
            Normalized user embeddings (B, embedding_dim).
        """
        u_emb = self.user_embedding(user_ids)  # (B, D)

        if self.use_history and hist_items is not None and hist_weights is not None:
            # Map past item embeddings: (B, seq_len, D)
            h_emb = self.item_history_embedding(hist_items)
            # Weighted mean pooling over sequence: (B, D)
            pooled_history = (h_emb * hist_weights.unsqueeze(-1)).sum(dim=1)
            combined = torch.cat([u_emb, pooled_history], dim=-1)
        else:
            combined = u_emb

        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item Tower encoding item ID, multi-hot genre features, and release year."""

    def __init__(
        self,
        num_items: int,
        num_genres: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

        self.genre_proj = nn.Linear(num_genres, embedding_dim)
        self.year_proj = nn.Linear(1, 16)

        in_dim = embedding_dim * 2 + 16

        layers: list[nn.Module] = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        item_ids: torch.Tensor,
        genres: torch.Tensor,
        years: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes item inputs into L2-normalized representations.

        Args:
            item_ids: Tensor of item indices (B,).
            genres: Multi-hot genre tensor (B, num_genres).
            years: Normalized release year tensor (B, 1).

        Returns:
            Normalized item embeddings (B, embedding_dim).
        """
        id_emb = self.item_embedding(item_ids)  # (B, D)
        g_emb = F.relu(self.genre_proj(genres))  # (B, D)
        y_emb = F.relu(self.year_proj(years))    # (B, 16)

        combined = torch.cat([id_emb, g_emb, y_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """End-to-end Two-Tower candidate retrieval model."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        embedding_dim: int = 64,
        user_hidden_dims: list[int] | None = None,
        item_hidden_dims: list[int] | None = None,
        use_history: bool = True,
    ):
        super().__init__()
        self.user_tower = UserTower(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dims=user_hidden_dims,
            use_history=use_history,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            num_genres=num_genres,
            embedding_dim=embedding_dim,
            hidden_dims=item_hidden_dims,
        )

    def encode_users(
        self,
        user_ids: torch.Tensor,
        hist_items: torch.Tensor | None = None,
        hist_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.user_tower(user_ids, hist_items, hist_weights)

    def encode_items(
        self,
        item_ids: torch.Tensor,
        genres: torch.Tensor,
        years: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_ids, genres, years)
