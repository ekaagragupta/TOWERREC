"""DCN-v2 (Deep & Cross Network v2) neural ranker in PyTorch (stretch model)."""


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from recsys.utils.logger import get_logger

logger = get_logger("recsys.ranking.dcnv2")


class CrossNetwork(nn.Module):
    """Cross Network with explicit feature interactions at each layer: x_{l+1} = x_0 * (W_l x_l + b) + x_l."""

    def __init__(self, in_features: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_features, in_features)) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.empty(in_features)) for _ in range(num_layers)
        ])

        for w, b in zip(self.weights, self.biases):
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for w, b in zip(self.weights, self.biases):
            # xl_next = x0 * (xl @ W + b) + xl
            xl = x0 * (torch.matmul(xl, w) + b) + xl
        return xl


class DCNv2Model(nn.Module):
    """DCN-v2 architecture combining Cross Network and Deep MLP."""

    def __init__(self, in_features: int, cross_layers: int = 2, deep_layers: list[int] | None = None):
        super().__init__()
        if deep_layers is None:
            deep_layers = [128, 64]

        self.cross_net = CrossNetwork(in_features, num_layers=cross_layers)

        deep_modules: list[nn.Module] = []
        curr_dim = in_features
        for h_dim in deep_layers:
            deep_modules.append(nn.Linear(curr_dim, h_dim))
            deep_modules.append(nn.LayerNorm(h_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(0.1))
            curr_dim = h_dim

        self.deep_net = nn.Sequential(*deep_modules)

        # Stacking / Combination head
        combined_dim = in_features + deep_layers[-1]
        self.head = nn.Linear(combined_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cross = self.cross_net(x)
        x_deep = self.deep_net(x)
        combined = torch.cat([x_cross, x_deep], dim=-1)
        return self.head(combined).squeeze(-1)


class DCNv2Ranker:
    """Wrapper to train DCN-v2 and score candidate items."""

    def __init__(
        self,
        cross_layers: int = 2,
        deep_layers: list[int] | None = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 10,
        device: str = "cpu",
    ):
        self.cross_layers = cross_layers
        self.deep_layers = deep_layers or [128, 64]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.model: DCNv2Model | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DCNv2Ranker":
        in_dim = X_train.shape[1]
        self.model = DCNv2Model(in_dim, self.cross_layers, self.deep_layers).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        logger.info("Training DCN-v2 (%d features, %d epochs)...", in_dim, self.num_epochs)
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)

            if epoch % 5 == 0 or epoch == self.num_epochs:
                logger.info("DCN-v2 Epoch %d/%d - Loss: %.4f", epoch, self.num_epochs, total_loss / len(dataset))

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).float().to(self.device)
            logits = self.model(x_t)
            scores = torch.sigmoid(logits).cpu().numpy()
        return scores

    def rerank(
        self,
        candidate_df: pd.DataFrame,
        features_X: np.ndarray,
        top_n: int = 10,
    ) -> dict[int, list[int]]:
        scores = self.predict(features_X)
        df = candidate_df.copy()
        df["rank_score"] = scores

        recommendations: dict[int, list[int]] = {}
        for u, group in df.groupby("user_idx"):
            sorted_items = group.sort_values(by="rank_score", ascending=False)["item_idx"].head(top_n).tolist()
            recommendations[u] = [int(i) for i in sorted_items]

        return recommendations
