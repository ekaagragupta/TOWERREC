"""PyTorch Two-Tower training loop with validation Recall@K early stopping and checkpointing."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.retrieval.faiss_index import FaissFlatIPIndex
from recsys.retrieval.losses import SampledSoftmaxLoss
from recsys.retrieval.towers import TwoTowerModel
from recsys.utils.logger import get_logger

logger = get_logger("recsys.retrieval.trainer")


class TwoTowerTrainer:
    """Trains Two-Tower retrieval model with in-batch sampled softmax and validation early stopping."""

    def __init__(
        self,
        model: TwoTowerModel,
        loss_fn: SampledSoftmaxLoss,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 3,
        checkpoint_dir: Path | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_recall: float = -1.0
        self.patience_counter: int = 0
        self.best_checkpoint_path: Path | None = None

    def save_checkpoint(self, epoch: int, val_recall: float, filename: str = "best_two_tower.pt") -> Path | None:
        """Saves model and optimizer state."""
        if not self.checkpoint_dir:
            return None
        ckpt_path = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "val_recall": val_recall,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Resumes training from checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.best_val_recall = state.get("val_recall", -1.0)
        logger.info("Resumed checkpoint from %s (best val recall: %.4f)", checkpoint_path, self.best_val_recall)

    def precompute_item_embeddings(
        self,
        num_items: int,
        genre_matrix: np.ndarray,
        year_array: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Encodes all items in catalog into normalized embedding matrix."""
        self.model.eval()
        item_embs = []

        with torch.no_grad():
            for start_idx in range(0, num_items, batch_size):
                end_idx = min(start_idx + batch_size, num_items)
                i_ids = torch.arange(start_idx, end_idx, dtype=torch.long, device=self.device)
                g = torch.from_numpy(genre_matrix[start_idx:end_idx]).float().to(self.device)
                y = torch.from_numpy(year_array[start_idx:end_idx, None]).float().to(self.device)

                emb = self.model.encode_items(i_ids, g, y)
                item_embs.append(emb.cpu().numpy())

        return np.vstack(item_embs)

    def evaluate_validation_recall(
        self,
        val_df: pd.DataFrame,
        item_embeddings: np.ndarray,
        user_histories: dict[int, tuple[np.ndarray, np.ndarray]],
        k: int = 20,
    ) -> float:
        """Computes Recall@K on validation split."""
        self.model.eval()
        val_ground_truth = val_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        val_users = list(val_ground_truth.keys())

        if not val_users:
            return 0.0

        index = FaissFlatIPIndex(dim=item_embeddings.shape[1])
        index.add(item_embeddings)

        hits = 0
        total_targets = 0

        with torch.no_grad():
            for u in val_users:
                u_t = torch.tensor([u], dtype=torch.long, device=self.device)
                if u in user_histories:
                    h_items, h_weights = user_histories[u]
                    h_i_t = torch.from_numpy(h_items[None, :] + 1).long().to(self.device)
                    h_w_t = torch.from_numpy(h_weights[None, :]).float().to(self.device)
                else:
                    h_i_t, h_w_t = None, None

                u_emb = self.model.encode_users(u_t, h_i_t, h_w_t).cpu().numpy()
                _, top_items = index.search(u_emb, k=k)

                gt_set = val_ground_truth[u]
                hits += len(gt_set.intersection(top_items[0]))
                total_targets += len(gt_set)

        return float(hits / total_targets) if total_targets > 0 else 0.0

    def fit(
        self,
        train_dataloader: DataLoader,
        val_df: pd.DataFrame,
        num_items: int,
        genre_matrix: np.ndarray,
        year_array: np.ndarray,
        val_user_histories: dict[int, tuple[np.ndarray, np.ndarray]],
        num_epochs: int = 15,
        eval_k: int = 20,
    ) -> tuple[np.ndarray, dict[str, list[float]]]:
        """Runs the training loop with validation early stopping.

        Returns:
            Tuple of (precomputed_item_embeddings, training_metrics_history).
        """
        logger.info("Starting Two-Tower retrieval training (%d epochs, patience %d)...", num_epochs, self.early_stopping_patience)
        history: dict[str, list[float]] = {"train_loss": [], "val_recall": []}

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            for u_ids, h_items, h_weights, i_ids, genres, years, sampling_probs in train_dataloader:
                u_ids = u_ids.to(self.device)
                h_items = h_items.to(self.device)
                h_weights = h_weights.to(self.device)
                i_ids = i_ids.to(self.device)
                genres = genres.to(self.device)
                years = years.to(self.device)
                sampling_probs = sampling_probs.to(self.device)

                self.optimizer.zero_grad()
                user_embs = self.model.encode_users(u_ids, h_items, h_weights)
                item_embs = self.model.encode_items(i_ids, genres, years)

                loss = self.loss_fn(user_embs, item_embs, sampling_probs)
                loss.backward()
                self.optimizer.step()

                batch_n = u_ids.size(0)
                total_loss += loss.item() * batch_n
                total_samples += batch_n

            epoch_train_loss = total_loss / max(total_samples, 1)
            history["train_loss"].append(epoch_train_loss)

            # Validation evaluation
            item_embeddings = self.precompute_item_embeddings(num_items, genre_matrix, year_array)
            val_recall = self.evaluate_validation_recall(val_df, item_embeddings, val_user_histories, k=eval_k)
            history["val_recall"].append(val_recall)

            logger.info("Epoch %02d/%02d - Train Loss: %.4f | Val Recall@%d: %.4f", epoch, num_epochs, epoch_train_loss, eval_k, val_recall)

            # Checkpoint & Early stopping
            if val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                self.patience_counter = 0
                self.best_checkpoint_path = self.save_checkpoint(epoch, val_recall)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d (best Recall@%d: %.4f)", epoch, eval_k, self.best_val_recall)
                    break

        if self.best_checkpoint_path is not None and self.best_checkpoint_path.is_file():
            self.load_checkpoint(self.best_checkpoint_path)

        final_item_embeddings = self.precompute_item_embeddings(num_items, genre_matrix, year_array)
        return final_item_embeddings, history
