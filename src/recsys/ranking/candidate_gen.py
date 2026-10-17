"""Candidate generation joining Stage 1 retrieval with interaction labels without leakage."""

import numpy as np
import pandas as pd
import torch

from recsys.retrieval.faiss_index import BaseItemIndex
from recsys.retrieval.towers import TwoTowerModel
from recsys.utils.logger import get_logger

logger = get_logger("recsys.ranking.candidate_gen")


def generate_candidates_for_split(
    model: TwoTowerModel,
    item_index: BaseItemIndex,
    eval_df: pd.DataFrame,
    user_histories: dict[int, tuple[np.ndarray, np.ndarray]],
    top_k: int = 200,
    device: str = "cpu",
    reward_threshold: float = 4.0,
) -> pd.DataFrame:
    """Generates Stage 1 candidate pool for evaluation users and pairs with ground truth labels.

    Args:
        model: Trained TwoTowerModel.
        item_index: Pre-built FAISS item index.
        eval_df: Interactions DataFrame for the target split (train or test).
        user_histories: Pre-split historical sequences per user.
        top_k: Number of candidates retrieved per user.
        device: Device string ('cpu').
        reward_threshold: Minimum rating to consider a positive interaction.

    Returns:
        DataFrame with columns: [user_idx, item_idx, label, two_tower_score, retrieval_rank, timestamp].
    """
    model.eval()
    user_positives: dict[int, set[int]] = {}
    user_latest_ts: dict[int, float] = {}

    for row in eval_df.itertuples():
        u = row.user_idx
        i = row.item_idx
        r = getattr(row, "rating", 5.0)
        ts = getattr(row, "timestamp", 0.0)

        if u not in user_positives:
            user_positives[u] = set()
            user_latest_ts[u] = ts
        else:
            user_latest_ts[u] = max(user_latest_ts[u], ts)

        if r >= reward_threshold:
            user_positives[u].add(i)

    users = list(user_positives.keys())
    candidate_rows = []

    logger.info("Generating top-%d candidates for %d queries...", top_k, len(users))

    with torch.no_grad():
        for u in users:
            u_t = torch.tensor([u], dtype=torch.long, device=device)
            if u in user_histories:
                h_items, h_weights = user_histories[u]
                h_i_t = torch.from_numpy(h_items[None, :] + 1).long().to(device)
                h_w_t = torch.from_numpy(h_weights[None, :]).float().to(device)
            else:
                h_i_t, h_w_t = None, None

            u_emb = model.encode_users(u_t, h_i_t, h_w_t).cpu().numpy()
            scores, item_ids = item_index.search(u_emb, k=top_k)

            retrieved_items = item_ids[0]
            retrieved_scores = scores[0]
            pos_set = user_positives[u]
            ts = user_latest_ts[u]

            for rank_idx, (cand_item, score) in enumerate(zip(retrieved_items, retrieved_scores), start=1):
                label = 1.0 if cand_item in pos_set else 0.0
                candidate_rows.append({
                    "user_idx": u,
                    "item_idx": int(cand_item),
                    "label": label,
                    "two_tower_score": float(score),
                    "retrieval_rank": float(rank_idx),
                    "timestamp": ts,
                })

    cand_df = pd.DataFrame(candidate_rows)
    logger.info("Generated %d candidate pairs across %d queries (positive rate: %.2f%%)", len(cand_df), len(users), 100.0 * cand_df["label"].mean() if len(cand_df) > 0 else 0.0)
    return cand_df
