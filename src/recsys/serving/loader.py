"""Artifact loader and serving orchestrator for Two-Tower + LightGBM pipeline."""

import time

import numpy as np
import pandas as pd
import torch

from recsys.ranking.features import extract_ranking_features
from recsys.ranking.lgbm_ranker import LGBMRankModel
from recsys.retrieval.faiss_index import BaseItemIndex
from recsys.retrieval.towers import TwoTowerModel
from recsys.serving.schemas import RecommendationItem, RecommendResponse, StageLatency
from recsys.utils.logger import get_logger

logger = get_logger("recsys.serving.loader")


class ServingPipeline:
    """Production serving pipeline integrating Two-Tower retrieval and LightGBM ranking."""

    def __init__(
        self,
        two_tower_model: TwoTowerModel | None = None,
        item_index: BaseItemIndex | None = None,
        ranker_model: LGBMRankModel | None = None,
        item_embeddings: np.ndarray | None = None,
        movie_dict: dict[int, dict[str, str]] | None = None,
        user_to_idx: dict[int, int] | None = None,
        item_to_idx: dict[int, int] | None = None,
        idx_to_item: dict[int, int] | None = None,
        popular_items: list[int] | None = None,
        stats: dict[str, any] | None = None,
        genre_matrix: np.ndarray | None = None,
        year_array: np.ndarray | None = None,
        user_histories: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        user_last_item_map: dict[int, int] | None = None,
        cooccur_map: dict[int, dict[int, float]] | None = None,
    ):
        self.two_tower_model = two_tower_model
        self.item_index = item_index
        self.ranker_model = ranker_model
        self.item_embeddings = item_embeddings
        self.movie_dict = movie_dict or {}
        self.user_to_idx = user_to_idx or {}
        self.item_to_idx = item_to_idx or {}
        self.idx_to_item = idx_to_item or {}
        self.popular_items = popular_items or []
        self.stats = stats or {}
        self.genre_matrix = genre_matrix
        self.year_array = year_array
        self.user_histories = user_histories or {}
        self.user_last_item_map = user_last_item_map or {}
        self.cooccur_map = cooccur_map or {}

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        genre_filter: str | None = None,
        top_k_candidates: int = 50,
    ) -> RecommendResponse:
        """Executes full two-stage recommendation pipeline with stage latency tracking."""
        t_start = time.perf_counter()

        # Check for cold start user
        if user_id not in self.user_to_idx or self.two_tower_model is None or self.item_index is None:
            t_cold = (time.perf_counter() - t_start) * 1000.0
            recs = []
            for rank_pos, raw_mid in enumerate(self.popular_items, start=1):
                meta = self.movie_dict.get(raw_mid, {})
                genres_str = meta.get("genres", "")
                if genre_filter and genre_filter.lower() not in genres_str.lower():
                    continue
                recs.append(
                    RecommendationItem(
                        item_id=raw_mid,
                        score=float(1.0 / rank_pos),
                        title=meta.get("title", f"Movie {raw_mid}"),
                        genres=genres_str,
                        rank=len(recs) + 1,
                    )
                )
                if len(recs) == k:
                    break

            return RecommendResponse(
                user_id=user_id,
                recommendations=recs,
                stage_latencies=StageLatency(
                    feature_fetch_ms=t_cold,
                    retrieval_ms=0.0,
                    ranking_ms=0.0,
                    total_ms=t_cold,
                ),
                is_cold_start=True,
            )

        u_idx = self.user_to_idx[user_id]

        # Stage 1: Feature fetch & user embedding
        t_feat0 = time.perf_counter()
        if u_idx in self.user_histories:
            h_items, h_weights = self.user_histories[u_idx]
            h_i_t = torch.from_numpy(h_items[None, :] + 1).long()
            h_w_t = torch.from_numpy(h_weights[None, :]).float()
        else:
            h_i_t, h_w_t = None, None

        u_t = torch.tensor([u_idx], dtype=torch.long)
        with torch.no_grad():
            u_emb = self.two_tower_model.encode_users(u_t, h_i_t, h_w_t).cpu().numpy()
        t_feat_fetch = (time.perf_counter() - t_feat0) * 1000.0

        # Stage 2: ANN Retrieval
        t_ann0 = time.perf_counter()
        scores, cand_indices = self.item_index.search(u_emb, k=top_k_candidates)
        cand_item_indices = cand_indices[0]
        cand_scores = scores[0]
        t_retrieval = (time.perf_counter() - t_ann0) * 1000.0

        # Stage 3: Ranking Stage
        t_rank0 = time.perf_counter()
        if self.ranker_model is not None and self.genre_matrix is not None and self.year_array is not None:
            cand_rows = []
            for rank_pos, (c_idx, c_score) in enumerate(zip(cand_item_indices, cand_scores), start=1):
                cand_rows.append({
                    "user_idx": u_idx,
                    "item_idx": int(c_idx),
                    "label": 0.0,
                    "two_tower_score": float(c_score),
                    "retrieval_rank": float(rank_pos),
                    "timestamp": float(time.time()),
                })
            cand_df = pd.DataFrame(cand_rows)

            X, _, _, _ = extract_ranking_features(
                candidate_df=cand_df,
                stats=self.stats,
                genre_matrix=self.genre_matrix,
                year_array=self.year_array,
                user_last_item_map=self.user_last_item_map,
                cooccur_map=self.cooccur_map,
                include_two_tower_score=True,
            )
            ranking_scores = self.ranker_model.predict(X)
            cand_df["final_score"] = ranking_scores
            sorted_candidates = cand_df.sort_values(by="final_score", ascending=False)
        else:
            # Direct two-tower fallback
            sorted_candidates = pd.DataFrame({
                "item_idx": cand_item_indices,
                "final_score": cand_scores,
            })

        t_ranking = (time.perf_counter() - t_rank0) * 1000.0

        # Format top-k responses
        recs = []
        for _, row in sorted_candidates.iterrows():
            c_idx = int(row["item_idx"])
            raw_mid = self.idx_to_item.get(c_idx, c_idx)
            meta = self.movie_dict.get(raw_mid, {})
            genres_str = meta.get("genres", "")

            if genre_filter and genre_filter.lower() not in genres_str.lower():
                continue

            recs.append(
                RecommendationItem(
                    item_id=raw_mid,
                    score=float(row["final_score"]),
                    title=meta.get("title", f"Movie {raw_mid}"),
                    genres=genres_str,
                    rank=len(recs) + 1,
                )
            )
            if len(recs) == k:
                break

        t_total = (time.perf_counter() - t_start) * 1000.0

        return RecommendResponse(
            user_id=user_id,
            recommendations=recs,
            stage_latencies=StageLatency(
                feature_fetch_ms=t_feat_fetch,
                retrieval_ms=t_retrieval,
                ranking_ms=t_ranking,
                total_ms=t_total,
            ),
            is_cold_start=False,
        )
