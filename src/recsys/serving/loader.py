"""Artifact loader and serving orchestrator for Two-Tower + LightGBM pipeline."""

import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recsys.data.features import compute_training_statistics, extract_movie_features
from recsys.data.split import time_based_split
from recsys.ranking.candidate_gen import generate_candidates_for_split
from recsys.ranking.features import build_item_cooccurrence_matrix, extract_ranking_features
from recsys.ranking.lgbm_ranker import LGBMRankModel
from recsys.retrieval.faiss_index import BaseItemIndex, FaissFlatIPIndex
from recsys.retrieval.towers import TwoTowerModel
from recsys.serving.schemas import (
    MovieItem,
    MovieSearchResponse,
    RecommendationItem,
    RecommendResponse,
    SimilarMovieItem,
    StageLatency,
    UserHistoryItem,
    UserProfileResponse,
)
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
        movie_dict: dict[int, dict[str, Any]] | None = None,
        user_to_idx: dict[int, int] | None = None,
        item_to_idx: dict[int, int] | None = None,
        idx_to_item: dict[int, int] | None = None,
        popular_items: list[int] | None = None,
        stats: dict[str, Any] | None = None,
        genre_matrix: np.ndarray | None = None,
        year_array: np.ndarray | None = None,
        user_histories: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        user_last_item_map: dict[int, int] | None = None,
        cooccur_map: dict[int, dict[int, float]] | None = None,
        user_ratings_history: dict[int, list[dict[str, Any]]] | None = None,
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
        self.user_ratings_history = user_ratings_history or {}

        # Compute item statistics for movie search
        self.movie_stats: dict[int, dict[str, float]] = {}
        if self.stats and "item_counts" in self.stats:
            item_counts = self.stats["item_counts"]
            for idx, count in enumerate(item_counts):
                mid = self.idx_to_item.get(idx, idx)
                self.movie_stats[mid] = {"rating_count": int(count), "avg_rating": 3.5}

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        genre_filter: str | None = None,
        top_k_candidates: int = 50,
        model_type: str = "two_tower_lgbm",
    ) -> RecommendResponse:
        """Executes recommendation pipeline with model selection and stage latency tracking."""
        t_start = time.perf_counter()

        # Check for cold start user or uninitialized models
        is_cold = user_id not in self.user_to_idx or self.two_tower_model is None or self.item_index is None

        if is_cold or model_type == "popularity":
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
                        year=int(meta.get("year", 1995)) if "year" in meta else None,
                        rank=len(recs) + 1,
                        retrieval_score=float(1.0 / rank_pos),
                        explanation="Global Popularity Baseline: Highly frequent interaction in catalog.",
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
                is_cold_start=is_cold,
                model_used="popularity" if model_type == "popularity" else "popularity_cold_start",
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

        if model_type == "two_tower_retrieval":
            # Stage 1 Only: Sort purely by retrieval vector dot-product score
            sorted_candidates = pd.DataFrame({
                "item_idx": cand_item_indices,
                "final_score": cand_scores,
                "retrieval_score": cand_scores,
                "retrieval_rank": np.arange(1, len(cand_scores) + 1),
            })
            model_used_label = "two_tower_retrieval"
        elif model_type == "item_knn":
            # Item-KNN Collaborative Filtering based on co-occurrence
            last_item = self.user_last_item_map.get(u_idx, -1)
            knn_scores = []
            for c_idx in cand_item_indices:
                score = self.cooccur_map.get(last_item, {}).get(int(c_idx), 0.0)
                knn_scores.append(score)
            sorted_candidates = pd.DataFrame({
                "item_idx": cand_item_indices,
                "final_score": knn_scores,
                "retrieval_score": cand_scores,
                "retrieval_rank": np.arange(1, len(cand_scores) + 1),
            }).sort_values(by="final_score", ascending=False)
            model_used_label = "item_knn"
        else:
            # Default: Two-Tower + LightGBM LambdaRank Stage 2 Re-Ranker
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
                sorted_candidates = pd.DataFrame({
                    "item_idx": cand_item_indices,
                    "final_score": cand_scores,
                    "retrieval_score": cand_scores,
                    "retrieval_rank": np.arange(1, len(cand_scores) + 1),
                })
            model_used_label = "two_tower_lgbm"

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

            r_score = float(row.get("two_tower_score", row.get("retrieval_score", 0.0)))
            f_score = float(row["final_score"])

            if model_type == "two_tower_retrieval":
                explanation = f"Stage 1 Latent Retrieval: High vector similarity ({r_score:.3f}) in 64-dim embedding space."
            elif model_type == "item_knn":
                explanation = f"Item-KNN Cosine: Co-occurrence match with User {user_id}'s recent viewing history."
            else:
                orig_rank = int(row.get("retrieval_rank", len(recs) + 1))
                new_rank = len(recs) + 1
                jump = orig_rank - new_rank
                jump_text = f"(re-ranked +{jump} spots)" if jump > 0 else "(retained top rank)"
                explanation = f"Two-Tower + LightGBM {jump_text}: Ret score {r_score:.3f} + User genre affinity."

            recs.append(
                RecommendationItem(
                    item_id=raw_mid,
                    score=f_score,
                    title=meta.get("title", f"Movie {raw_mid}"),
                    genres=genres_str,
                    year=int(meta.get("year", 1995)) if "year" in meta else None,
                    rank=len(recs) + 1,
                    retrieval_score=r_score,
                    explanation=explanation,
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
            model_used=model_used_label,
        )

    def get_similar_movies(self, movie_id: int, k: int = 6) -> list[SimilarMovieItem]:
        """Finds nearest neighbor movies in Two-Tower item embedding space."""
        if self.item_embeddings is None or movie_id not in self.item_to_idx:
            return []

        item_idx = self.item_to_idx[movie_id]
        target_emb = self.item_embeddings[item_idx : item_idx + 1]

        # Cosine dot product with all item embeddings
        sims = np.dot(self.item_embeddings, target_emb.T).squeeze()
        top_indices = np.argsort(-sims)[: k + 1]

        results = []
        for idx in top_indices:
            if idx == item_idx:
                continue
            mid = self.idx_to_item.get(idx, idx)
            meta = self.movie_dict.get(mid, {})
            results.append(
                SimilarMovieItem(
                    item_id=mid,
                    title=meta.get("title", f"Movie {mid}"),
                    genres=meta.get("genres", ""),
                    similarity=float(sims[idx]),
                )
            )
            if len(results) == k:
                break

        return results

    def search_movies(
        self,
        query: str = "",
        genre: str = "",
        sort_by: str = "popularity",
        page: int = 1,
        limit: int = 24,
    ) -> MovieSearchResponse:
        """Searches movies by title and genre with sorting and pagination."""
        q = query.strip().lower()
        g = genre.strip().lower()

        filtered = []
        for mid, meta in self.movie_dict.items():
            title = str(meta.get("title", ""))
            genres = str(meta.get("genres", ""))

            if q and q not in title.lower():
                continue
            if g and g not in genres.lower():
                continue

            # Extract year from title if available
            year = None
            if "(" in title and title.endswith(")"):
                try:
                    year = int(title[title.rfind("(") + 1 : -1])
                except ValueError:
                    year = None

            stats_info = self.movie_stats.get(mid, {"rating_count": 0, "avg_rating": 3.5})

            filtered.append(
                MovieItem(
                    item_id=mid,
                    title=title,
                    genres=genres,
                    year=year,
                    avg_rating=stats_info.get("avg_rating", 3.5),
                    rating_count=stats_info.get("rating_count", 0),
                )
            )

        # Sorting
        if sort_by == "popularity":
            filtered.sort(key=lambda x: x.rating_count, reverse=True)
        elif sort_by == "title":
            filtered.sort(key=lambda x: x.title)
        elif sort_by == "year":
            filtered.sort(key=lambda x: x.year or 0, reverse=True)
        elif sort_by == "rating":
            filtered.sort(key=lambda x: x.avg_rating, reverse=True)

        total = len(filtered)
        total_pages = max(1, math.ceil(total / limit))
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_movies = filtered[start_idx:end_idx]

        return MovieSearchResponse(
            movies=paginated_movies,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
        )

    def get_all_genres(self) -> list[dict[str, Any]]:
        """Returns list of distinct genres and movie counts."""
        genre_counts: dict[str, int] = {}
        for meta in self.movie_dict.values():
            genres = str(meta.get("genres", "")).split("|")
            for genre in genres:
                genre = genre.strip()
                if genre:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"name": name, "count": count} for name, count in sorted_genres]

    def get_sample_users(self) -> list[dict[str, Any]]:
        """Returns representative user personas across activity tiers."""
        sample_ids = [1, 5, 10, 42, 100, 250, 400, 580]
        results = []

        for uid in sample_ids:
            if uid not in self.user_to_idx:
                continue
            history = self.user_ratings_history.get(uid, [])
            n_ratings = len(history)

            # Activity Tier
            if n_ratings < 30:
                tier = "Cold (Casual User)"
            elif n_ratings < 150:
                tier = "Warm (Regular Moviegoer)"
            else:
                tier = "Heavy (Film Enthusiast)"

            # Top Genres
            genre_freq: dict[str, int] = {}
            for item in history:
                mid = item["movieId"]
                meta = self.movie_dict.get(mid, {})
                for g in str(meta.get("genres", "")).split("|"):
                    if g:
                        genre_freq[g] = genre_freq.get(g, 0) + 1
            top_g = [g for g, _ in sorted(genre_freq.items(), key=lambda x: x[1], reverse=True)[:3]]

            results.append({
                "user_id": uid,
                "activity_tier": tier,
                "total_ratings": n_ratings,
                "top_genres": top_g,
            })

        # Add explicit Cold Start persona
        results.append({
            "user_id": 9999,
            "activity_tier": "Cold-Start (New User)",
            "total_ratings": 0,
            "top_genres": ["Unseen Profile", "Global Popularity Fallback"],
        })

        return results

    def get_user_profile(self, user_id: int) -> UserProfileResponse:
        """Returns detailed interaction history and genre preferences for a user."""
        history_records = self.user_ratings_history.get(user_id, [])

        if not history_records:
            return UserProfileResponse(
                user_id=user_id,
                activity_tier="Cold-Start (Zero prior interactions)",
                total_ratings=0,
                avg_rating=0.0,
                favorite_genres=[],
                history=[],
            )

        n_ratings = len(history_records)
        avg_rating = float(np.mean([r["rating"] for r in history_records]))

        if n_ratings < 30:
            tier = "Cold User (<30 interactions)"
        elif n_ratings < 150:
            tier = "Warm User (30-150 interactions)"
        else:
            tier = "Heavy User (150+ interactions)"

        genre_counts: dict[str, int] = {}
        formatted_history = []
        for r in history_records[:30]:  # return top 30 recent
            mid = r["movieId"]
            meta = self.movie_dict.get(mid, {})
            title = meta.get("title", f"Movie {mid}")
            genres = meta.get("genres", "")

            for g in genres.split("|"):
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

            formatted_history.append(
                UserHistoryItem(
                    movie_id=mid,
                    title=title,
                    genres=genres,
                    rating=float(r["rating"]),
                    timestamp=float(r["timestamp"]),
                )
            )

        top_genres = [{"genre": g, "count": c} for g, c in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]]

        return UserProfileResponse(
            user_id=user_id,
            activity_tier=tier,
            total_ratings=n_ratings,
            avg_rating=round(avg_rating, 2),
            favorite_genres=top_genres,
            history=formatted_history,
        )


def init_production_pipeline(
    data_dir: Path = Path("data/raw/ml-latest-small"),
    reports_dir: Path = Path("reports"),
) -> ServingPipeline:
    """Initializes and returns the complete production serving pipeline."""
    cache_path = Path("data/processed/serving_meta.pkl")
    ranker_path = reports_dir / "checkpoints" / "lgbm_ranker.pkl"
    ckpt_path = reports_dir / "checkpoints" / "best_two_tower.pt"

    movies_path = data_dir / "movies.csv"
    ratings_path = data_dir / "ratings.csv"

    if not movies_path.exists() or not ratings_path.exists():
        logger.warning("MovieLens raw dataset missing. Initializing fallback mock pipeline.")
        return ServingPipeline(popular_items=list(range(1, 11)))

    df_movies = pd.read_csv(movies_path)
    movie_dict = df_movies.set_index("movieId").to_dict(orient="index")

    # Fast load from precomputed cache
    if cache_path.exists() and ranker_path.exists() and ckpt_path.exists():
        logger.info("Loading precomputed serving metadata and ranker checkpoint...")
        try:
            with open(cache_path, "rb") as f:
                meta = pickle.load(f)
            with open(ranker_path, "rb") as f:
                ranker = pickle.load(f)

            num_users = len(meta["user_to_idx"])
            num_items = len(meta["item_to_idx"])

            two_tower = TwoTowerModel(
                num_users=num_users,
                num_items=num_items,
                num_genres=meta["genre_matrix"].shape[1],
                embedding_dim=64,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu")
            two_tower.load_state_dict(ckpt["model_state_dict"])
            two_tower.eval()

            index = FaissFlatIPIndex(dim=meta["item_embeddings"].shape[1])
            index.add(meta["item_embeddings"])

            top_popular = [meta["idx_to_item"][i] for i in meta["stats"]["item_counts"].argsort()[::-1][:50]]

            return ServingPipeline(
                two_tower_model=two_tower,
                item_index=index,
                ranker_model=ranker,
                item_embeddings=meta["item_embeddings"],
                movie_dict=movie_dict,
                user_to_idx=meta["user_to_idx"],
                item_to_idx=meta["item_to_idx"],
                idx_to_item=meta["idx_to_item"],
                popular_items=top_popular,
                stats=meta["stats"],
                genre_matrix=meta["genre_matrix"],
                year_array=meta["year_array"],
                user_histories=meta["user_train_history_map"],
                user_last_item_map=meta["user_last_item_map"],
                cooccur_map=meta["cooccur_map"],
                user_ratings_history=meta["user_ratings_history"],
            )
        except Exception as e:
            logger.error("Error loading cache: %s. Rebuilding pipeline from raw data...", str(e))

    # Rebuild metadata and train ranker if cache missing
    logger.info("Building serving pipeline from raw data...")
    df_ratings = pd.read_csv(ratings_path)
    train_df, val_df, _, metadata = time_based_split(df_ratings, 0.70, 0.15, 0.15, 5, 3)

    num_users = metadata["num_users"]
    num_items = metadata["num_items"]
    item_to_idx = metadata["item_to_idx"]
    idx_to_item = {idx: mid for mid, idx in item_to_idx.items()}
    user_to_idx = metadata["user_to_idx"]

    genre_matrix, year_array, genre_names = extract_movie_features(df_movies, item_to_idx)
    stats = compute_training_statistics(train_df, genre_matrix, num_users, num_items)
    cooccur_map = build_item_cooccurrence_matrix(train_df, num_items)

    max_hist_len = 20
    user_train_history_map = {}
    user_last_item_map = {}
    for u, u_group in train_df.groupby("user_idx"):
        last_items = u_group["item_idx"].values[-max_hist_len:]
        h_arr = np.zeros(max_hist_len, dtype=np.int64)
        w_arr = np.zeros(max_hist_len, dtype=np.float32)
        n_h = len(last_items)
        if n_h > 0:
            h_arr[-n_h:] = last_items
            w_arr[-n_h:] = 1.0 / n_h
            user_last_item_map[u] = int(last_items[-1])
        user_train_history_map[u] = (h_arr, w_arr)

    user_ratings_history = {}
    for u_id, group in df_ratings.groupby("userId"):
        user_ratings_history[int(u_id)] = group[["movieId", "rating", "timestamp"]].sort_values(by="timestamp", ascending=False).to_dict(orient="records")

    two_tower = TwoTowerModel(num_users=num_users, num_items=num_items, num_genres=genre_matrix.shape[1], embedding_dim=64)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        two_tower.load_state_dict(ckpt["model_state_dict"])
    two_tower.eval()

    with torch.no_grad():
        i_ids = torch.arange(num_items, dtype=torch.long)
        g = torch.from_numpy(genre_matrix).float()
        y = torch.from_numpy(year_array[:, None]).float()
        item_embs = two_tower.encode_items(i_ids, g, y).cpu().numpy()

    index = FaissFlatIPIndex(dim=item_embs.shape[1])
    index.add(item_embs)

    train_cands = generate_candidates_for_split(two_tower, index, train_df, user_train_history_map, top_k=50)
    val_cands = generate_candidates_for_split(two_tower, index, val_df, user_train_history_map, top_k=50)
    X_train, y_train, group_train, feat_names = extract_ranking_features(train_cands, stats, genre_matrix, year_array, user_last_item_map, cooccur_map, include_two_tower_score=True)
    X_val, y_val, group_val, _ = extract_ranking_features(val_cands, stats, genre_matrix, year_array, user_last_item_map, cooccur_map, include_two_tower_score=True)

    ranker = LGBMRankModel(n_estimators=60, early_stopping_rounds=10).fit(X_train, y_train, group_train, X_val, y_val, group_val, feature_names=feat_names)

    # Persist cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ranker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ranker_path, "wb") as f:
        pickle.dump(ranker, f)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "item_to_idx": item_to_idx,
            "idx_to_item": idx_to_item,
            "user_to_idx": user_to_idx,
            "genre_matrix": genre_matrix,
            "year_array": year_array,
            "genre_names": genre_names,
            "stats": stats,
            "cooccur_map": cooccur_map,
            "user_train_history_map": user_train_history_map,
            "user_last_item_map": user_last_item_map,
            "user_ratings_history": user_ratings_history,
            "item_embeddings": item_embs,
        }, f)

    top_popular = [idx_to_item[i] for i in stats["item_counts"].argsort()[::-1][:50]]

    return ServingPipeline(
        two_tower_model=two_tower,
        item_index=index,
        ranker_model=ranker,
        item_embeddings=item_embs,
        movie_dict=movie_dict,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        popular_items=top_popular,
        stats=stats,
        genre_matrix=genre_matrix,
        year_array=year_array,
        user_histories=user_train_history_map,
        user_last_item_map=user_last_item_map,
        cooccur_map=cooccur_map,
        user_ratings_history=user_ratings_history,
    )
