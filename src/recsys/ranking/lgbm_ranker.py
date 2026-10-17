"""LightGBM LambdaRank ranking model with query grouping and feature importance."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.utils.logger import get_logger

logger = get_logger("recsys.ranking.lgbm_ranker")


class LGBMRankModel:
    """LightGBM LambdaRank re-ranker optimizing NDCG with query groups."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_estimators: int = 150,
        min_child_samples: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 15,
        ndcg_eval_at: tuple[int, ...] = (10,),
        importance_type: str = "gain",
        seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.ndcg_eval_at = list(ndcg_eval_at)
        self.importance_type = importance_type
        self.seed = seed

        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        group_val: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> "LGBMRankModel":
        """Fits LightGBM LambdaRank booster."""
        self.feature_names = feature_names if feature_names else [f"f_{i}" for i in range(X_train.shape[1])]

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=group_train,
            feature_name=self.feature_names,
            free_raw_data=False,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and group_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=group_val,
                feature_name=self.feature_names,
                reference=train_data,
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": self.ndcg_eval_at,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.seed,
            "num_threads": 1,
            "verbose": -1,
            "force_col_wise": True,
        }

        callbacks = [lgb.log_evaluation(period=25)]
        if self.early_stopping_rounds and len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))

        logger.info("Training LightGBM LambdaRank (%d queries, %d features)...", len(group_train), len(self.feature_names))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Computes ranking scores for candidate feature matrix."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict(X)

    def get_feature_importances(self) -> dict[str, float]:
        """Returns feature importance dictionary sorted by importance descending."""
        if self.model is None:
            return {}
        raw_importances = self.model.feature_importance(importance_type=self.importance_type)
        total = sum(raw_importances) if sum(raw_importances) > 0 else 1.0
        norm_importances = {name: float(imp / total) for name, imp in zip(self.feature_names, raw_importances)}
        return dict(sorted(norm_importances.items(), key=lambda item: item[1], reverse=True))

    def rerank(
        self,
        candidate_df: pd.DataFrame,
        features_X: np.ndarray,
        top_n: int = 10,
    ) -> dict[int, list[int]]:
        """Scores candidates and extracts top-N recommendations per user query."""
        scores = self.predict(features_X)
        df = candidate_df.copy()
        df["rank_score"] = scores

        recommendations: dict[int, list[int]] = {}
        for u, group in df.groupby("user_idx"):
            sorted_items = group.sort_values(by="rank_score", ascending=False)["item_idx"].head(top_n).tolist()
            recommendations[u] = [int(i) for i in sorted_items]

        return recommendations
