"""Semi-synthetic bandit simulator and ground-truth counterfactual environment."""

import numpy as np
import pandas as pd

from recsys.ope.diagnostics import summarize_importance_weights
from recsys.ope.estimators import (
    direct_method,
    doubly_robust,
    inverse_propensity_scoring,
    self_normalized_ips,
    switch_doubly_robust,
)
from recsys.ope.logging_policy import PopularityLoggingPolicy
from recsys.utils.logger import get_logger

logger = get_logger("recsys.ope.simulator")


class SemiSyntheticBanditSimulator:
    """Semi-synthetic bandit environment with known ground truth reward model and logging policy."""

    def __init__(
        self,
        test_df: pd.DataFrame,
        item_train_counts: np.ndarray,
        reward_threshold: float = 4.0,
        candidate_pool_size: int = 50,
        seed: int = 42,
    ):
        self.item_train_counts = item_train_counts
        self.reward_threshold = reward_threshold
        self.candidate_pool_size = candidate_pool_size
        self.rng = np.random.default_rng(seed)

        # 1. Build observed rewards from test interactions
        self.user_positives = test_df[test_df["rating"] >= reward_threshold].groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.eval_users = list(self.user_positives.keys())

        # Candidate item pool: union of evaluated items and top popular items
        top_popular = np.argsort(-item_train_counts)[:candidate_pool_size].tolist()
        self.candidate_pool: dict[int, list[int]] = {}
        for u in self.eval_users:
            pos = list(self.user_positives.get(u, set()))
            pool = list(set(pos + top_popular))[:candidate_pool_size]
            self.candidate_pool[u] = pool

        # 2. Build ground-truth reward model q*(u, a)
        # Probability of positive reward: 0.95 for observed positive, 0.05 baseline
        self.ground_truth_q: dict[int, dict[int, float]] = {}
        for u, pool in self.candidate_pool.items():
            pos_set = self.user_positives.get(u, set())
            self.ground_truth_q[u] = {
                item: 0.95 if item in pos_set else 0.05 for item in pool
            }

    def compute_true_policy_value(
        self,
        policy_action_probs: dict[int, dict[int, float]],
    ) -> float:
        """Computes EXACT ground truth value V*(pi_e) = E_u E_{a ~ pi_e} [q*(u, a)]."""
        total_val = 0.0
        n_users = len(self.eval_users)
        for u in self.eval_users:
            pool = self.candidate_pool[u]
            u_probs = policy_action_probs.get(u, {})
            u_val = sum(u_probs.get(a, 0.0) * self.ground_truth_q[u].get(a, 0.05) for a in pool)
            total_val += u_val
        return float(total_val / n_users) if n_users > 0 else 0.0

    def generate_logged_bandit_data(
        self,
        logging_policy: PopularityLoggingPolicy,
        n_samples: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Simulates logged interactions (user, action, reward, propensity) from logging policy."""
        contexts: list[int] = []
        actions: list[int] = []
        rewards: list[float] = []
        propensities: list[float] = []
        pool_indices: list[int] = []

        for _ in range(n_samples):
            u = self.rng.choice(self.eval_users)
            pool = self.candidate_pool[u]

            chosen_item, p0 = logging_policy.sample_action(pool, rng=self.rng)
            # Sample stochastic reward from ground truth q*(u, chosen_item)
            expected_r = self.ground_truth_q[u].get(chosen_item, 0.05)
            r = 1.0 if self.rng.random() < expected_r else 0.0

            contexts.append(u)
            actions.append(chosen_item)
            rewards.append(r)
            propensities.append(p0)
            pool_indices.append(pool.index(chosen_item))

        return {
            "contexts": np.array(contexts, dtype=np.int64),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float64),
            "propensities": np.array(propensities, dtype=np.float64),
            "pool_indices": np.array(pool_indices, dtype=np.int64),
        }

    def fit_reward_model(
        self,
        bandit_data: dict[str, np.ndarray],
    ) -> dict[int, dict[int, float]]:
        """Fits an empirical reward model q_hat(u, a) on logged bandit feedback."""
        # Calculate observed mean reward per (u, a) pair with shrinkage toward global mean
        global_mean = float(np.mean(bandit_data["rewards"]))
        counts: dict[tuple[int, int], int] = {}
        sum_rewards: dict[tuple[int, int], float] = {}

        for u, a, r in zip(bandit_data["contexts"], bandit_data["actions"], bandit_data["rewards"]):
            counts[(u, a)] = counts.get((u, a), 0) + 1
            sum_rewards[(u, a)] = sum_rewards.get((u, a), 0.0) + r

        q_hat: dict[int, dict[int, float]] = {}
        shrinkage = 5.0
        for u, pool in self.candidate_pool.items():
            q_hat[u] = {}
            for a in pool:
                n = counts.get((u, a), 0)
                s = sum_rewards.get((u, a), 0.0)
                # Empirical Bayes shrinkage estimate
                est = (s + shrinkage * global_mean) / (n + shrinkage)
                q_hat[u][a] = float(est)

        return q_hat

    def evaluate_policy_ope(
        self,
        policy_action_probs: dict[int, dict[int, float]],
        bandit_data: dict[str, np.ndarray],
        q_hat: dict[int, dict[int, float]],
        clip_threshold: float = 20.0,
    ) -> dict[str, float]:
        """Runs IPS, SNIPS, DM, DR, and SWITCH-DR estimators on logged bandit data."""
        contexts = bandit_data["contexts"]
        actions = bandit_data["actions"]
        rewards = bandit_data["rewards"]
        logging_probs = bandit_data["propensities"]

        n_samples = len(rewards)
        eval_probs = np.zeros(n_samples, dtype=np.float64)
        reward_model_logged_preds = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            u = contexts[i]
            a = actions[i]
            eval_probs[i] = policy_action_probs.get(u, {}).get(a, 0.0)
            reward_model_logged_preds[i] = q_hat.get(u, {}).get(a, 0.1)

        # Construct full action distributions for DM / DR
        # For simplicity and vectorization, evaluate DM across the logged contexts
        max_pool = self.candidate_pool_size
        eval_action_probs_matrix = np.zeros((n_samples, max_pool), dtype=np.float64)
        q_hat_matrix = np.zeros((n_samples, max_pool), dtype=np.float64)

        for i in range(n_samples):
            u = contexts[i]
            pool = self.candidate_pool[u]
            for idx, a in enumerate(pool[:max_pool]):
                eval_action_probs_matrix[i, idx] = policy_action_probs.get(u, {}).get(a, 0.0)
                q_hat_matrix[i, idx] = q_hat.get(u, {}).get(a, 0.1)

        v_ips = inverse_propensity_scoring(rewards, eval_probs, logging_probs, clip_threshold=None)
        v_clipped_ips = inverse_propensity_scoring(rewards, eval_probs, logging_probs, clip_threshold=clip_threshold)
        v_snips = self_normalized_ips(rewards, eval_probs, logging_probs, clip_threshold=clip_threshold)
        v_dm = direct_method(eval_action_probs_matrix, q_hat_matrix)
        v_dr = doubly_robust(
            rewards,
            eval_probs,
            logging_probs,
            reward_model_logged_preds,
            eval_action_probs_matrix,
            q_hat_matrix,
            clip_threshold=clip_threshold,
        )
        v_switch_dr = switch_doubly_robust(
            rewards,
            eval_probs,
            logging_probs,
            reward_model_logged_preds,
            eval_action_probs_matrix,
            q_hat_matrix,
            tau=clip_threshold,
        )

        weights = eval_probs / np.maximum(logging_probs, 1e-12)
        diag = summarize_importance_weights(weights)

        return {
            "v_ips": v_ips,
            "v_clipped_ips": v_clipped_ips,
            "v_snips": v_snips,
            "v_dm": v_dm,
            "v_dr": v_dr,
            "v_switch_dr": v_switch_dr,
            "ess": diag["ess"],
            "max_weight": diag["max_weight"],
            "mean_weight": diag["mean_weight"],
        }
