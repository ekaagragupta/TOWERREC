"""Vectorized Off-Policy Evaluation (OPE) estimators: IPS, SNIPS, DM, DR, SWITCH-DR."""

import numpy as np


def compute_importance_weights(
    eval_probs: np.ndarray,
    logging_probs: np.ndarray,
    clip_threshold: float | None = None,
) -> np.ndarray:
    """Computes importance weights w = pi_e(a|x) / pi_0(a|x) with optional clipping.

    Args:
        eval_probs: (N,) probabilities of logged actions under evaluation policy.
        logging_probs: (N,) probabilities of logged actions under logging policy.
        clip_threshold: Maximum allowed weight value (M).

    Returns:
        (N,) array of importance weights.
    """
    weights = np.asarray(eval_probs, dtype=np.float64) / np.maximum(
        np.asarray(logging_probs, dtype=np.float64), 1e-12
    )
    if clip_threshold is not None and clip_threshold > 0:
        weights = np.minimum(weights, clip_threshold)
    return weights


def inverse_propensity_scoring(
    rewards: np.ndarray,
    eval_probs: np.ndarray,
    logging_probs: np.ndarray,
    clip_threshold: float | None = None,
) -> float:
    """Inverse Propensity Scoring (IPS) estimator: V_IPS = (1/N) * sum(w_i * r_i)."""
    weights = compute_importance_weights(eval_probs, logging_probs, clip_threshold)
    return float(np.mean(weights * np.asarray(rewards, dtype=np.float64)))


def self_normalized_ips(
    rewards: np.ndarray,
    eval_probs: np.ndarray,
    logging_probs: np.ndarray,
    clip_threshold: float | None = None,
) -> float:
    """Self-Normalized IPS (SNIPS) estimator: V_SNIPS = sum(w_i * r_i) / sum(w_i).

    Guaranteed bounded in [min(rewards), max(rewards)], trading small bias for lower variance.
    """
    weights = compute_importance_weights(eval_probs, logging_probs, clip_threshold)
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return 0.0
    return float(np.sum(weights * np.asarray(rewards, dtype=np.float64)) / w_sum)


def direct_method(
    eval_policy_action_probs: np.ndarray,
    reward_model_preds: np.ndarray,
) -> float:
    """Direct Method (DM) estimator: V_DM = (1/N) * sum_i sum_a pi_e(a|x_i) * q_hat(x_i, a).

    Args:
        eval_policy_action_probs: (N, |A|) action probability distribution under pi_e.
        reward_model_preds: (N, |A|) predicted expected reward from fitted reward model.
    """
    expected_rewards_per_context = np.sum(
        np.asarray(eval_policy_action_probs, dtype=np.float64) * np.asarray(reward_model_preds, dtype=np.float64),
        axis=-1,
    )
    return float(np.mean(expected_rewards_per_context))


def doubly_robust(
    rewards: np.ndarray,
    eval_probs: np.ndarray,
    logging_probs: np.ndarray,
    reward_model_logged_preds: np.ndarray,
    eval_policy_action_probs: np.ndarray,
    reward_model_all_preds: np.ndarray,
    clip_threshold: float | None = None,
) -> float:
    """Doubly Robust (DR) estimator combining DM and IPS residuals.

    Formula:
        V_DR = V_DM + (1/N) * sum(w_i * (r_i - q_hat(x_i, a_i)))
    Unbiased if either logging propensities pi_0 or reward model q_hat is correctly specified.
    """
    v_dm = direct_method(eval_policy_action_probs, reward_model_all_preds)
    weights = compute_importance_weights(eval_probs, logging_probs, clip_threshold)
    residuals = np.asarray(rewards, dtype=np.float64) - np.asarray(reward_model_logged_preds, dtype=np.float64)
    correction = np.mean(weights * residuals)
    return float(v_dm + correction)


def switch_doubly_robust(
    rewards: np.ndarray,
    eval_probs: np.ndarray,
    logging_probs: np.ndarray,
    reward_model_logged_preds: np.ndarray,
    eval_policy_action_probs: np.ndarray,
    reward_model_all_preds: np.ndarray,
    tau: float = 20.0,
) -> float:
    """SWITCH-DR estimator: switches from DR to DM when importance weight exceeds threshold tau."""
    unclipped_weights = compute_importance_weights(eval_probs, logging_probs, clip_threshold=None)
    v_dm = direct_method(eval_policy_action_probs, reward_model_all_preds)

    # Use weight only where weight <= tau
    mask = unclipped_weights <= tau
    residuals = np.asarray(rewards, dtype=np.float64) - np.asarray(reward_model_logged_preds, dtype=np.float64)
    correction = np.sum(unclipped_weights * residuals * mask) / len(rewards)

    return float(v_dm + correction)
