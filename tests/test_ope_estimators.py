"""Unit tests verifying OPE estimators against closed-form toy bandit cases."""

import numpy as np
import pytest

from recsys.ope.diagnostics import check_support_overlap, effective_sample_size
from recsys.ope.estimators import (
    direct_method,
    doubly_robust,
    inverse_propensity_scoring,
    self_normalized_ips,
)


def test_ips_unbiasedness_on_toy_bandit():
    """Validates that IPS converges to the exact expected value on a known 2-armed bandit."""
    np.random.seed(42)
    n_samples = 100000

    # Arms: 0 and 1
    # Ground truth expected rewards
    q_star = np.array([0.20, 0.80])

    # Logging policy: pi_0 = [0.40, 0.60]
    pi_0 = np.array([0.40, 0.60])

    # Evaluation policy: pi_e = [0.10, 0.90]
    pi_e = np.array([0.10, 0.90])

    # True target policy value
    true_val = float(np.dot(pi_e, q_star))  # 0.10 * 0.20 + 0.90 * 0.80 = 0.74

    # Sample logged actions from pi_0
    actions = np.random.choice([0, 1], size=n_samples, p=pi_0)
    logging_probs = pi_0[actions]
    eval_probs = pi_e[actions]

    # Sample rewards r ~ Bernoulli(q_star[a])
    rewards = (np.random.random(size=n_samples) < q_star[actions]).astype(np.float64)

    # Estimate via IPS
    v_ips = inverse_propensity_scoring(rewards, eval_probs, logging_probs)

    # Assert within 1.5% of true value
    assert v_ips == pytest.approx(true_val, abs=0.015)


def test_snips_boundedness():
    """Validates that SNIPS is strictly bounded between [min(r), max(r)] regardless of weights."""
    np.random.seed(42)
    n = 1000
    rewards = np.random.uniform(0.0, 1.0, size=n)

    # Extremely skewed eval vs logging probabilities creating massive weights
    logging_probs = np.full(n, 0.001)
    eval_probs = np.full(n, 0.999)

    v_snips = self_normalized_ips(rewards, eval_probs, logging_probs)
    assert 0.0 <= v_snips <= 1.0


def test_doubly_robust_reduction_properties():
    """Validates that DR reduces to IPS when reward model is zero, and matches DM when residuals are zero."""
    rewards = np.array([1.0, 0.0, 1.0, 1.0])
    eval_probs = np.array([0.5, 0.5, 0.8, 0.2])
    logging_probs = np.array([0.25, 0.25, 0.4, 0.1])

    # Case 1: When reward model prediction is 0 everywhere
    q_hat_logged_zeros = np.zeros(4)
    eval_matrix = np.column_stack([eval_probs, 1.0 - eval_probs])
    q_hat_matrix_zeros = np.zeros_like(eval_matrix)

    v_dr_zero_q = doubly_robust(
        rewards,
        eval_probs,
        logging_probs,
        q_hat_logged_zeros,
        eval_matrix,
        q_hat_matrix_zeros,
    )
    v_ips = inverse_propensity_scoring(rewards, eval_probs, logging_probs)
    assert v_dr_zero_q == pytest.approx(v_ips)

    # Case 2: When reward model exactly predicts observed rewards (residuals = 0)
    v_dr_perfect = doubly_robust(
        rewards,
        eval_probs,
        logging_probs,
        rewards,  # residuals = 0
        eval_matrix,
        np.column_stack([rewards, np.zeros(4)]),
    )
    v_dm = direct_method(eval_matrix, np.column_stack([rewards, np.zeros(4)]))
    assert v_dr_perfect == pytest.approx(v_dm)


def test_diagnostics():
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    # ESS for equal weights is N = 4
    assert effective_sample_size(weights) == pytest.approx(4.0)

    # One huge weight collapses ESS
    skewed_weights = np.array([0.0, 0.0, 0.0, 100.0])
    assert effective_sample_size(skewed_weights) == pytest.approx(1.0)

    # Support violation check
    ep = np.array([0.5, 0.2, 0.0])
    lp = np.array([0.5, 0.0, 0.1])  # index 1 has ep > 0 and lp == 0
    violations, pct = check_support_overlap(ep, lp)
    assert violations == 1
    assert pct == pytest.approx(100.0 / 3.0)
