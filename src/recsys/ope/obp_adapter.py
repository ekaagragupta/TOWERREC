"""Open Bandit Dataset (ZOZOTOWN) evaluation adapter with graceful offline fallback."""

from typing import Any

import numpy as np

from recsys.utils.logger import get_logger

logger = get_logger("recsys.ope.obp_adapter")


def run_open_bandit_evaluation(data_path: str | None = None) -> dict[str, Any]:
    """Runs OPE on Open Bandit Dataset using obp library if available, with graceful offline fallback.

    Compares off-policy estimators against on-policy ground-truth estimate from the random policy.
    """
    try:
        from obp.dataset import OpenBanditDataset
        from obp.ope import (
            DirectMethod,
            DoublyRobust,
            InversePropensityScoring,
            OffPolicyEvaluation,
            SelfNormalizedInversePropensityScoring,
        )

        logger.info("Open Bandit Pipeline (obp) detected. Loading ZOZOTOWN dataset...")
        dataset = OpenBanditDataset(behavior_policy="random", campaign="all", data_path=data_path)
        bandit_feedback = dataset.obtain_batch_bandit_feedback()

        # On-policy ground truth policy value of behavior policy
        on_policy_ground_truth = float(bandit_feedback["reward"].mean())

        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=[
                InversePropensityScoring(),
                SelfNormalizedInversePropensityScoring(),
                DirectMethod(),
                DoublyRobust(),
            ],
        )

        # Evaluate against a uniform Bernoulli policy as target evaluation policy
        n_rounds = bandit_feedback["n_rounds"]
        n_actions = dataset.n_actions
        action_dist = np.ones((n_rounds, n_actions, 1)) / n_actions

        estimated_values = ope.estimate_policy_values(action_dist=action_dist)

        return {
            "status": "success",
            "is_real_data": True,
            "on_policy_ground_truth": on_policy_ground_truth,
            "ips_estimate": float(estimated_values.get("ips", 0.0)),
            "snips_estimate": float(estimated_values.get("snips", 0.0)),
            "dm_estimate": float(estimated_values.get("dm", 0.0)),
            "dr_estimate": float(estimated_values.get("dr", 0.0)),
        }

    except Exception as e:
        logger.warning(
            "Open Bandit Pipeline/Dataset unavailable or network blocked: %s. "
            "Gracefully degrading: documenting real-track status and relying on our validated semi-synthetic benchmark.",
            str(e),
        )
        return {
            "status": "degraded_graceful",
            "is_real_data": False,
            "error_reason": str(e),
            "note": "Documented graceful degradation per specification; synthetic benchmark used for primary validation.",
        }
