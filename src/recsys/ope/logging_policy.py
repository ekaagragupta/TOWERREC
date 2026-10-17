"""Known stochastic logging policy with popularity bias and epsilon-uniform exploration."""

from collections.abc import Sequence

import numpy as np


class PopularityLoggingPolicy:
    """Stochastic logging policy: pi_0(a|x) = (1 - eps) * Softmax(log(1 + pop(a)) / T) + eps / |A|."""

    def __init__(
        self,
        item_train_counts: np.ndarray,
        temperature: float = 1.5,
        epsilon_uniform: float = 0.05,
    ):
        self.item_train_counts = item_train_counts
        self.temperature = max(temperature, 1e-4)
        self.epsilon_uniform = epsilon_uniform

    def get_action_probabilities(self, candidate_items: Sequence[int]) -> np.ndarray:
        """Computes propensity distribution pi_0(a|x) across candidate actions."""
        actions = np.asarray(candidate_items, dtype=np.int64)
        n_actions = len(actions)
        if n_actions == 0:
            return np.array([], dtype=np.float64)

        # Popularity score: log1p(item_counts)
        counts = self.item_train_counts[actions]
        logits = np.log1p(counts) / self.temperature

        # Softmax with numerical stability
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        softmax_probs = exp_logits / np.sum(exp_logits)

        # Mix with epsilon-uniform exploration for guaranteed support
        uniform_prob = 1.0 / n_actions
        probs = (1.0 - self.epsilon_uniform) * softmax_probs + self.epsilon_uniform * uniform_prob
        return probs

    def sample_action(
        self,
        candidate_items: Sequence[int],
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Samples an action according to pi_0 and returns (chosen_item, propensity)."""
        probs = self.get_action_probabilities(candidate_items)
        if rng is None:
            rng = np.random.default_rng()
        chosen_idx = rng.choice(len(candidate_items), p=probs)
        return int(candidate_items[chosen_idx]), float(probs[chosen_idx])
