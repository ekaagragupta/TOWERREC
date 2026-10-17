"""Sampled Softmax loss with logQ correction for popularity bias mitigation."""


import torch
import torch.nn.functional as F
from torch import nn


class SampledSoftmaxLoss(nn.Module):
    """In-batch Sampled Softmax Loss with optional logQ popularity correction.

    Mathematical formulation:
        s(u_i, v_j) = (u_i^T v_j) / tau - log(q_j)
    where q_j is the sampling probability (relative frequency) of item j in the training set.
    Correcting logits by log(q_j) yields unbiased softmax gradients under non-uniform negative sampling.
    """

    def __init__(self, temperature: float = 0.07, use_logq_correction: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_logq_correction = use_logq_correction

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        sampling_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes sampled softmax loss over the batch.

        Args:
            user_embeddings: (B, D) L2-normalized user vectors.
            item_embeddings: (B, D) L2-normalized item vectors.
            sampling_probs: (B,) sampling probability / relative frequency of each item in batch.

        Returns:
            Scalar cross-entropy loss.
        """
        # Similarity matrix between all users and all items in batch: (B, B)
        similarity = torch.matmul(user_embeddings, item_embeddings.T) / self.temperature

        if self.use_logq_correction and sampling_probs is not None:
            # Broadcast log(q_j) across user queries: (1, B)
            logq = torch.log(sampling_probs.clamp(min=1e-8)).unsqueeze(0)
            similarity = similarity - logq

        # Targets are positive matches on diagonal
        batch_size = user_embeddings.size(0)
        targets = torch.arange(batch_size, device=user_embeddings.device)

        return F.cross_entropy(similarity, targets)
