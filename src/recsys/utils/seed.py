"""Deterministic seed utilities for reproducibility across random, numpy, and torch."""

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch (if available).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
