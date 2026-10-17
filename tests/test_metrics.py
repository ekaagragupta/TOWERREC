"""Unit tests for ranking and diversity metrics against hand-computed ground truth."""

import numpy as np
import pytest

from recsys.evaluation.diversity import catalog_coverage, gini_index
from recsys.evaluation.metrics import (
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_hand_computed_recall_and_precision():
    actual = [1, 2, 3]
    predicted = [1, 4, 5, 2, 6]

    # At K=3: predicted[:3] = [1, 4, 5], hits = [1]
    assert recall_at_k(actual, predicted, k=3) == pytest.approx(1.0 / 3.0)
    assert precision_at_k(actual, predicted, k=3) == pytest.approx(1.0 / 3.0)

    # At K=4: predicted[:4] = [1, 4, 5, 2], hits = [1, 2]
    assert recall_at_k(actual, predicted, k=4) == pytest.approx(2.0 / 3.0)
    assert precision_at_k(actual, predicted, k=4) == pytest.approx(2.0 / 4.0)

    # At K=1: hits = [1]
    assert recall_at_k(actual, predicted, k=1) == pytest.approx(1.0 / 3.0)
    assert precision_at_k(actual, predicted, k=1) == pytest.approx(1.0)


def test_hand_computed_ndcg():
    actual = [1, 2]
    predicted = [1, 3, 2, 4]

    # Rank 1: item 1 (hit) -> 1 / log2(2) = 1.0
    # Rank 2: item 3 (miss)
    # Rank 3: item 2 (hit) -> 1 / log2(4) = 0.5
    # DCG@3 = 1.0 + 0.5 = 1.5
    # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975357 = 1.63092975357
    expected_ndcg = 1.5 / (1.0 + 1.0 / np.log2(3.0))
    assert ndcg_at_k(actual, predicted, k=3) == pytest.approx(expected_ndcg, rel=1e-5)

    # Perfect ranking: [1, 2, 3, 4] -> NDCG = 1.0
    assert ndcg_at_k(actual, [1, 2, 5, 6], k=2) == pytest.approx(1.0)
    # Zero hits -> NDCG = 0.0
    assert ndcg_at_k(actual, [9, 8, 7], k=3) == pytest.approx(0.0)


def test_hand_computed_map_and_mrr():
    actual = [1, 2]
    predicted = [1, 3, 2, 4]

    # At K=3:
    # rank 1 hit: P@1 = 1/1 = 1.0
    # rank 2 miss
    # rank 3 hit: P@3 = 2/3
    # AP@3 = (1.0 + 2/3) / min(3, 2) = (5/3) / 2 = 5/6 = 0.833333
    assert map_at_k(actual, predicted, k=3) == pytest.approx(5.0 / 6.0)

    # MRR: first hit at rank 1 -> 1.0
    assert mrr_at_k(actual, predicted, k=3) == pytest.approx(1.0)

    # First hit at rank 3:
    assert mrr_at_k(actual, [5, 6, 1, 2], k=3) == pytest.approx(1.0 / 3.0)


def test_hand_computed_diversity():
    # Catalog coverage
    recs = {1: [10, 20], 2: [20, 30]}
    assert catalog_coverage(recs, num_total_items=100) == pytest.approx(3 / 100)

    # Gini Index: perfectly equal distribution should be 0.0
    equal_freqs = [10.0, 10.0, 10.0, 10.0]
    assert gini_index(equal_freqs) == pytest.approx(0.0, abs=1e-5)

    # Concentrated distribution: 1 item has everything
    unequal_freqs = [0.0, 0.0, 0.0, 100.0]
    # For n=4, y=[0, 0, 0, 100]: (2 * 4 * 100)/(4 * 100) - 5/4 = 2 - 1.25 = 0.75
    assert gini_index(unequal_freqs) == pytest.approx(0.75, abs=1e-5)
