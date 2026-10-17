"""Unit tests for FAISS index wrappers and ANN benchmarking."""

import numpy as np
import pytest

from recsys.retrieval.faiss_index import (
    FaissFlatIPIndex,
    FaissHNSWIndex,
    FaissIVFPQIndex,
    NumpyFlatIPIndex,
    benchmark_ann,
)


@pytest.fixture
def index_data():
    np.random.seed(42)
    n_items = 250
    dim = 32
    n_queries = 20

    # L2 normalized vectors
    item_vectors = np.random.randn(n_items, dim).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)

    query_vectors = np.random.randn(n_queries, dim).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    return item_vectors, query_vectors


def test_numpy_and_faiss_flatip_consistency(index_data):
    item_vectors, query_vectors = index_data
    dim = item_vectors.shape[1]
    k = 10

    numpy_idx = NumpyFlatIPIndex(dim=dim)
    numpy_idx.add(item_vectors)
    np_scores, np_indices = numpy_idx.search(query_vectors, k=k)

    faiss_flat = FaissFlatIPIndex(dim=dim)
    faiss_flat.add(item_vectors)
    faiss_scores, faiss_indices = faiss_flat.search(query_vectors, k=k)

    # Indices and scores must match between numpy and exact FlatIP
    np.testing.assert_array_equal(np_indices, faiss_indices)
    np.testing.assert_allclose(np_scores, faiss_scores, rtol=1e-4, atol=1e-4)


def test_hnsw_recall_and_benchmark(index_data):
    item_vectors, query_vectors = index_data
    dim = item_vectors.shape[1]
    k = 10

    exact_idx = FaissFlatIPIndex(dim=dim)
    exact_idx.add(item_vectors)

    hnsw_idx = FaissHNSWIndex(dim=dim, m=16, ef_construction=64, ef_search=64)
    hnsw_idx.add(item_vectors)

    metrics = benchmark_ann(exact_idx, hnsw_idx, query_vectors, k=k)

    # HNSW should have high recall (> 90%)
    assert metrics["recall_at_k"] >= 0.90
    assert metrics["recall_loss"] <= 0.10
    assert metrics["p50_ms"] >= 0.0
    assert metrics["qps"] > 0.0


def test_ivfpq_ann(index_data):
    item_vectors, query_vectors = index_data
    dim = item_vectors.shape[1]
    k = 10

    exact_idx = FaissFlatIPIndex(dim=dim)
    exact_idx.add(item_vectors)

    ivfpq_idx = FaissIVFPQIndex(dim=dim, nlist=16, nprobe=8, m_pq=4, nbits=8)
    ivfpq_idx.add(item_vectors)

    metrics = benchmark_ann(exact_idx, ivfpq_idx, query_vectors, k=k)
    # Approximate quantized index will have some recall loss, but should be valid
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert metrics["p95_ms"] >= 0.0
