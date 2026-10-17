"""FAISS index implementations: FlatIP (exact), HNSW, and IVF-PQ with benchmarking."""

import time

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from recsys.utils.logger import get_logger

logger = get_logger("recsys.retrieval.faiss_index")


class BaseItemIndex:
    """Base interface for vector similarity search."""

    def add(self, item_vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_vectors: np.ndarray, k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, item_indices) for each query."""
        raise NotImplementedError


class NumpyFlatIPIndex(BaseItemIndex):
    """Exact inner-product search implemented in pure NumPy as reference / fallback."""

    def __init__(self, dim: int):
        self.dim = dim
        self.item_vectors: np.ndarray | None = None

    def add(self, item_vectors: np.ndarray) -> None:
        self.item_vectors = item_vectors.astype(np.float32)

    def search(self, query_vectors: np.ndarray, k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        if self.item_vectors is None:
            raise RuntimeError("Index is empty.")
        k = min(k, self.item_vectors.shape[0])
        sims = np.dot(query_vectors, self.item_vectors.T)  # (N, I)
        top_indices = np.argpartition(-sims, range(k), axis=1)[:, :k]
        sorted_top_indices = np.zeros_like(top_indices)
        scores = np.zeros((len(query_vectors), k), dtype=np.float32)

        for i in range(len(query_vectors)):
            sub_scores = sims[i, top_indices[i]]
            sub_order = np.argsort(-sub_scores)
            sorted_top_indices[i] = top_indices[i, sub_order]
            scores[i] = sub_scores[sub_order]

        return scores, sorted_top_indices


class FaissFlatIPIndex(BaseItemIndex):
    """FAISS exact inner-product index (IndexFlatIP)."""

    def __init__(self, dim: int):
        self.dim = dim
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = NumpyFlatIPIndex(dim)

    def add(self, item_vectors: np.ndarray) -> None:
        if HAS_FAISS:
            self.index.add(np.ascontiguousarray(item_vectors, dtype=np.float32))
        else:
            self.index.add(item_vectors)

    def search(self, query_vectors: np.ndarray, k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        if HAS_FAISS:
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(
                np.ascontiguousarray(query_vectors, dtype=np.float32), k
            )
            return scores, indices
        return self.index.search(query_vectors, k)


class FaissHNSWIndex(BaseItemIndex):
    """FAISS Hierarchical Navigable Small World (HNSW) index."""

    def __init__(self, dim: int, m: int = 32, ef_construction: int = 64, ef_search: int = 64):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        if HAS_FAISS:
            self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            self.index = NumpyFlatIPIndex(dim)

    def add(self, item_vectors: np.ndarray) -> None:
        if HAS_FAISS:
            self.index.add(np.ascontiguousarray(item_vectors, dtype=np.float32))
        else:
            self.index.add(item_vectors)

    def search(self, query_vectors: np.ndarray, k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        if HAS_FAISS:
            k = min(k, self.index.ntotal)
            self.index.hnsw.efSearch = self.ef_search
            scores, indices = self.index.search(
                np.ascontiguousarray(query_vectors, dtype=np.float32), k
            )
            return scores, indices
        return self.index.search(query_vectors, k)


class FaissIVFPQIndex(BaseItemIndex):
    """FAISS Inverted File Product Quantization (IVF-PQ) index."""

    def __init__(self, dim: int, nlist: int = 32, nprobe: int = 8, m_pq: int = 8, nbits: int = 8):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.m_pq = m_pq
        self.nbits = nbits

        # Ensure m_pq divides dim
        while dim % self.m_pq != 0 and self.m_pq > 1:
            self.m_pq //= 2

        if HAS_FAISS:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, dim, nlist, self.m_pq, nbits, faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = nprobe
            self.is_trained = False
        else:
            self.index = NumpyFlatIPIndex(dim)
            self.is_trained = True

    def add(self, item_vectors: np.ndarray) -> None:
        if HAS_FAISS:
            data = np.ascontiguousarray(item_vectors, dtype=np.float32)
            n_samples = len(data)
            if not self.is_trained:
                # Ensure 2^nbits <= n_samples and nlist <= n_samples
                safe_nbits = self.nbits
                while (1 << safe_nbits) > n_samples and safe_nbits > 1:
                    safe_nbits -= 1

                safe_nlist = min(self.nlist, max(1, n_samples // 4))

                quantizer = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIVFPQ(
                    quantizer, self.dim, safe_nlist, self.m_pq, safe_nbits, faiss.METRIC_INNER_PRODUCT
                )
                self.index.nprobe = min(self.nprobe, safe_nlist)
                self.index.train(data)
                self.is_trained = True
            self.index.add(data)
        else:
            self.index.add(item_vectors)

    def search(self, query_vectors: np.ndarray, k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        if HAS_FAISS:
            k = min(k, self.index.ntotal)
            self.index.nprobe = self.nprobe
            scores, indices = self.index.search(
                np.ascontiguousarray(query_vectors, dtype=np.float32), k
            )
            return scores, indices
        return self.index.search(query_vectors, k)


def benchmark_ann(
    exact_index: BaseItemIndex,
    ann_index: BaseItemIndex,
    query_vectors: np.ndarray,
    k: int = 50,
) -> dict[str, float]:
    """Benchmarks an approximate nearest neighbor index against exact FlatIP.

    Returns:
        Dictionary with recall_at_k, recall_loss, p50_ms, p95_ms, p99_ms, qps.
    """
    # 1. Exact ground truth
    _, exact_indices = exact_index.search(query_vectors, k=k)

    # 2. ANN Latency & retrieval
    n_queries = len(query_vectors)
    latencies_ms: list[float] = []

    ann_results = []
    for i in range(n_queries):
        q = query_vectors[i : i + 1]
        t0 = time.perf_counter()
        _, idxs = ann_index.search(q, k=k)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        ann_results.append(idxs[0])

    # 3. Recall calculation
    recall_hits = 0
    total_expected = n_queries * k
    for i in range(n_queries):
        gt_set = set(exact_indices[i][:k])
        ann_set = set(ann_results[i][:k])
        recall_hits += len(gt_set.intersection(ann_set))

    recall_at_k = float(recall_hits / total_expected) if total_expected > 0 else 1.0
    recall_loss = 1.0 - recall_at_k

    return {
        "recall_at_k": recall_at_k,
        "recall_loss": recall_loss,
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "qps": float(n_queries / (sum(latencies_ms) / 1000.0)) if sum(latencies_ms) > 0 else 0.0,
    }
