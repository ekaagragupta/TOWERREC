"""Standalone ANN benchmark comparing FAISS FlatIP, HNSW, and IVF-PQ."""

import argparse

import numpy as np
import pandas as pd

from recsys.retrieval.faiss_index import (
    FaissFlatIPIndex,
    FaissHNSWIndex,
    FaissIVFPQIndex,
    benchmark_ann,
)
from recsys.utils.logger import get_logger

logger = get_logger("recsys.benchmark_ann")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS ANN index variants.")
    parser.add_argument("--num_items", type=int, default=5000, help="Number of catalog items.")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of test queries.")
    parser.add_argument("--k", type=int, default=50, help="Top-K nearest neighbors.")
    args = parser.parse_args()

    np.random.seed(42)
    logger.info("Generating %d items and %d queries (dim=%d)...", args.num_items, args.num_queries, args.dim)

    items = np.random.randn(args.num_items, args.dim).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)

    queries = np.random.randn(args.num_queries, args.dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    exact_idx = FaissFlatIPIndex(dim=args.dim)
    exact_idx.add(items)

    hnsw_idx = FaissHNSWIndex(dim=args.dim, m=32, ef_construction=64, ef_search=64)
    hnsw_idx.add(items)

    ivfpq_idx = FaissIVFPQIndex(dim=args.dim, nlist=32, nprobe=8, m_pq=8, nbits=8)
    ivfpq_idx.add(items)

    results = {
        "FlatIP (Exact)": benchmark_ann(exact_idx, exact_idx, queries, k=args.k),
        "HNSW (Graph)": benchmark_ann(exact_idx, hnsw_idx, queries, k=args.k),
        "IVF-PQ (Quantized)": benchmark_ann(exact_idx, ivfpq_idx, queries, k=args.k),
    }

    df = pd.DataFrame(results).T
    print("\n=== ANN INDEX BENCHMARK RESULTS ===")
    print(df.to_string())


if __name__ == "__main__":
    main()
