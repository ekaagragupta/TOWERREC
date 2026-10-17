"""High-throughput async latency and benchmark load test for recommender serving."""

import asyncio
import time
from typing import Optional

import httpx
import numpy as np

from recsys.utils.logger import get_logger

logger = get_logger("recsys.serving.loadtest")


async def run_load_test(
    base_url: str = "http://127.0.0.1:8000",
    n_requests: int = 500,
    concurrency: int = 10,
    user_pool: Optional[list[int]] = None,
) -> dict[str, float]:
    """Runs concurrent load test against the recommendation server and reports latency percentiles.

    Returns:
        Dictionary containing p50, p95, p99, mean latency, throughput (QPS), and stage breakdown.
    """
    if user_pool is None:
        user_pool = list(range(1, 100))

    latencies_ms: list[float] = []
    feature_fetch_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    ranking_latencies: list[float] = []

    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        async def send_req(u_id: int):
            nonlocal success_count
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    resp = await client.post("/recommend", json={"user_id": u_id, "k": 10})
                    t1 = time.perf_counter()
                    if resp.status_code == 200:
                        latencies_ms.append((t1 - t0) * 1000.0)
                        data = resp.json()
                        st = data.get("stage_latencies", {})
                        feature_fetch_latencies.append(st.get("feature_fetch_ms", 0.0))
                        retrieval_latencies.append(st.get("retrieval_ms", 0.0))
                        ranking_latencies.append(st.get("ranking_ms", 0.0))
                        success_count += 1
                except Exception:
                    pass

        t_overall_start = time.perf_counter()
        tasks = [send_req(int(np.random.choice(user_pool))) for _ in range(n_requests)]
        await asyncio.gather(*tasks)
        t_overall_total = time.perf_counter() - t_overall_start

    qps = float(success_count / t_overall_total) if t_overall_total > 0 else 0.0

    return {
        "total_requests": n_requests,
        "successful_requests": success_count,
        "concurrency": concurrency,
        "throughput_qps": qps,
        "p50_total_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "p95_total_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "p99_total_ms": float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0,
        "mean_total_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "p50_feature_fetch_ms": float(np.percentile(feature_fetch_latencies, 50)) if feature_fetch_latencies else 0.0,
        "p50_retrieval_ms": float(np.percentile(retrieval_latencies, 50)) if retrieval_latencies else 0.0,
        "p50_ranking_ms": float(np.percentile(ranking_latencies, 50)) if ranking_latencies else 0.0,
    }
