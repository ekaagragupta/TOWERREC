"""End-to-end smoke integration test executing the full pipeline on dev scale."""

from scripts.run_ope_study import run_ope_study
from scripts.run_pipeline import run_pipeline


def test_pipeline_smoke_dev():
    """Runs end-to-end dev pipeline verifying all stages execute and produce output tables."""
    results = run_pipeline(config_path="configs/dev.yaml", stage="all")

    assert "metrics" in results
    assert "Two-Tower + LightGBM" in results["metrics"]
    assert "Global Popularity" in results["metrics"]

    # Verify NDCG@10 is computed and non-zero
    lgbm_ndcg = results["metrics"]["Two-Tower + LightGBM"]["ndcg@10"]
    assert lgbm_ndcg > 0.0

    # Verify ANN benchmarks exist
    assert "FlatIP (Exact)" in results["ann"]
    assert "HNSW" in results["ann"]

    # Verify serving latencies are present
    assert results["latency"]["p50_total_ms"] > 0.0


def test_ope_study_smoke_dev():
    """Runs OPE study verifying temperature and sample size sweeps execute successfully."""
    results = run_ope_study(config_path="configs/dev.yaml")

    assert "true_values" in results
    assert "Two-Tower + LightGBM" in results["true_values"]
    assert "temp_results" in results
    assert len(results["temp_results"]["temperature"]) > 0
