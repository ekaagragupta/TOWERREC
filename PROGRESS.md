# Project Progress: Two-Tower Retrieval + Re-Ranker with Counterfactual Evaluation

## Project Status Overview
- **Active Workspace Directory**: `/Users/apple/.gemini/antigravity/scratch/recsys-twotower-ope`
- **Python Environment**: Python 3.12 (`.venv`)
- **Execution Target**: CPU-only, 16 GB RAM friendly, deterministic seeds
- **Status**: 100% COMPLETE & VERIFIED

---

## Phase Tracking & Verification

### Phase 0: Scaffolding, Environment, Configuration & CI
- [x] Implementation Plan artifact published (`implementation_plan.md`)
- [x] Virtual environment `.venv` configured with Python 3.12
- [x] `pyproject.toml` and `requirements.txt` written with pinned dependencies
- [x] Dependencies installed: PyTorch, FAISS (faiss-cpu), LightGBM, scikit-learn, FastAPI, uvicorn, pydantic, omegaconf, pytest, ruff
- [x] Configuration YAMLs (`dev.yaml`, `full.yaml`, model configs, ope configs) written
- [x] Utilities implemented (`seed.py`, `logger.py`, `config.py`, `io.py`)
- [x] GitHub Actions CI workflow (`.github/workflows/ci.yml`) and `Makefile` created
- [x] Code quality verified: `make lint` -> `All checks passed!`

### Phase 1: Data Pipeline, Time-Splits & Leakage Tests
- [x] Data downloader (`download.py`) with MovieLens checksum verification and synthetic fallback
- [x] Strict temporal splitter (`split.py`) enforcing $t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$
- [x] Feature extraction and recency-weighted history sequence builder (`features.py`)
- [x] Leakage and split unit tests (`tests/test_splits_and_leakage.py`) passing:
  - `test_time_based_split_ordering` PASSED
  - `test_no_history_leakage` PASSED
  - `test_no_train_feature_contamination` PASSED

### Phase 2: Offline Metrics & Baselines
- [x] Metrics module (`metrics.py`): NDCG@K, Recall@K, Precision@K, MAP@K, HitRate@K, MRR
- [x] Diversity & distribution metrics (`diversity.py`): catalog coverage, Gini index, avg popularity, tail share
- [x] Segmentation analysis (`segments.py`): cold/warm/heavy users, head/torso/tail items
- [x] Statistical significance and 95% bootstrap CIs (`bootstrap.py`)
- [x] Baselines implemented: Global Popularity, Item-KNN (cosine + shrinkage), PyTorch BPR-MF
- [x] Metrics unit tests (`tests/test_metrics.py`) passing against hand-computed ground truth:
  - `test_hand_computed_recall_and_precision` PASSED
  - `test_hand_computed_ndcg` PASSED
  - `test_hand_computed_map_and_mrr` PASSED
  - `test_hand_computed_diversity` PASSED

### Phase 3: Stage 1 - Two-Tower Retrieval & FAISS Indexing
- [x] User Tower & Item Tower PyTorch architectures (`towers.py`) with L2 normalization
- [x] Sampled Softmax loss with logQ popularity correction (`losses.py`)
- [x] PyTorch training loop with validation Recall@K early stopping & checkpointing (`trainer.py`)
- [x] FAISS index wrappers: FlatIP, HNSW, IVF-PQ (`faiss_index.py`)
- [x] ANN recall vs latency vs memory benchmarking (`scripts/benchmark_ann.py`)
- [x] FAISS unit test suite (`tests/test_faiss.py`) passing:
  - `test_numpy_and_faiss_flatip_consistency` PASSED
  - `test_hnsw_recall_and_benchmark` PASSED
  - `test_ivfpq_ann` PASSED

### Phase 4: Stage 2 - Ranker (LightGBM LambdaRank & DCN-v2)
- [x] Stage 1 candidate extraction without future leakage (`candidate_gen.py`)
- [x] 10 documented ranking features (`features.py`): retrieval score, rank, popularity, age, user activity, genre affinity, genre match count, co-occurrence, hour, day of week
- [x] LightGBM LambdaRank model training with group queries (`lgbm_ranker.py`)
- [x] Feature importance & gain summary extracted
- [x] DCN-v2 PyTorch neural ranker stretch (`dcnv2.py`) with Cross Network and Deep MLP

### Phase 5: Counterfactual Off-Policy Evaluation (OPE)
- [x] Vectorized estimators (`estimators.py`): IPS, SNIPS, DM, DR, SWITCH-DR with weight clipping
- [x] Diagnostics (`diagnostics.py`): Effective Sample Size (ESS), max/mean weights, support check
- [x] Semi-synthetic bandit simulator with known logging policy (`simulator.py`, `logging_policy.py`)
- [x] Open Bandit Dataset adapter (`obp_adapter.py`) with graceful fallback
- [x] OPE study runner sweeping temperature, sample size, clipping threshold (`scripts/run_ope_study.py`)
- [x] OPE unit tests (`tests/test_ope_estimators.py`) passing:
  - `test_ips_unbiasedness_on_toy_bandit` PASSED
  - `test_snips_boundedness` PASSED
  - `test_doubly_robust_reduction_properties` PASSED
  - `test_diagnostics` PASSED

### Phase 6: Production Serving & Containerization
- [x] Pydantic request/response schemas (`schemas.py`)
- [x] Serving pipeline loader with cold-start popularity fallback (`loader.py`)
- [x] FastAPI application (`app.py`) with `/health`, `/recommend`, `/feedback`
- [x] Async load testing benchmark (`loadtest.py`) reporting p50, p95, p99 latencies
- [x] Dockerfile & docker-compose.yml
- [x] API unit tests with TestClient (`tests/test_api.py`) passing:
  - `test_health_endpoint` PASSED
  - `test_recommend_endpoint_cold_start` PASSED
  - `test_recommend_with_genre_filter` PASSED
  - `test_feedback_endpoint` PASSED

### Phase 7: Reporting, Ablations & Documentation
- [x] Automated end-to-end pipeline run (`scripts/run_pipeline.py`)
- [x] Publication figures generated in `reports/figures/`
- [x] Ablation studies (logQ, history features, K sweep, two-tower score feature, IPS clipping)
- [x] Auto-generated `reports/RESULTS.md` with real execution numbers and honest limitations
- [x] Technical blog outline (`docs/BLOG_OUTLINE.md`) with 3 strong interview talking points
- [x] Comprehensive `README.md` with Mermaid diagram, quickstart, and reproduction guide

---

## Final Verification & Results

### Pytest Execution Summary
```
======================== 20 passed in 68.19s (0:01:08) =========================
```

### Empirical Offline Ranking Comparison (from `reports/RESULTS.md`)
| Model | NDCG@10 (95% CI) | Recall@10 | Precision@10 | MAP@10 | MRR | Catalog Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Popularity** | 0.0133 `[0.0104, 0.0162]` | 0.0200 | 0.0142 | 0.0034 | 0.0216 | 3.3% |
| **Item-KNN** | 0.0140 `[0.0107, 0.0172]` | 0.0202 | 0.0143 | 0.0038 | 0.0247 | 4.4% |
| **BPR-MF** | 0.0137 `[0.0105, 0.0165]` | 0.0200 | 0.0142 | 0.0035 | 0.0232 | 4.5% |
| **Two-Tower (Stage 1)** | 0.0130 `[0.0101, 0.0157]` | 0.0194 | 0.0142 | 0.0032 | 0.0210 | 4.3% |
| **Two-Tower + LightGBM** | **0.0263** `[0.0221, 0.0316]` | **0.0300** | **0.0208** | **0.0095** | **0.0585** | **8.1%** |
| **Two-Tower + DCN-v2 (Stretch)** | 0.0224 `[0.0181, 0.0269]` | 0.0286 | 0.0187 | 0.0076 | 0.0459 | 8.2% |

*Two-Tower + LightGBM statistically significantly outperforms all 3 baselines and Stage 1 retrieval with $p < 0.0001$ on paired tests.*

### Counterfactual Policy Value Comparison
- **True Policy Value $V^*(\text{Popularity})$**: `0.0736`
- **True Policy Value $V^*(\text{BPR-MF})$**: `0.2564`
- **True Policy Value $V^*(\text{Two-Tower + LightGBM})$**: **`0.6674`** (9x higher true utility than popularity under counterfactual exploration!)

### Serving Latency Benchmark (CPU-only)
- **p50 Latency**: `0.78 ms`
- **p95 Latency**: `0.87 ms`
- **p99 Latency**: `0.98 ms` (well under the 100 ms target)
- **Throughput**: `1260.6 req/s`

---

## Remaining Engineering Limitations
1. **Full Scale Scaling**: MovieLens-25M subsampling is configured in `configs/full.yaml` to ensure zero swap paging on 16 GB RAM.
2. **Slate Position Bias**: Single-item bandit evaluation should be extended to position-dependent cascade slate models in future iterations.
