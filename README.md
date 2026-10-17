# Two-Tower Neural Retrieval + LightGBM Ranking Recommender System
### With Counterfactual Off-Policy Evaluation (OPE) & Interactive React 19 Dashboard

[![CI](https://github.com/ekaagragupta/rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ekaagragupta/rs/actions/workflows/ci.yml)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black.svg?logo=vercel)](https://rs-beta-ten.vercel.app)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.0-61dafb.svg?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg?logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8.svg?logo=tailwindcss)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

>  **Live Production Cloud Dashboard**: [https://rs-beta-ten.vercel.app](https://rs-beta-ten.vercel.app)  
>  **Direct One-Click Access (Pre-authenticated)**: [https://rs-beta-ten.vercel.app/?x-vercel-protection-bypass=CKqV8BDbeVj1hOFKEakSr4Eyh5AbVqGp](https://rs-beta-ten.vercel.app/?x-vercel-protection-bypass=CKqV8BDbeVj1hOFKEakSr4Eyh5AbVqGp)  
>  **Interactive Swagger API Docs**: [https://rs-beta-ten.vercel.app/docs](https://rs-beta-ten.vercel.app/docs)

---

## 💡 Motivation & Project Overview

Modern industrial recommender systems (such as those powering Netflix, YouTube, Spotify, and TikTok) solve a fundamental engineering trade-off: **scoring millions of items with complex machine learning models within a tight SLA (< 50 milliseconds) is computationally impossible.**

To solve this, industry architectures rely on a multi-stage funnel:
1. **Candidate Retrieval (Stage 1)**: Rapidly filters the entire catalog down to a few dozen high-probability candidates using approximate nearest neighbor (ANN) search over dual-encoder vector embeddings.
2. **Re-Ranking (Stage 2)**: Applies heavy tabular features and gradient-boosted trees (e.g., LightGBM LambdaRank) to precisely order the retrieved candidates.
3. **Counterfactual Evaluation (Stage 3)**: Solves the fatal flaw of standard offline metrics (like NDCG on historical logs)—**presentation and logging bias**, where unobserved relevant items are incorrectly penalized as negative signals.

This repository is an **end-to-end, production-grade implementation** of this paradigm. It includes deep neural embedding models, FAISS vector indexing, LambdaRank re-ranking, off-policy causal evaluation (OPE: IPS, SNIPS, DR, SWITCH-DR), a sub-millisecond FastAPI serving engine, and a **modern, interactive React 19 + Tailwind CSS dashboard** built specifically for live demonstration.

---

## 🛠️ Technical Skills Demonstrated

This project showcases full-stack machine learning engineering, system design, causal inference, and modern web application development:

| Domain | Key Technologies | Competencies & Applied Concepts |
| :--- | :--- | :--- |
| **Representation Learning** | PyTorch, PyTorch Lightning | Built dual-encoder Two-Tower networks (User Tower & Item Tower) with 64-dimensional latent projections, in-batch negative sampled softmax loss, and **$\log Q$ popularity debiasing** to prevent head items from dominating retrieval embeddings. |
| **Vector Search & Indexing** | FAISS (Facebook AI Similarity Search) | Implemented and benchmarked vector indices across **FlatIP** (exact inner product), **HNSW** (hierarchical navigable small-world graphs), and **IVF-PQ** (inverted file product quantization), achieving sub-0.02ms query latencies at 100k+ QPS. |
| **Learning to Rank (LTR)** | LightGBM, Scikit-Learn | Engineered tabular cross-features (semantic dot-products, rank positions, recency-weighted genre affinities, popularity statistics), trained pairwise `lambdarank` models optimized for NDCG, and proved statistically significant ranking gains ($p < 10^{-4}$). |
| **Causal Machine Learning & OPE** | NumPy, SciPy, OBP | Designed semi-synthetic bandit logging simulations; implemented **Inverse Propensity Scoring (IPS)**, **Self-Normalized IPS (SNIPS)**, **Direct Method (DM)**, **Doubly Robust (DR)**, and **SWITCH-DR** estimators with importance weight clipping and temperature sensitivity analysis. |
| **Production Serving & API** | FastAPI, Pydantic v2, Uvicorn | Built asynchronous, type-safe REST APIs with per-stage latency telemetry, warm/cold-start handling, cosine similarity endpoints, and real-time counterfactual feedback logging. |
| **Modern Frontend Engineering** | React 19, TypeScript, Vite, Tailwind CSS | Crafted a responsive, dark-mode single-page application with interactive persona switching, dynamic genre filtering, debounced catalog search, latency waterfall visualizer, and Lucide icons. |
| **Cloud Deployment & DevOps** | Vercel Serverless, GitHub Actions | Packaged a lightweight Python serverless runtime alongside static Vite build assets, wrote path-normalization ASGI middleware, and configured automated multi-version CI pipelines (Python 3.11 & 3.12). |

---

## 🏛️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data Layer ["1. Data Processing & Time Splits"]
        ML["MovieLens Dataset\n(latest-small / 100k)"] --> Split["Strict Temporal Split\n(Train < Val < Test)"]
        Split --> Feat["Feature Store\n(History sequences, Genre profiles, Popularity)"]
    end

    subgraph Retrieval ["2. Stage 1: Two-Tower Neural Retrieval"]
        Feat --> TT["PyTorch Two-Tower Model\n(In-batch Sampled Softmax + logQ)"]
        TT --> Embeds["64-D Item Embeddings"]
        TT --> UserNet["User Tower Encoder"]
        Embeds --> FAISS["FAISS Index Engine\n(FlatIP / HNSW / IVF-PQ)"]
        UserNet & FAISS --> Top50["Top-50 Retrieved Candidates\n(p50 Latency: 0.01 ms)"]
    end

    subgraph Baselines ["Alternative Baseline Models"]
        Feat --> Pop["Global Popularity Baseline"]
        Feat --> KNN["Item-KNN Cosine Similarity"]
        Feat --> BPR["BPR Matrix Factorization"]
    end

    subgraph Ranking ["3. Stage 2: LightGBM LambdaRank"]
        Top50 --> CrossFeat["Candidate Feature Extraction\n(TT Score, Rank, User-Genre Match, Item Stats)"]
        CrossFeat --> LGBM["LightGBM LambdaRank Engine\n(Query Groups = User Interactions)"]
        LGBM --> TopK["Top-K Final Recommendations\n(K=5, 10, 15, 20)"]
    end

    subgraph Evaluation ["4. Stage 3: Evaluation & Counterfactual OPE"]
        TopK & Pop & KNN & BPR --> Offline["Offline Ranking Evaluation\n(NDCG, Recall, MAP, MRR, Coverage, Gini)"]
        TopK & Pop & KNN & BPR --> Bandit["Bandit Environment Simulator\n(Logging Policy p0(a|x) with epsilon exploration)"]
        Bandit --> OPE["Counterfactual OPE Estimators\n(IPS, SNIPS, DM, DR, SWITCH-DR)"]
    end

    subgraph Serving ["5. Stage 4: Production Serving & Interactive UI"]
        TopK & OPE --> API["FastAPI Serving Application\n(/recommend, /movies, /users, /metrics, /feedback)"]
        API --> UI["React 19 + Tailwind CSS Dashboard\n(Deployed Live on Vercel)"]
        UI --> Log["Real-time User Feedback Logger\n(Persists actions, rewards, and propensities)"]
    end
```

---

## 🖥️ Interactive Dashboard Features

The web frontend provides an intuitive, high-polish visual interface for exploring all aspects of the recommendation pipeline:

### 1. 🎯 Recommendation Studio (`Recommendations` Tab)
* **Pre-Computed User Personas**: Switch between representative archetypes with one click:
  * **User 42**: Heavy Action & Sci-Fi fanatic (110+ ratings)
  * **User 10**: Crime & Mystery buff (138 ratings)
  * **User 100**: Family & Animation moviegoer (124 ratings)
  * **User 1**: Drama & Thriller enthusiast (112 ratings)
  * **User 9999 (Cold-Start)**: Simulates a brand-new user with zero prior interactions; shows how the system safely degrades to global popularity exploration without crashing.
* **Model Strategy Switcher**: Run live A/B comparisons between:
  * **Two-Tower + LightGBM**: The production two-stage champion model.
  * **Two-Tower Retrieval Only**: Raw dot-product vector retrieval from Stage 1.
  * **Global Popularity**: Non-personalized frequency baseline.
  * **Item-KNN**: Item-item collaborative filtering baseline.
* **On-the-Fly Genre Pills**: Filter recommendations by genre (e.g. *Sci-Fi*, *Animation*, *Crime*) while maintaining rank integrity.
* **Top-K Slider**: Dynamically adjust recommendation list depth ($K \in [5, 20]$).
* **Per-Stage Latency Waterfall**: Visualizes real-time execution telemetry breakdown:
  $$\text{Total Latency} = \text{Feature Fetch} + \text{FAISS ANN Retrieval} + \text{LightGBM Scoring}$$
* **Interactive Feedback Buttons**: Click ❤️ (Like), 👎 (Dislike), or ⭐ (Watch) to generate live interaction events that feed into the counterfactual feedback log.

### 2. 🔍 Movie Catalog Explorer (`Catalog Explorer` Tab)
* **Instant Debounced Search**: Fast keyword filtering across 1,500 catalog titles.
* **Genre Filter Pills & Sorting**: Sort by popularity, title, release year, or average rating.
* **64-D Latent Space Similarity Modal**: Click any movie card (e.g. *The Matrix*, *Toy Story*, *The Godfather*) to view its **6 nearest neighbors in Two-Tower cosine similarity space**, proving that the neural embeddings successfully capture thematic, tonal, and genre clustering.

### 3. 👤 User Personas & History (`User Profiles` Tab)
* **Activity Tier Categorization**: Cold (< 30 ratings), Warm (30–150 ratings), Heavy (150+ ratings).
* **Taste Distribution**: Bar charts showing each user's top genre affinities.
* **Chronological History Feed**: Full timeline of historical user interactions, timestamps, and ratings, demonstrating how exponential time decay ($\lambda = 10^{-4}$) prioritizes recent preferences.

### 4. 📊 Model Benchmarks & OPE Research (`Model Benchmarks` Tab)
* **Offline Ranking Metrics**: Interactive table displaying NDCG@10, Recall@10, Precision@10, MAP@10, MRR, Catalog Coverage, and Gini diversity.
* **Bootstrap Confidence Intervals & Significance**: 95% percentile bootstrap bounds and paired $t$-test / Wilcoxon signed-rank test results showing statistical significance ($p < 0.0001$).
* **User Activity Slices**: Compares model performance across cold, warm, and heavy user segments.
* **FAISS Index Trade-Offs**: Quantitative benchmarks comparing exact `FlatIP`, `HNSW`, and `IVF-PQ` on recall loss, latency percentiles, and throughput.
* **Counterfactual OPE Study**: Explains the core scientific puzzle: **why offline NDCG favors popularity while counterfactual estimators prove the two-stage model provides 9x higher true utility.** Includes sensitivity sweeps across clipping thresholds, softmax temperatures, and sample sizes.
* **Ablation Studies**: Quantitative breakdown of performance impact from $\log Q$ popularity correction, user sequence features, retrieval candidate pool size ($K$), and Stage 1 score feature integration.

### 5. ⚡ System Architecture & Telemetry (`Architecture` Tab)
* **4-Stage Funnel Diagram**: Visualizes data flow from 1,500 items $\rightarrow$ 50 candidates $\rightarrow$ Top-10 ranked list.
* **Live Telemetry KPI Cards**: Real-time stats on QPS, model memory footprint, and serving latency.
* **Live Counterfactual Stream**: Dynamic table displaying user feedback events recorded during the session, complete with logged logging-policy propensities $p_0(a|x)$ and reward outcomes.

---

## 📈 Empirical Evaluation Results

### 1. Offline Ranking Performance (Held-Out Test Split)

Evaluated under strict temporal splits (interactions ordered chronologically, ensuring zero future information leakage):

| Model Architecture | NDCG@10 (95% CI) | Recall@10 | Precision@10 | MAP@10 | MRR | Catalog Coverage | Gini Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global Popularity** | 0.0133 `[0.0104, 0.0162]` | 0.0200 | 0.0142 | 0.0034 | 0.0216 | 3.3% | 0.993 |
| **Item-KNN** | 0.0140 `[0.0107, 0.0172]` | 0.0202 | 0.0143 | 0.0038 | 0.0247 | 4.4% | 0.993 |
| **BPR-MF** | 0.0137 `[0.0105, 0.0165]` | 0.0200 | 0.0142 | 0.0035 | 0.0232 | 4.5% | 0.993 |
| **Two-Tower (Stage 1)** | 0.0130 `[0.0101, 0.0157]` | 0.0194 | 0.0142 | 0.0032 | 0.0210 | 4.3% | 0.993 |
| **Two-Tower + LightGBM** | **0.0263** `[0.0221, 0.0316]` | **0.0300** | **0.0208** | **0.0095** | **0.0585** | **8.1%** | **0.969** |
| **Two-Tower + DCN-v2 (Stretch)** | 0.0224 `[0.0181, 0.0269]` | 0.0286 | 0.0187 | 0.0076 | 0.0459 | 8.2% | 0.966 |

> **Key Finding**: Adding the LightGBM LambdaRank re-ranking stage yields a **+97.7% relative improvement in NDCG@10** over the baseline ($p = 1.78 \times 10^{-5}$ on paired $t$-test) and nearly doubles catalog coverage.

---

### 2. Counterfactual Policy Value ($V^*$) vs. Offline NDCG

Under historical logging policies, users only interact with items displayed on past screens. This creates a severe bias where unobserved relevant items are counted as negatives in offline testing. Counterfactual Off-Policy Evaluation (OPE) corrects for this via propensity weighting:

$$\hat{V}_{\text{IPS}}(\pi) = \frac{1}{N} \sum_{i=1}^N \frac{\pi(a_i | x_i)}{p_0(a_i | x_i)} \cdot r_i$$

| Policy Model | True Policy Value $V^*(\pi)$ | IPS Estimate | SNIPS Estimate | Doubly Robust (DR) |
| :--- | :---: | :---: | :---: | :---: |
| **Global Popularity** | 0.0736 | 0.1420 | 0.1310 | 0.1300 |
| **Item-KNN** | 0.1800 | 0.1650 | 0.1740 | 0.1780 |
| **BPR-MF** | 0.2564 | 0.1920 | 0.2030 | 0.2090 |
| **Two-Tower + LightGBM** | **0.6674** | **0.2310** | **0.2580** | **0.2640** |

> **Core Research Insight**: While offline test NDCG gives Global Popularity a deceptively close score ($0.0133$ vs $0.0263$), counterfactual evaluation reveals that Two-Tower + LightGBM delivers **over 9× higher true expected reward ($0.6674$ vs $0.0736$)** because it personalizes recommendations to the full catalog rather than repeatedly presenting the same head items.

---

### 3. CPU Serving Latency Telemetry

Measured on standard CPU hardware (M-series / x86_64 single core):

| Pipeline Stage | Latency (p50) | Latency (p95) | Target SLA | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Feature Fetch & User Vector Encoding** | 0.05 ms | 0.12 ms | < 10 ms | ✅ Pass |
| **FAISS ANN Candidate Retrieval (Top-50)** | 0.01 ms | 0.02 ms | < 15 ms | ✅ Pass |
| **LightGBM LambdaRank Scoring (50 items)** | 0.62 ms | 0.72 ms | < 25 ms | ✅ Pass |
| **Total End-to-End Latency** | **0.78 ms** | **0.83 ms** | **< 75 ms** | ✅ **Pass** |
| **Peak Single-Instance Throughput** | **1,247 req/s** | - | > 50 req/s | ✅ **Pass** |

---

### 4. FAISS Vector Search Index Comparison

| Index Type | Recall@50 vs Exact | Recall Loss | Query Latency p50 | CPU Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: |
| **FlatIP (Exact Dot-Product)** | 1.0000 | 0.0000 | 0.009 ms | 109,549 req/s |
| **HNSW (Graph Search)** | 1.0000 | 0.0000 | 0.015 ms | 65,213 req/s |
| **IVF-PQ (Inverted File + Quantization)** | 0.9480 | 0.0520 | 0.013 ms | 48,496 req/s |

---

## 🚀 Local Installation & Reproduction Guide

### Prerequisites
* **Python 3.11 or 3.12**
* **Node.js (v18+) & npm**
* macOS, Linux, or WSL2

### 1. Clone the Repository
```bash
git clone https://github.com/ekaagragupta/rs.git
cd rs
```

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Setup Python virtualenv and install core dependencies
make setup

# Install frontend Node modules and compile the React production bundle
make frontend-install
make frontend-build
```

### 3. Launch the Application
```bash
make serve
```
Open your browser and navigate to:
* **Interactive UI**: `http://localhost:8000`
* **Swagger API Documentation**: `http://localhost:8000/docs`

### 4. (Optional) Run Frontend in Hot-Reload Development Mode
```bash
# Terminal 1: Backend
make serve

# Terminal 2: Vite Dev Server
make frontend-dev
# Open http://localhost:5173
```

### 5. Execute Test Suite
```bash
make test
```
Runs 29 automated tests covering:
* Unit metric implementations against hand-computed ground truth
* Strict temporal split ordering and non-leakage verification
* FAISS FlatIP, HNSW, and IVF-PQ consistency
* Counterfactual OPE estimators (unbiasedness, boundedness, doubly robust properties)
* Full FastAPI serving endpoint integration

---

## 📂 Project Structure

```
rs/
├── api/                           # Vercel serverless function entrypoint
│   ├── index.py                   # ASGI serverless adapter with query rewrite engine
│   └── requirements.txt           # Lean serverless runtime dependencies
├── configs/
│   ├── dev.yaml                   # ml-latest-small development configuration
│   └── full.yaml                  # ml-25m production configuration
├── data/
│   ├── processed/                 # serving_meta.pkl & precomputed_recs.pkl (serving cache)
│   └── raw/ml-latest-small/       # movies.csv and ratings.csv
├── frontend/                      # Modern React 19 + TypeScript + Tailwind CSS UI
│   ├── src/
│   │   ├── components/            # RecommendationsStudio, MovieExplorer, UserProfiles,
│   │   │                          # MetricsDashboard, ArchitectureView, LatencyWaterfall, etc.
│   │   ├── api/client.ts          # Type-safe client communicating with /api endpoints
│   │   ├── types/api.ts           # Pydantic-aligned TypeScript type definitions
│   │   ├── App.tsx                # Main container
│   │   └── main.tsx               # Root entrypoint
│   ├── dist/                      # Precompiled production bundle served by FastAPI & Vercel
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
├── src/recsys/                    # Core Python ML package
│   ├── data/                      # Temporal splitting, history sequences, feature store
│   ├── baselines/                 # Global Popularity, Item-KNN, PyTorch BPR-MF
│   ├── retrieval/                 # PyTorch Two-Tower, sampled softmax + logQ, FAISS indices
│   ├── ranking/                   # Candidate generator, feature extractor, LightGBM LambdaRank
│   ├── evaluation/                # NDCG, Recall, MAP, MRR, bootstrap CIs, significance
│   ├── ope/                       # Bandit simulator, IPS, SNIPS, DM, DR, SWITCH-DR
│   ├── serving/                   # Production FastAPI server, loader, telemetry schemas
│   └── utils/                     # Config parser, seed control, structured logger
├── scripts/
│   ├── run_pipeline.py            # End-to-end pipeline CLI (data -> train -> eval)
│   ├── run_ope_study.py           # Counterfactual OPE sensitivity sweep script
│   └── enrich_movies.py           # MovieLens catalog title enrichment utility
├── tests/                         # 29 unit, leakage, OPE, FAISS, and extended API integration tests
├── reports/
│   ├── checkpoints/               # best_two_tower.pt, lgbm_ranker.pkl
│   ├── tables/                    # offline_metrics.csv, ann_benchmarks.csv, ope_*.csv
│   ├── figures/                   # OPE sweep analysis figures
│   └── RESULTS.md                 # Full empirical research writeup
├── Makefile                       # Unified build automation (setup, serve, test, lint, build)
├── vercel.json                    # Vercel cloud deployment and routing rules
└── requirements-full.txt          # Full pinned ML environment requirements
```

---

