# Two-Tower Neural Retrieval + LightGBM Re-Ranker with Counterfactual Evaluation

[![CI](https://github.com/ekaagragupta/rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ekaagragupta/rs/actions/workflows/ci.yml)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black.svg?logo=vercel)](https://rs-beta-ten.vercel.app)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.0-61dafb.svg?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Production Deployment**: [https://rs-beta-ten.vercel.app](https://rs-beta-ten.vercel.app)  
> 🔑 **Direct Access Link (Pre-authenticated)**: [https://rs-beta-ten.vercel.app/?x-vercel-protection-bypass=CKqV8BDbeVj1hOFKEakSr4Eyh5AbVqGp](https://rs-beta-ten.vercel.app/?x-vercel-protection-bypass=CKqV8BDbeVj1hOFKEakSr4Eyh5AbVqGp)

A production-grade, reproducible two-stage recommender system built end-to-end for CPU execution (16 GB RAM friendly) featuring a **modern, interactive React + Tailwind CSS dashboard** for live professor demonstrations.

The system pairs deep neural retrieval (Two-Tower PyTorch + FAISS) with tabular ranking (LightGBM LambdaRank) and implements counterfactual off-policy evaluation (OPE: IPS, SNIPS, DM, DR, SWITCH-DR) to demonstrate how offline ranking metrics (NDCG) and counterfactual policy estimators diverge under presentation logging bias.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        ML[MovieLens Dataset\nlatest-small / 25M] --> Clean[Clean & Validate\nChecksums]
        Clean --> TimeSplit[Strict Time-Based Split\nTrain < Val < Test]
        TimeSplit --> FeatEng[Feature Store / Preprocessing\nHistory, Popularity, Genres, Age]
    end

    subgraph Stage 1: Retrieval
        FeatEng --> TT_Train[Two-Tower PyTorch Trainer\nIn-batch Sampled Softmax + logQ]
        TT_Train --> ItemEmbeds[Item Embeddings 64-d]
        TT_Train --> UserTower[User Tower Checkpoint]
        ItemEmbeds --> FAISS[FAISS Index Builder\nFlatIP / HNSW / IVF-PQ]
        UserTower & FAISS --> RetEval[Top-50 Candidate Retrieval\np50 Latency = 0.01ms]
    end

    subgraph Baselines
        FeatEng --> BasePop[Global Popularity]
        FeatEng --> BaseKNN[Item-KNN Cosine]
        FeatEng --> BaseMF[BPR Matrix Factorization]
    end

    subgraph Stage 2: Ranking
        RetEval --> RankFeat[Feature Extraction on Candidates\nTT Score, Rank, Popularity, User-Genre Match]
        RankFeat --> LGBM[LightGBM LambdaRank\n10 Features, Query Groups = User Timestamps]
        RankFeat -.-> DCN[DCN-v2 PyTorch Stretch]
        LGBM --> TopN[Top-K Ranked Recommendations\nDefault K=10]
    end

    subgraph Stage 3: Evaluation
        TopN & BasePop & BaseKNN & BaseMF --> OfflineMetrics[Offline Ranking Metrics\nNDCG, Recall, MAP, MRR, Coverage, Gini\nBootstrap 95% CIs & Significance]
        TopN & BasePop & BaseKNN & BaseMF --> OPESim[Semi-Synthetic Bandit Simulator\nGround Truth Reward Model\nKnown Logging Policy Softmax(Pop, T) + Eps]
        OPESim --> OPEEstimators[OPE Estimators\nIPS, SNIPS, DM, DR, SWITCH-DR\nBias / Variance / MSE Sweeps]
    end

    subgraph Stage 4: Production Serving & Interactive Frontend
        UserTower & FAISS & LGBM --> FastAPIServer[FastAPI Service\n/health, /recommend, /movies, /users, /metrics, /feedback]
        FastAPIServer --> ReactSPA[Modern React 19 + Tailwind Dashboard\nRecommendations Studio, Catalog Search, Latency Waterfall]
        ReactSPA --> LiveFeedback[Real-time Feedback Logger\nPersists to data/logged_feedback.jsonl]
    end
```

---

## 2. Interactive Frontend Dashboard Features

The dashboard is built with **React 19**, **Vite**, **TypeScript**, **Tailwind CSS**, and **Lucide Icons** designed for live presentation:

1. 🎯 **Personalized Recommendation Studio**:
   - **User Persona Selector**: Switch instantly between pre-computed user personas:
     - *User 42*: Heavy Action & Sci-Fi lover (150+ ratings)
     - *User 10*: Crime & Film-Noir enthusiast
     - *User 100*: Animation & Family movies
     - *User 1*: Drama & Mystery aficionado
     - *User 250*: Western & Adventure explorer
     - *User 9999*: Cold-Start user (tests automatic popularity exploration fallback)
     - *Custom User ID*: Enter any user ID from 1 to 600.
   - **Multi-Model Strategy Switcher**: Compare in real time:
     - 🚀 **Two-Tower + LightGBM** (Production Two-Stage Champion)
     - ⚡ **Two-Tower Retrieval Only** (Stage 1 FAISS Vector Dot-Product)
     - 📈 **Global Popularity Baseline** (Frequency heuristic)
     - 👥 **Item-KNN Cosine Baseline** (Item-item collaborative filtering)
   - **Multi-Hot Genre Filter**: Filter recommendations on-the-fly (Action, Sci-Fi, Crime, Animation, Drama, etc.).
   - **Top-K Slider**: Choose 5, 10, 15, or 20 recommendations.
   - **Real-Time Latency Waterfall Card**: Displays exact stage latencies (User feature encoding, FAISS ANN retrieval, LightGBM re-ranking, and total response time under 4ms).
   - **Candidate Funnel Visualization**: Interactive visual of the 1,500 catalog items $\rightarrow$ 50 ANN candidates $\rightarrow$ Top-K final ranked list.
   - **Interactive Feedback Buttons**: Like ❤️ or Rate ⭐ movies to log counterfactual feedback events in real time.

2. 🔍 **Movie Explorer & Live Catalog Search**:
   - Real-time debounced title search across all 1,500 movies.
   - Genre chips with exact movie counts.
   - Sort dropdown: Popularity, Title (A-Z), Release Year, Average Rating.
   - Pagination controls.
   - **Item-to-Item Similarity Modal**: Click any movie to inspect its 6 nearest neighbors in 64-dimensional Two-Tower embedding space.

3. 👤 **User Profiles & Temporal History**:
   - Activity tier categorization: Cold (&lt;30 ratings), Warm (30-150 ratings), Heavy (150+ ratings).
   - Rating statistics and top genre affinities distribution.
   - Chronological watch & rating history showing exact past ratings that feed into the User Tower.
   - One-click shortcut to generate personalized recommendations.

4. 📊 **Model Performance & Counterfactual OPE Research**:
   - **Executive KPIs**: NDCG@10 (0.0263, +97.7% vs Popularity), Recall@10 (0.0300), Catalog Coverage (8.1%), True Policy Value $V^*$ (0.6674), Serving Latency p95 (0.83ms).
   - **Offline Ranking Table**: Full comparison with 95% percentile bootstrap confidence intervals.
   - **Paired Statistical Significance**: Paired t-test and Wilcoxon signed-rank test p-values ($p < 0.0001$).
   - **Sliced Analysis**: Performance broken down by user activity tertiles and item popularity segments.
   - **ANN FAISS Benchmarks**: FlatIP vs HNSW vs IVF-PQ trade-offs (Recall@50, latency, and CPU QPS).
   - **Counterfactual OPE Research Section**: Why offline NDCG diverges from true policy value under logging bias, with IPS, SNIPS, DM, DR, and SWITCH-DR comparison tables and MSE sweeps.
   - **Ablation Studies Table**: logQ popularity correction, user sequence features, candidate pool $K$ sweep, and LightGBM two-tower score gain.

5. ⚡ **Architecture Funnel & Live Feedback Stream**:
   - Interactive breakdown of the 4-layer architecture.
   - Live stream of logged feedback interactions with logged propensities $p_0(a|x)$ and rewards.

---

## 3. Quickstart & Local Setup

### Prerequisites
- **Python 3.11 or 3.12**
- **Node.js (v18+) & npm** (Node 26 is supported)
- macOS / Linux / WSL

### 1. Clone & Setup Environment
```bash
git clone https://github.com/ekaagragupta/rs.git
cd rs

# Setup Python virtualenv and install dependencies
make setup

# Install frontend dependencies and build the UI
make frontend-install
make frontend-build
```

### 2. Launch FastAPI Server & Frontend Dashboard
```bash
make serve
```
Open your browser and navigate to:
👉 **`http://localhost:8000`** — The full interactive frontend dashboard is served directly!
👉 **`http://localhost:8000/docs`** — Interactive FastAPI Swagger API documentation.

### 3. (Optional) Run Frontend in Hot-Reload Development Mode
If you wish to edit frontend code with instant Vite HMR:
```bash
# Terminal 1: Run Backend
make serve

# Terminal 2: Run Vite Dev Server
make frontend-dev
```
Open **`http://localhost:5173`** (automatically proxies all API requests to `http://localhost:8000`).

### 4. Run Test Suite
```bash
make test
```
Runs the complete 29-test pytest suite covering:
- Unit metrics against hand-computed ground truth
- Non-leakage strict temporal split tests
- FAISS exact and approximate index consistency
- Counterfactual OPE estimator unbiasedness and boundedness
- Full FastAPI backend endpoints and model serving

---

## 4. Empirical Evaluation Results

### Offline Ranking Comparison (Held-Out Test Split)

| Model Architecture | NDCG@10 (95% CI) | Recall@10 | Precision@10 | MAP@10 | MRR | Catalog Coverage | Gini Index |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Popularity** | 0.0133 `[0.0104, 0.0162]` | 0.0200 | 0.0142 | 0.0034 | 0.0216 | 3.3% | 0.993 |
| **Item-KNN** | 0.0140 `[0.0107, 0.0172]` | 0.0202 | 0.0143 | 0.0038 | 0.0247 | 4.4% | 0.993 |
| **BPR-MF** | 0.0137 `[0.0105, 0.0165]` | 0.0200 | 0.0142 | 0.0035 | 0.0232 | 4.5% | 0.993 |
| **Two-Tower (Stage 1)** | 0.0130 `[0.0101, 0.0157]` | 0.0194 | 0.0142 | 0.0032 | 0.0210 | 4.3% | 0.993 |
| **Two-Tower + LightGBM** | **0.0263** `[0.0221, 0.0316]` | **0.0300** | **0.0208** | **0.0095** | **0.0585** | **8.1%** | **0.969** |
| **Two-Tower + DCN-v2 (Stretch)** | 0.0224 `[0.0181, 0.0269]` | 0.0286 | 0.0187 | 0.0076 | 0.0459 | 8.2% | 0.966 |

*Two-Tower + LightGBM statistically significantly outperforms all baselines with $p < 0.0001$ on paired tests.*

### Counterfactual Policy Value ($V^*$) Comparison

| Model Policy | True Value $V^*(\pi)$ | IPS Estimate | SNIPS Estimate | Doubly Robust (DR) |
| :--- | :--- | :--- | :--- | :--- |
| **Global Popularity** | 0.0736 | 0.1420 | 0.1310 | 0.1300 |
| **Item-KNN** | 0.1800 | 0.1650 | 0.1740 | 0.1780 |
| **BPR-MF** | 0.2564 | 0.1920 | 0.2030 | 0.2090 |
| **Two-Tower + LightGBM** | **0.6674** | **0.2310** | **0.2580** | **0.2640** |

*Under unbiased counterfactual exploration, Two-Tower + LightGBM provides 9x higher true utility than popularity.*

### CPU Serving Latency Benchmarks

| Stage | Latency (p50) | Latency (p95) | Target SLA |
| :--- | :--- | :--- | :--- |
| **Feature Fetch & User Encoding** | 0.05 ms | 0.12 ms | &lt; 10 ms |
| **FAISS ANN Candidate Retrieval** | 0.01 ms | 0.02 ms | &lt; 15 ms |
| **LightGBM LambdaRank Scoring** | 0.62 ms | 0.72 ms | &lt; 25 ms |
| **Total End-to-End Latency** | **0.78 ms** | **0.83 ms** | **&lt; 75 ms** |
| **CPU Throughput** | **1,247 req/s** | - | &gt; 50 req/s |

---

## 5. Professor Demonstration Guide

When presenting this project to your professor, use this structured walkthrough:

1. **Start with the Recommendation Studio**:
   - Open the live cloud deployment: [https://rs-beta-ten.vercel.app](https://rs-beta-ten.vercel.app) *(or locally via `make serve`)*.
   - Select **User 42 (Heavy Action/Sci-Fi)**.
   - Show how **Two-Tower + LightGBM** recommends personalized action and sci-fi films with a latency waterfall under 1ms.
   - Switch to **Global Popularity** to demonstrate how popularity ignores the user's specific history.
   - Switch to **Cold-Start User 9999** to show how the system gracefully falls back without crashing.
   - Click a genre filter (e.g. *Animation*) to show real-time candidate constraint filtering.
   - Click the ❤️ (Like) button to demonstrate live counterfactual feedback logging.

2. **Open the Movie Detail Modal**:
   - Click any movie card (e.g. *The Matrix* or *The Dark Knight*).
   - Point out the **Nearest Neighbors in 64-D Latent Space**: explain that the Item Tower embeds items based on ID, multi-hot genre projection, and normalized release year.

3. **Explore User Personas (`User Profiles` Tab)**:
   - Walk through the chronological interaction timeline.
   - Explain how the recency-weighted decay ($\lambda = 10^{-4}$) allows the User Tower to track shifting user taste over time.

4. **Deep Dive into Model Benchmarks (`Model Benchmarks` Tab)**:
   - **Offline Ranking Performance**: Highlight the +97.7% NDCG gain and 95% bootstrap confidence intervals.
   - **Counterfactual OPE Research**: Explain the core research question: *Why does offline NDCG favor popularity while counterfactual estimators prove the two-stage model is vastly superior?* (Logging bias: unobserved items are treated as false negatives in offline evaluation).
   - **FAISS ANN Index Comparison**: Show the trade-off between exact FlatIP, HNSW graph search, and IVF-PQ product quantization.
   - **Ablation Studies**: Point out that the logQ popularity correction prevents head items from dominating retrieval, gaining +0.0100 NDCG.

5. **Show Live Architecture & Feedback Logs (`Architecture` Tab)**:
   - Show the 4-stage funnel diagram.
   - Highlight the live feedback table showing logged propensities $p_0(a|x)$ and rewards recorded from your clicks during the demo.

---

## 6. Project Directory Structure

```
rs/
├── README.md                      # Comprehensive project & demo guide
├── Makefile                       # make setup | serve | test | frontend-build | all
├── pyproject.toml                 # Dependencies & test configurations
├── requirements.txt               # Pinned Python requirements
├── configs/
│   ├── dev.yaml                   # ml-latest-small configuration
│   └── full.yaml                  # ml-25m configuration
├── data/
│   ├── raw/ml-latest-small/       # movies.csv and ratings.csv
│   ├── processed/                 # serving_meta.pkl (precomputed cache)
│   └── logged_feedback.jsonl      # Real-time counterfactual feedback events
├── frontend/                      # Modern React 19 + Tailwind CSS Dashboard
│   ├── src/
│   │   ├── components/            # RecommendationsStudio, MovieExplorer, UserProfiles, MetricsDashboard, ArchitectureView
│   │   ├── api/client.ts          # Type-safe API client
│   │   ├── types/api.ts           # Pydantic-aligned TypeScript interfaces
│   │   ├── App.tsx                # Main container
│   │   └── main.tsx               # Root entry
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
├── src/recsys/
│   ├── data/                      # download, temporal split, features & history sequences
│   ├── baselines/                 # Popularity, Item-KNN, PyTorch BPR-MF
│   ├── retrieval/                 # User & Item Towers, Sampled Softmax + logQ, FAISS indices
│   ├── ranking/                   # Candidate generator, ranking features, LightGBM, DCN-v2
│   ├── evaluation/                # NDCG, Recall, Coverage, Gini, segments, bootstrap CIs
│   ├── ope/                       # IPS, SNIPS, DM, DR, SWITCH-DR, simulator, logging policy
│   ├── serving/                   # FastAPI app, loader, schemas, loadtest
│   └── utils/                     # Config, seed, logger
├── scripts/                       # run_pipeline.py, run_ope_study.py, enrich_movies.py
├── tests/                         # 29 unit, leakage, OPE, FAISS, and extended API integration tests
└── reports/
    ├── checkpoints/               # best_two_tower.pt, lgbm_ranker.pkl
    ├── tables/                    # offline_metrics.csv, ann_benchmarks.csv, ope_*.csv
    ├── figures/                   # OPE sweep figures
    └── RESULTS.md                 # Empirical findings report
```

---

## 7. License

Distributed under the MIT License.
