"""Counterfactual Off-Policy Evaluation study sweeping temperature, sample size, and clipping."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys.data.download import download_movielens
from recsys.data.features import compute_training_statistics, extract_movie_features
from recsys.data.split import time_based_split
from recsys.ope.logging_policy import PopularityLoggingPolicy
from recsys.ope.obp_adapter import run_open_bandit_evaluation
from recsys.ope.simulator import SemiSyntheticBanditSimulator
from recsys.utils.config import load_config, to_dict
from recsys.utils.logger import get_logger
from recsys.utils.seed import seed_everything

logger = get_logger("recsys.ope.study")


def run_ope_study(config_path: str = "configs/dev.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg_dict = to_dict(cfg)
    seed = cfg_dict.get("seed", 42)
    seed_everything(seed)

    raw_dir = Path(cfg_dict.get("raw_data_dir", "data/raw"))
    reports_dir = Path(cfg_dict.get("artifacts_dir", "reports"))
    fig_dir = reports_dir / "figures"
    table_dir = reports_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data & partition
    logger.info("Loading dataset for OPE study...")
    ratings_path, movies_path = download_movielens(
        dataset_name=cfg_dict.get("dataset_name", "ml-latest-small"), raw_dir=raw_dir
    )
    df_ratings = pd.read_csv(ratings_path)
    df_movies = pd.read_csv(movies_path)

    split_cfg = cfg_dict.get("split", {})
    train_df, _, test_df, metadata = time_based_split(
        df_ratings,
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    num_users = metadata["num_users"]
    num_items = metadata["num_items"]
    genre_matrix, year_array, _ = extract_movie_features(df_movies, metadata["item_to_idx"])
    stats = compute_training_statistics(train_df, genre_matrix, num_users, num_items)

    # 2. Initialize Semi-Synthetic Bandit Simulator
    simulator = SemiSyntheticBanditSimulator(
        test_df=test_df,
        item_train_counts=stats["item_counts"],
        reward_threshold=cfg_dict.get("ope", {}).get("reward_threshold", 4.0),
        candidate_pool_size=40,
        seed=seed,
    )

    # Define target evaluation policies (stochastic distribution over candidates)
    # Policy A: Popularity-skewed
    # Policy B: Collaborative Filtering (BPR-MF)
    # Policy C: Personalized Two-Tower + LightGBM
    policy_distributions: dict[str, dict[int, dict[int, float]]] = {}
    models_to_simulate = ["Global Popularity", "BPR-MF", "Two-Tower + LightGBM"]

    for model_name in models_to_simulate:
        policy_distributions[model_name] = {}
        for u in simulator.eval_users:
            pool = simulator.candidate_pool[u]
            pos_items = simulator.user_positives.get(u, set())
            scores = []
            for item in pool:
                if model_name == "Global Popularity":
                    score = np.log1p(stats["item_counts"][item])
                elif model_name == "BPR-MF":
                    # Preference for positive items with some noise
                    score = 2.0 if item in pos_items else 0.5
                else:  # Two-Tower + LightGBM
                    score = 4.0 if item in pos_items else 0.2
                scores.append(score)

            scores_arr = np.array(scores)
            exp_s = np.exp(scores_arr - np.max(scores_arr))
            probs = exp_s / np.sum(exp_s)
            policy_distributions[model_name][u] = {item: float(p) for item, p in zip(pool, probs)}

    # Compute EXACT Ground Truth Policy Values
    true_values: dict[str, float] = {}
    for name, p_dist in policy_distributions.items():
        tv = simulator.compute_true_policy_value(p_dist)
        true_values[name] = tv
        logger.info("Exact Ground Truth Policy Value V*(%s) = %.4f", name, tv)

    target_eval_policy = policy_distributions["Two-Tower + LightGBM"]
    target_true_val = true_values["Two-Tower + LightGBM"]

    # 3. Sweep A: Temperature Sweep
    logger.info("=== Running Sweep A: Logging Policy Temperature Sweep ===")
    temperatures = [0.5, 1.0, 1.5, 2.0, 3.0]
    n_reps = 15
    sample_size = 5000

    temp_results: dict[str, list[float]] = {
        "temperature": temperatures,
        "ips_mse": [],
        "snips_mse": [],
        "dm_mse": [],
        "dr_mse": [],
        "switch_dr_mse": [],
    }

    for temp in temperatures:
        log_pol = PopularityLoggingPolicy(stats["item_counts"], temperature=temp, epsilon_uniform=0.05)
        rep_estimates: dict[str, list[float]] = {"ips": [], "snips": [], "dm": [], "dr": [], "switch_dr": []}

        for _ in range(n_reps):
            bandit_data = simulator.generate_logged_bandit_data(log_pol, n_samples=sample_size)
            q_hat = simulator.fit_reward_model(bandit_data)
            est = simulator.evaluate_policy_ope(target_eval_policy, bandit_data, q_hat, clip_threshold=20.0)

            rep_estimates["ips"].append(est["v_ips"])
            rep_estimates["snips"].append(est["v_snips"])
            rep_estimates["dm"].append(est["v_dm"])
            rep_estimates["dr"].append(est["v_dr"])
            rep_estimates["switch_dr"].append(est["v_switch_dr"])

        for k in ["ips", "snips", "dm", "dr", "switch_dr"]:
            errors = np.array(rep_estimates[k]) - target_true_val
            mse = float(np.mean(errors ** 2))
            temp_results[f"{k}_mse"].append(mse)

    # 4. Sweep B: Sample Size Sweep
    logger.info("=== Running Sweep B: Logged Bandit Sample Size Sweep ===")
    sample_sizes = [1000, 2500, 5000, 10000]
    log_pol_fixed = PopularityLoggingPolicy(stats["item_counts"], temperature=1.5, epsilon_uniform=0.05)

    size_results: dict[str, list[float]] = {
        "sample_size": sample_sizes,
        "ips_var": [],
        "snips_var": [],
        "dr_var": [],
        "ips_bias": [],
        "snips_bias": [],
        "dr_bias": [],
    }

    for n_s in sample_sizes:
        rep_estimates = {"ips": [], "snips": [], "dr": []}
        for _ in range(n_reps):
            b_data = simulator.generate_logged_bandit_data(log_pol_fixed, n_samples=n_s)
            q_h = simulator.fit_reward_model(b_data)
            est = simulator.evaluate_policy_ope(target_eval_policy, b_data, q_h, clip_threshold=20.0)
            rep_estimates["ips"].append(est["v_ips"])
            rep_estimates["snips"].append(est["v_snips"])
            rep_estimates["dr"].append(est["v_dr"])

        for k in ["ips", "snips", "dr"]:
            vals = np.array(rep_estimates[k])
            size_results[f"{k}_var"].append(float(np.var(vals)))
            size_results[f"{k}_bias"].append(float(np.mean(vals) - target_true_val))

    # 5. Sweep C: Clipping Threshold Sweep (Ablation e)
    logger.info("=== Running Sweep C: Importance Weight Clipping Sweep ===")
    clipping_thresholds = [5.0, 10.0, 20.0, 50.0, 100.0]
    clip_results: dict[str, list[float]] = {
        "clip_threshold": clipping_thresholds,
        "bias": [],
        "variance": [],
        "mse": [],
    }
    b_data_fixed = simulator.generate_logged_bandit_data(log_pol_fixed, n_samples=5000)
    q_h_fixed = simulator.fit_reward_model(b_data_fixed)

    for clip in clipping_thresholds:
        clip_reps = []
        for _ in range(n_reps):
            b_d = simulator.generate_logged_bandit_data(log_pol_fixed, n_samples=5000)
            est = simulator.evaluate_policy_ope(target_eval_policy, b_d, q_h_fixed, clip_threshold=clip)
            clip_reps.append(est["v_clipped_ips"])

        c_vals = np.array(clip_reps)
        b = float(np.mean(c_vals) - target_true_val)
        v = float(np.var(c_vals))
        clip_results["bias"].append(b)
        clip_results["variance"].append(v)
        clip_results["mse"].append(b ** 2 + v)

    # 6. Real OBD Track
    logger.info("=== Running Real Open Bandit Dataset Track ===")
    obd_res = run_open_bandit_evaluation()

    # 7. Generate Publication Figures
    logger.info("Generating publication figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: MSE vs Temperature
    plt.figure(figsize=(7, 4.5))
    plt.plot(temperatures, temp_results["ips_mse"], "o-", label="Vanilla IPS", color="#d62728")
    plt.plot(temperatures, temp_results["snips_mse"], "s-", label="SNIPS", color="#ff7f0e")
    plt.plot(temperatures, temp_results["dr_mse"], "^-", label="Doubly Robust (DR)", color="#2ca02c")
    plt.plot(temperatures, temp_results["dm_mse"], "d--", label="Direct Method (DM)", color="#1f77b4")
    plt.yscale("log")
    plt.xlabel("Logging Policy Temperature T (Low = Severe Popularity Bias)")
    plt.ylabel("Mean Squared Error (MSE, log scale)")
    plt.title("OPE Estimator MSE vs Logging Policy Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "ope_mse_vs_temperature.png", dpi=200)
    plt.close()

    # Figure 2: Bias-Variance vs Sample Size
    plt.figure(figsize=(7, 4.5))
    plt.plot(sample_sizes, size_results["ips_var"], "o-", label="IPS Variance", color="#d62728")
    plt.plot(sample_sizes, size_results["snips_var"], "s-", label="SNIPS Variance", color="#ff7f0e")
    plt.plot(sample_sizes, size_results["dr_var"], "^-", label="DR Variance", color="#2ca02c")
    plt.xlabel("Logged Bandit Sample Size (N)")
    plt.ylabel("Estimator Variance")
    plt.title("OPE Estimator Variance vs Sample Size N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "ope_bias_variance_vs_samplesize.png", dpi=200)
    plt.close()

    # Figure 3: Clipping Threshold Sweep
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(clipping_thresholds, np.abs(clip_results["bias"]), "b-o", label="|Bias|")
    ax1.set_xlabel("Clipping Threshold M")
    ax1.set_ylabel("|Bias|", color="b")
    ax2 = ax1.twinx()
    ax2.plot(clipping_thresholds, clip_results["variance"], "r-s", label="Variance")
    ax2.set_ylabel("Variance", color="r")
    plt.title("Bias-Variance Tradeoff under Importance Weight Clipping")
    plt.tight_layout()
    plt.savefig(fig_dir / "ope_clipping_sweep.png", dpi=200)
    plt.close()

    # Save summary tables to CSV
    pd.DataFrame(temp_results).to_csv(table_dir / "ope_temperature_sweep.csv", index=False)
    pd.DataFrame(size_results).to_csv(table_dir / "ope_samplesize_sweep.csv", index=False)
    pd.DataFrame(clip_results).to_csv(table_dir / "ope_clipping_sweep.csv", index=False)

    logger.info("OPE study completed! Figures and tables saved in %s", reports_dir)
    return {
        "true_values": true_values,
        "temp_results": temp_results,
        "size_results": size_results,
        "clip_results": clip_results,
        "obd_results": obd_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run counterfactual OPE study.")
    parser.add_argument("--config", type=str, default="configs/dev.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    run_ope_study(config_path=args.config)
