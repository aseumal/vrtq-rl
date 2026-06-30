"""
evaluation/compare_baselines.py
--------------------------------
Runs all four methods and produces comparison results.

Methods compared:
1. Random (floor baseline)
2. VRTQ Heuristic (Anthony's proprietary framework)
3. DQN (RL comparison)
4. PPO — VRTQ-RL (primary contribution)

All four methods are evaluated identically on the same held-out EVAL_SEEDS
datasets (data.dataset_splits) — datasets PPO never saw during training, so
the comparison isn't confounded by RL-specific overfitting to a training
dataset the other (stateless) methods were never at risk of overfitting to.

Usage:
    python -m evaluation.compare_baselines

Author: Anthony Seumal
Project: VRTQ-RL
"""

import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO
import mlflow

from data.dataset_splits import EVAL_SEEDS

load_dotenv()

N_TESTS = int(os.getenv("N_TESTS", 200))
TEST_BUDGET = int(os.getenv("TEST_BUDGET", 50))
PPO_MODEL_PATH = os.getenv("MODEL_PATH", "./models/ppo_vrtq_rl.zip")
DQN_MODEL_PATH = os.getenv("DQN_MODEL_PATH", "./models/dqn_vrtq_rl.zip")
N_EVAL_EPISODES = 10  # resampling count for Random's internal stochasticity


def run_rl_episode(model, env, masked: bool = False) -> dict:
    """Run one greedy episode with a trained SB3 model."""
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        if masked:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return info


def compare_all_methods(
    n_episodes: int = N_EVAL_EPISODES,
    output_path: str = "evaluation/results.json",
    eval_seeds=None,
) -> pd.DataFrame:
    from data.fault_injection import create_training_dataset
    from environment.test_prioritization_env import TestPrioritizationEnv
    from rl.baselines.random_selector import RandomSelector
    from rl.baselines.vrtq_heuristic import VRTQHeuristicSelector
    from evaluation.metrics import compute_all_metrics, print_metrics_table

    eval_seeds = eval_seeds or EVAL_SEEDS

    print("\n" + "=" * 60)
    print("VRTQ-RL: Baseline Comparison")
    print(f"Held-out eval seeds: {eval_seeds}")
    print("=" * 60)

    eval_dfs = [create_training_dataset(n_tests=N_TESTS, seed=s) for s in eval_seeds]

    ppo_model = MaskablePPO.load(PPO_MODEL_PATH) if os.path.exists(PPO_MODEL_PATH) else None
    dqn_model = DQN.load(DQN_MODEL_PATH) if os.path.exists(DQN_MODEL_PATH) else None

    method_metrics = {"Random": [], "VRTQ Heuristic": [], "DQN": [], "PPO (VRTQ-RL)": []}

    # --- Random Baseline ---
    print("\n[1/4] Running Random baseline...")
    random_selector = RandomSelector()
    for seed, df in zip(eval_seeds, eval_dfs):
        for ep in range(n_episodes):
            rng = np.random.default_rng(seed * 1000 + ep)
            shuffled = list(range(N_TESTS))
            rng.shuffle(shuffled)
            method_metrics["Random"].append(
                compute_all_metrics(shuffled[:TEST_BUDGET], df, TEST_BUDGET)
            )

    # --- VRTQ Heuristic ---
    print("[2/4] Running VRTQ Heuristic baseline...")
    vrtq_selector = VRTQHeuristicSelector()
    for df in eval_dfs:
        indices = vrtq_selector.select(df, budget=TEST_BUDGET)
        method_metrics["VRTQ Heuristic"].append(compute_all_metrics(indices, df, TEST_BUDGET))

    # --- DQN ---
    print("[3/4] Running DQN agent...")
    if dqn_model is not None:
        for df in eval_dfs:
            env = TestPrioritizationEnv(df, budget=TEST_BUDGET)
            info = run_rl_episode(dqn_model, env)
            method_metrics["DQN"].append(compute_all_metrics(info["selected"], df, TEST_BUDGET))
            env.close()
    else:
        print("  DQN model not found — skipping (train first with: python -m rl.train_dqn)")

    # --- PPO (VRTQ-RL) ---
    print("[4/4] Running PPO (VRTQ-RL) agent...")
    if ppo_model is not None:
        for df in eval_dfs:
            env = TestPrioritizationEnv(df, budget=TEST_BUDGET)
            info = run_rl_episode(ppo_model, env, masked=True)
            method_metrics["PPO (VRTQ-RL)"].append(compute_all_metrics(info["selected"], df, TEST_BUDGET))
            env.close()
    else:
        print("  PPO model not found — skipping (train first with: python -m rl.train_ppo)")

    all_results = []
    for method, metrics_list in method_metrics.items():
        if not metrics_list:
            all_results.append({
                "method": method, "fdr_25": 0, "fdr_50": 0, "fdr_100": 0, "ttff": 0, "tsr": 0,
                "note": "model not found — train first",
            })
            continue
        all_results.append({
            "method": method,
            **{k: round(float(np.mean([r[k] for r in metrics_list])), 4)
               for k in ["fdr_25", "fdr_50", "fdr_100", "ttff", "tsr"]}
        })

    # Print table
    print_metrics_table(all_results)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("vrtq-rl-experiments")
    with mlflow.start_run(run_name="baseline_comparison"):
        for r in all_results:
            method = r["method"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            for metric, val in r.items():
                if metric not in ("method", "note"):
                    mlflow.log_metric(f"{method}_{metric}", val)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    results_df = compare_all_methods()
    print("\nFull results DataFrame:")
    print(results_df.to_string(index=False))
