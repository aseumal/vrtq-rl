"""
evaluation/run_seed_sweep.py
------------------------------
Multi-seed PPO training + evaluation sweep.

A single seed=42 training run is not sufficient evidence that PPO beats its
baselines — RL training has real run-to-run variance, especially on a
small problem (200 tests, 37 faults) like this one. This script trains PPO
once per TRAIN_SEEDS entry (reusing each seed for both the model's own
stochasticity and as its primary training dataset), evaluates each
resulting model against the same held-out EVAL_SEEDS, and reports
mean +/- std FDR@100% (and the other metrics) across the sweep — this,
not a single run, is the headline result that should go in the README.

Usage:
    python -m evaluation.run_seed_sweep

Author: Anthony Vallente
Project: VRTQ-RL
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

OMP = os.environ.setdefault("OMP_NUM_THREADS", "1")

from data.dataset_splits import TRAIN_SEEDS, EVAL_SEEDS

MODELS_DIR = os.getenv("MODELS_DIR", "./models")


def run_sweep(train_seeds=None, eval_seeds=None, total_timesteps=None, output_path="evaluation/seed_sweep_results.json"):
    import torch
    torch.set_num_threads(1)

    from rl.train_ppo import train_ppo, TOTAL_TIMESTEPS
    from evaluation.compare_baselines import run_rl_episode
    from data.fault_injection import create_training_dataset
    from environment.test_prioritization_env import TestPrioritizationEnv
    from evaluation.metrics import compute_all_metrics
    from sb3_contrib import MaskablePPO

    train_seeds = train_seeds or TRAIN_SEEDS
    eval_seeds = eval_seeds or EVAL_SEEDS
    total_timesteps = total_timesteps or TOTAL_TIMESTEPS

    eval_dfs = [create_training_dataset(n_tests=int(os.getenv("N_TESTS", 200)), seed=s) for s in eval_seeds]
    budget = int(os.getenv("TEST_BUDGET", 50))

    per_run_metrics = []

    for i, seed in enumerate(train_seeds):
        print(f"\n{'#'*60}")
        print(f"# Seed sweep run {i+1}/{len(train_seeds)} — seed={seed}")
        print(f"{'#'*60}")

        model_path = os.path.join(MODELS_DIR, f"ppo_vrtq_rl_seed{seed}.zip")
        train_ppo(
            total_timesteps=total_timesteps,
            save_path=model_path,
            run_name=f"ppo_vrtq_rl_seed{seed}",
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            seed=seed,
        )

        model = MaskablePPO.load(model_path)
        fdrs = []
        for df in eval_dfs:
            env = TestPrioritizationEnv(df, budget=budget)
            info = run_rl_episode(model, env, masked=True)
            fdrs.append(compute_all_metrics(info["selected"], df, budget)["fdr_100"])
            env.close()

        per_run_metrics.append({"seed": seed, "fdr_100_mean": float(np.mean(fdrs)), "fdr_100_per_eval_seed": fdrs})
        print(f"Run seed={seed}: held-out FDR@100% = {np.mean(fdrs):.1%}")

    fdr_means = [r["fdr_100_mean"] for r in per_run_metrics]
    summary = {
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "total_timesteps": total_timesteps,
        "per_run": per_run_metrics,
        "fdr_100_mean": float(np.mean(fdr_means)),
        "fdr_100_std": float(np.std(fdr_means)),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SEED SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"PPO FDR@100% across {len(train_seeds)} seeds: "
          f"{summary['fdr_100_mean']:.1%} +/- {summary['fdr_100_std']:.1%}")
    print(f"Per-seed: {[f'{m:.1%}' for m in fdr_means]}")
    print(f"Results saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_sweep()
