"""
rl/train_ppo.py
---------------
Train PPO agent on the VRTQ-RL test prioritization environment.

Uses sb3-contrib's MaskablePPO so the policy can never select an
already-picked test (replaces the old soft -0.2 penalty, which let the
agent waste budget repeatedly proposing invalid actions). Trains across
multiple independent dataset seeds (TRAIN_SEEDS) and evaluates on disjoint
held-out seeds (EVAL_SEEDS) — training and evaluation on the same fixed
dataset previously let PPO overfit to seed-42-specific patterns rather than
learn a generalizable selection policy.

Usage:
    python -m rl.train_ppo

Author: Anthony Vallente
Project: VRTQ-RL
"""

import os

# Tiny MLP + small batch sizes mean PyTorch's default multi-threaded CPU
# matmul dispatch overhead dwarfs the actual compute (e.g. 4000 steps
# went from ~minutes to ~1.5s). Must be set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from dotenv import load_dotenv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import mlflow

from data.dataset_splits import TRAIN_SEEDS, EVAL_SEEDS
from evaluation.mlflow_logger import start_experiment_run

load_dotenv()
torch.set_num_threads(1)

MODEL_PATH = os.getenv("MODEL_PATH", "./models/ppo_vrtq_rl.zip")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_TESTS = int(os.getenv("N_TESTS", 200))
TEST_BUDGET = int(os.getenv("TEST_BUDGET", 50))

# PPO Hyperparameters
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "ent_coef": 0.05,   # raised from 0.01 — 200-way discrete action space
                         # needs more exploration pressure to avoid premature
                         # convergence to a narrow policy
    "verbose": 1,
    "seed": RANDOM_SEED,
}

TOTAL_TIMESTEPS = 500_000


def _mask_fn(env):
    return env.action_masks()


def _make_env(seed: int):
    """Build one action-masked TestPrioritizationEnv on a given dataset seed."""
    from data.fault_injection import create_training_dataset
    from environment.test_prioritization_env import TestPrioritizationEnv

    df = create_training_dataset(n_tests=N_TESTS, seed=seed)
    env = TestPrioritizationEnv(test_suite_df=df, budget=TEST_BUDGET)
    return ActionMasker(env, _mask_fn)


class FDRLoggingCallback(BaseCallback):
    """
    Custom SB3 callback that logs mean FDR across held-out eval datasets to
    MLflow every N steps, using masked greedy inference.
    """

    def __init__(self, eval_envs, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_envs = eval_envs
        self.log_freq = log_freq
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_freq:
            self._last_log = self.num_timesteps
            fdrs = []
            for env in self.eval_envs:
                obs, _ = env.reset()
                done = False
                info = {}
                while not done:
                    masks = env.action_masks()
                    action, _ = self.model.predict(
                        obs, deterministic=True, action_masks=masks
                    )
                    obs, _, terminated, truncated, info = env.step(int(action))
                    done = terminated or truncated
                fdrs.append(info.get("fdr", 0.0))

            fdr = float(np.mean(fdrs))
            mlflow.log_metric("eval_fdr", fdr, step=self.num_timesteps)

            if self.verbose > 0:
                print(
                    f"[Step {self.num_timesteps}] Eval FDR "
                    f"(mean over {len(fdrs)} held-out seeds): {fdr:.1%}"
                )
        return True


def train_ppo(
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_path: str = MODEL_PATH,
    run_name: str = "ppo_vrtq_rl",
    train_seeds=None,
    eval_seeds=None,
    seed: int = None,
) -> MaskablePPO:
    """
    Train action-masked PPO across multiple train-seed datasets and evaluate
    on held-out eval-seed datasets.

    Args:
        total_timesteps: Total env steps for training
        save_path: Path to save trained model (.zip)
        run_name: MLflow run name
        train_seeds: Override TRAIN_SEEDS (used by the multi-seed sweep)
        eval_seeds: Override EVAL_SEEDS
        seed: Override PPO's own stochasticity seed (PPO_CONFIG["seed"]);
            defaults to RANDOM_SEED if not given

    Returns:
        Trained MaskablePPO model
    """
    train_seeds = train_seeds or TRAIN_SEEDS
    eval_seeds = eval_seeds or EVAL_SEEDS
    ppo_config = {**PPO_CONFIG, "seed": seed if seed is not None else PPO_CONFIG["seed"]}

    print(f"\n{'='*50}")
    print("VRTQ-RL: Training PPO Agent (MaskablePPO)")
    print(f"{'='*50}")
    print(f"Tests: {N_TESTS} | Budget: {TEST_BUDGET} | Steps: {total_timesteps:,}")
    print(f"Train seeds: {train_seeds} | Eval seeds (held-out): {eval_seeds}")

    train_env = DummyVecEnv([(lambda s=s: _make_env(s)) for s in train_seeds])
    eval_envs = [_make_env(s) for s in eval_seeds]

    with start_experiment_run(run_name, {
        **ppo_config,
        "total_timesteps": total_timesteps,
        "n_tests": N_TESTS,
        "budget": TEST_BUDGET,
        "algorithm": "MaskablePPO",
        "train_seeds": str(train_seeds),
        "eval_seeds": str(eval_seeds),
    }) as run:
        model = MaskablePPO(env=train_env, **ppo_config)

        fdr_callback = FDRLoggingCallback(eval_envs=eval_envs, log_freq=5000, verbose=1)

        print(f"\nTraining PPO (run_id: {run.info.run_id[:8]}...)")
        model.learn(
            total_timesteps=total_timesteps,
            callback=fdr_callback,
            progress_bar=True,
        )

        # Final evaluation across all held-out eval seeds
        final_fdrs, final_faults, final_rewards = [], [], []
        for env in eval_envs:
            obs, _ = env.reset()
            done = False
            info = {}
            while not done:
                masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            final_fdrs.append(info.get("fdr", 0.0))
            final_faults.append(info.get("faults_found", 0))
            final_rewards.append(info.get("total_reward", 0.0))

        final_fdr = float(np.mean(final_fdrs))
        mlflow.log_metrics({
            "final_fdr": final_fdr,
            "final_fdr_std": float(np.std(final_fdrs)),
            "final_faults_found": float(np.mean(final_faults)),
            "final_total_reward": float(np.mean(final_rewards)),
        })

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        mlflow.log_artifact(save_path)

        print(f"\n{'='*50}")
        print("Training complete!")
        print(f"Final FDR (mean over {len(eval_seeds)} held-out seeds): {final_fdr:.1%}")
        print(f"Model saved to: {save_path}")
        print(f"MLflow run: {run.info.run_id}")
        print(f"{'='*50}")

    train_env.close()
    for env in eval_envs:
        env.close()

    # Minimum-FDR acceptance gate: fail loudly here rather than silently
    # saving a sub-random model that's only discovered broken later, at
    # comparison time. Doesn't raise/exit (train_ppo() is also called in a
    # loop by evaluation/run_seed_sweep.py) — just prints PASS/FAIL.
    from evaluation.validate_model import validate_model
    validate_model(save_path, eval_seeds=eval_seeds)

    return model


if __name__ == "__main__":
    train_ppo()
