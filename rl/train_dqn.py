"""
rl/train_dqn.py
---------------
Train DQN agent — comparison algorithm for the paper.

DQN (Deep Q-Network) learns Q-values for state-action pairs.
Used as the second RL baseline against PPO and VRTQ heuristic.

Usage:
    python -m rl.train_dqn

Author: Anthony Vallente
Project: VRTQ-RL
"""

import os

# See rl/train_ppo.py: must be set before importing torch to avoid
# CPU thread-dispatch overhead dominating tiny-MLP step time.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from dotenv import load_dotenv
from stable_baselines3 import DQN
import mlflow

from data.dataset_splits import TRAIN_SEEDS
from evaluation.mlflow_logger import start_experiment_run

load_dotenv()
torch.set_num_threads(1)

DQN_MODEL_PATH = os.getenv("DQN_MODEL_PATH", "./models/dqn_vrtq_rl.zip")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_TESTS = int(os.getenv("N_TESTS", 200))
TEST_BUDGET = int(os.getenv("TEST_BUDGET", 50))

DQN_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 50_000,
    "learning_starts": 1000,
    "batch_size": 64,
    "gamma": 0.99,
    "target_update_interval": 500,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.05,
    "verbose": 1,
    "seed": RANDOM_SEED,
}

TOTAL_TIMESTEPS = 100_000


def train_dqn(
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_path: str = DQN_MODEL_PATH,
    run_name: str = "dqn_vrtq_rl",
) -> DQN:
    from data.fault_injection import create_training_dataset
    from environment.test_prioritization_env import TestPrioritizationEnv

    print(f"\n{'='*50}")
    print("VRTQ-RL: Training DQN Agent")
    print(f"{'='*50}")

    # DQN trains on a single train-seed dataset only (no MaskableDQN exists
    # in sb3-contrib, and DQN is the secondary/cautionary baseline here, not
    # the project's RL contribution — the full multi-seed treatment is
    # reserved for PPO via rl/train_ppo.py).
    train_seed = TRAIN_SEEDS[0]
    df = create_training_dataset(n_tests=N_TESTS, seed=train_seed)
    env = TestPrioritizationEnv(test_suite_df=df, budget=TEST_BUDGET)

    with start_experiment_run(run_name, {
        **DQN_CONFIG,
        "total_timesteps": total_timesteps,
        "n_tests": N_TESTS,
        "budget": TEST_BUDGET,
        "algorithm": "DQN",
        "train_seed": train_seed,
    }) as run:
        model = DQN(env=env, **DQN_CONFIG)

        print(f"\nTraining DQN (run_id: {run.info.run_id[:8]}...)")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        # Final eval
        obs, _ = env.reset()
        done = False
        final_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, final_info = env.step(int(action))
            done = terminated or truncated

        mlflow.log_metrics({
            "final_fdr": final_info.get("fdr", 0.0),
            "final_faults_found": final_info.get("faults_found", 0),
        })

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        mlflow.log_artifact(save_path)

        print(f"\nDQN training complete! Final FDR: {final_info.get('fdr', 0):.1%}")
        print(f"Model saved to: {save_path}")

    env.close()
    return model


if __name__ == "__main__":
    train_dqn()
