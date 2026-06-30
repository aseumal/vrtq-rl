"""
evaluation/validate_model.py
-----------------------------
Minimum-FDR acceptance gate for a freshly-trained PPO model.

A trained model that doesn't meaningfully beat the Random baseline on
held-out data is a silent failure mode — this is exactly what happened
before the reward-shaping/action-masking/dataset-split fixes landed: a
sub-random model was saved and only discovered broken later, at comparison
time. This script fails loudly instead: load the model, evaluate against
EVAL_SEEDS, and require mean(fdr_100) > random_fdr_100 * MIN_RELATIVE_IMPROVEMENT.

Usage:
    python -m evaluation.validate_model
    python -m evaluation.validate_model --model-path models/ppo_vrtq_rl_seed142.zip

Author: Anthony Vallente
Project: VRTQ-RL
"""

import os
import sys
import argparse
import numpy as np
from dotenv import load_dotenv

load_dotenv()

N_TESTS = int(os.getenv("N_TESTS", 200))
TEST_BUDGET = int(os.getenv("TEST_BUDGET", 50))
MIN_RELATIVE_IMPROVEMENT = 1.2  # PPO must beat Random by >=20% relative on fdr_100
RANDOM_EPISODES_PER_SEED = 10  # match compare_baselines.py's averaging — a
                                # single shuffle per seed is noisy enough to
                                # make this gate's threshold meaningless


def validate_model(model_path: str, eval_seeds=None) -> bool:
    """
    Returns True if the model at model_path beats the Random baseline by
    at least MIN_RELATIVE_IMPROVEMENT on held-out FDR@100%, False otherwise.
    """
    from sb3_contrib import MaskablePPO
    from data.dataset_splits import EVAL_SEEDS
    from data.fault_injection import create_training_dataset
    from environment.test_prioritization_env import TestPrioritizationEnv
    from evaluation.metrics import compute_all_metrics

    eval_seeds = eval_seeds or EVAL_SEEDS

    if not os.path.exists(model_path):
        print(f"[validate_model] Model not found at {model_path}")
        return False

    model = MaskablePPO.load(model_path)
    eval_dfs = [create_training_dataset(n_tests=N_TESTS, seed=s) for s in eval_seeds]

    ppo_fdrs, random_fdrs = [], []
    for seed, df in zip(eval_seeds, eval_dfs):
        env = TestPrioritizationEnv(df, budget=TEST_BUDGET)
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        env.close()
        ppo_fdrs.append(compute_all_metrics(info["selected"], df, TEST_BUDGET)["fdr_100"])

        for ep in range(RANDOM_EPISODES_PER_SEED):
            rng = np.random.default_rng(seed * 1000 + ep)
            shuffled = list(range(N_TESTS))
            rng.shuffle(shuffled)
            random_fdrs.append(compute_all_metrics(shuffled[:TEST_BUDGET], df, TEST_BUDGET)["fdr_100"])

    ppo_mean = float(np.mean(ppo_fdrs))
    random_mean = float(np.mean(random_fdrs))
    threshold = random_mean * MIN_RELATIVE_IMPROVEMENT

    passed = ppo_mean > threshold
    status = "PASS" if passed else "FAIL"
    print(f"\n[validate_model] {status}")
    print(f"  Model: {model_path}")
    print(f"  PPO FDR@100% (held-out mean over {len(eval_seeds)} seeds): {ppo_mean:.1%}")
    print(f"  Random FDR@100% (held-out mean): {random_mean:.1%}")
    print(f"  Required threshold (Random x {MIN_RELATIVE_IMPROVEMENT}): {threshold:.1%}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a trained PPO model beats Random by a minimum margin"
    )
    parser.add_argument(
        "--model-path", default=os.getenv("MODEL_PATH", "./models/ppo_vrtq_rl.zip")
    )
    args = parser.parse_args()

    ok = validate_model(args.model_path)
    sys.exit(0 if ok else 1)
