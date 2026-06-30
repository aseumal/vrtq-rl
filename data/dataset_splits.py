"""
data/dataset_splits.py
-----------------------
Train/eval seed convention for VRTQ-RL.

TestPrioritizationEnv's observation/action space shape depends on n_tests,
so train and eval envs must keep n_tests fixed — splitting the 200-test pool
itself would just relocate data leakage to a new fixed subset. Instead, train
and eval each draw from disjoint dataset *seeds*, each producing a full
independent 200-test suite via data.fault_injection.create_training_dataset.

Author: Anthony Vallente
Project: VRTQ-RL
"""

TRAIN_SEEDS = [42, 142, 242, 342, 442]
EVAL_SEEDS = [1042, 1142, 1242]
