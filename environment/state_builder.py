"""
environment/state_builder.py
----------------------------
Converts test suite DataFrame into normalized numpy state vectors
for the Gymnasium RL environment.

State vector (10 features per test):
  [0] vrtq_value_score       - business value (VRTQ)
  [1] vrtq_risk_score        - defect risk (VRTQ)
  [2] vrtq_time_score        - execution efficiency (VRTQ)
  [3] vrtq_quality_score     - freshness / staleness (VRTQ)
  [4] files_changed_overlap  - git diff relevance (0-1)
  [5] module_dep_depth       - dependency depth normalized (0-1)
  [6] historical_failure_rate - raw failure rate
  [7] days_since_last_run_norm - normalized staleness
  [8] execution_time_norm    - normalized exec time
  [9] test_type_encoded      - unit=0.0, integration=0.5, e2e=1.0

Author: Anthony Seumal
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

STATE_DIM = 10  # number of features per test

TEST_TYPE_ENCODING = {"unit": 0.0, "integration": 0.5, "e2e": 1.0}

MAX_EXEC_TIME = 120.0   # seconds
MAX_DAYS = 30           # days since last run
MAX_DEP_DEPTH = 5       # maximum dependency depth


def build_state_matrix(
    df: pd.DataFrame,
    git_diff_features: Optional[Dict] = None,
) -> np.ndarray:
    """
    Build a [n_tests x STATE_DIM] state matrix from the test suite.

    Args:
        df: Test suite DataFrame (from synthetic_test_suite + fault_injection)
        git_diff_features: Optional dict with keys:
            - files_changed: list of changed file paths
            - modules_affected: list of affected module names
            - churn_score: float 0-1
            - dependency_depth: int

    Returns:
        numpy array of shape (n_tests, STATE_DIM), dtype float32
    """
    n = len(df)
    state = np.zeros((n, STATE_DIM), dtype=np.float32)

    # Features 0-3: VRTQ scores (already normalized 0-1)
    state[:, 0] = df["vrtq_value_score"].values
    state[:, 1] = df["vrtq_risk_score"].values
    state[:, 2] = df["vrtq_time_score"].values
    state[:, 3] = df["vrtq_quality_score"].values

    # Feature 4: git diff overlap
    # How much does the changed code affect each test's module?
    if git_diff_features and "modules_affected" in git_diff_features:
        affected = set(git_diff_features["modules_affected"])
        churn = git_diff_features.get("churn_score", 0.5)
        state[:, 4] = np.where(
            df["module"].isin(affected),
            churn,  # full churn weight if module is affected
            churn * 0.1,  # small residual for unaffected modules
        )
    else:
        # Default: uniform medium relevance when no git info available
        state[:, 4] = 0.5

    # Feature 5: module dependency depth (normalized)
    if git_diff_features and "dependency_depth" in git_diff_features:
        depth_norm = min(1.0, git_diff_features["dependency_depth"] / MAX_DEP_DEPTH)
        # Propagate depth to affected modules, attenuate for others
        affected = set(git_diff_features.get("modules_affected", []))
        state[:, 5] = np.where(df["module"].isin(affected), depth_norm, depth_norm * 0.2)
    else:
        state[:, 5] = 0.3  # default moderate depth

    # Feature 6: historical failure rate (already 0-1)
    state[:, 6] = df["historical_failure_rate"].values

    # Feature 7: days since last run (normalized)
    state[:, 7] = np.clip(df["days_since_last_run"].values / MAX_DAYS, 0.0, 1.0)

    # Feature 8: execution time (normalized, inverted — faster = higher score)
    state[:, 8] = np.clip(
        1.0 - (df["execution_time_seconds"].values / MAX_EXEC_TIME), 0.0, 1.0
    )

    # Feature 9: test type encoded
    state[:, 9] = df["test_type"].map(TEST_TYPE_ENCODING).values

    return state


def get_feature_names() -> list:
    """Returns ordered list of feature names (for explainability)."""
    return [
        "vrtq_value_score",
        "vrtq_risk_score",
        "vrtq_time_score",
        "vrtq_quality_score",
        "files_changed_overlap",
        "module_dep_depth",
        "historical_failure_rate",
        "days_since_last_run_norm",
        "execution_time_norm",
        "test_type_encoded",
    ]


def apply_selection_mask(
    state_matrix: np.ndarray,
    selected_indices: list,
) -> np.ndarray:
    """
    Zero out already-selected tests in the state matrix.
    This prevents the RL agent from re-selecting tests.
    """
    masked = state_matrix.copy()
    masked[selected_indices, :] = 0.0
    return masked
