"""
rl/baselines/random_selector.py
--------------------------------
Random test selection baseline.
Selects tests in random order — no intelligence.
Used as the floor baseline in comparisons.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
from typing import List


class RandomSelector:
    """
    Randomly shuffles the test suite and returns the ordering.
    Pure baseline — no features used.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select(self, df: pd.DataFrame, budget: int = 50) -> List[int]:
        """
        Returns a list of test indices in random order.

        Args:
            df: Test suite DataFrame
            budget: Number of tests to select

        Returns:
            List of integer indices (rows in df)
        """
        indices = list(range(len(df)))
        self.rng.shuffle(indices)
        return indices[:budget]

    def run_episode(self, df: pd.DataFrame, budget: int = 50) -> dict:
        """
        Simulate a full prioritization episode.

        Returns:
            dict with FDR, TTFF, TSR, and selected test IDs
        """
        selected_indices = self.select(df, budget)
        selected_df = df.iloc[selected_indices].reset_index(drop=True)

        total_faults = df["has_fault"].sum()
        faults_found = selected_df["has_fault"].sum()

        # Time to First Failure: position of first fault-revealing test
        ttff_mask = selected_df["has_fault"].values
        ttff = int(np.argmax(ttff_mask)) + 1 if ttff_mask.any() else budget + 1

        # FDR: fault detection rate
        fdr = faults_found / max(1, total_faults)

        # TSR: test suite reduction
        # Tests needed to find 80% of faults
        cumulative = selected_df["has_fault"].cumsum()
        target = int(0.8 * total_faults)
        tsr_indices = np.where(cumulative >= target)[0]
        tests_for_80pct = int(tsr_indices[0]) + 1 if len(tsr_indices) > 0 else budget
        tsr = 1.0 - (tests_for_80pct / len(df))

        return {
            "method": "random",
            "fdr": round(float(fdr), 4),
            "ttff": ttff,
            "tsr": round(float(tsr), 4),
            "faults_found": int(faults_found),
            "total_faults": int(total_faults),
            "budget_used": budget,
            "selected_test_ids": selected_df["test_id"].tolist(),
        }
