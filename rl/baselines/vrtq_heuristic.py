"""
rl/baselines/vrtq_heuristic.py
--------------------------------
VRTQ heuristic baseline — Anthony Seumal's proprietary framework.

Sorts tests by composite VRTQ score:
  composite = 0.30*Value + 0.35*Risk + 0.20*Time + 0.15*Quality

This is the key baseline VRTQ-RL must outperform to prove RL adds value
beyond the heuristic. It's also the intellectual contribution that makes
this research unique — we're comparing against our own prior work.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
from typing import List


VRTQ_WEIGHTS = {
    "value": 0.30,
    "risk": 0.35,
    "time": 0.20,
    "quality": 0.15,
}


class VRTQHeuristicSelector:
    """
    Deterministic test selector using VRTQ composite score.
    Always produces the same ordering for the same input.
    """

    def __init__(self):
        self.weights = VRTQ_WEIGHTS

    def compute_composite(self, df: pd.DataFrame) -> pd.Series:
        """Recompute composite score (even if already in df)."""
        return (
            self.weights["value"] * df["vrtq_value_score"]
            + self.weights["risk"] * df["vrtq_risk_score"]
            + self.weights["time"] * df["vrtq_time_score"]
            + self.weights["quality"] * df["vrtq_quality_score"]
        )

    def select(self, df: pd.DataFrame, budget: int = 50) -> List[int]:
        """
        Returns test indices sorted by VRTQ composite score (descending).

        Args:
            df: Test suite DataFrame
            budget: Number of tests to select

        Returns:
            List of integer indices sorted by VRTQ priority
        """
        scores = self.compute_composite(df)
        sorted_indices = scores.argsort()[::-1].tolist()
        return sorted_indices[:budget]

    def run_episode(self, df: pd.DataFrame, budget: int = 50) -> dict:
        """
        Simulate a full prioritization episode using VRTQ ordering.
        """
        selected_indices = self.select(df, budget)
        selected_df = df.iloc[selected_indices].reset_index(drop=True)

        total_faults = df["has_fault"].sum()
        faults_found = selected_df["has_fault"].sum()

        ttff_mask = selected_df["has_fault"].values
        ttff = int(np.argmax(ttff_mask)) + 1 if ttff_mask.any() else budget + 1

        fdr = faults_found / max(1, total_faults)

        cumulative = selected_df["has_fault"].cumsum()
        target = int(0.8 * total_faults)
        tsr_indices = np.where(cumulative >= target)[0]
        tests_for_80pct = int(tsr_indices[0]) + 1 if len(tsr_indices) > 0 else budget
        tsr = 1.0 - (tests_for_80pct / len(df))

        return {
            "method": "vrtq_heuristic",
            "fdr": round(float(fdr), 4),
            "ttff": ttff,
            "tsr": round(float(tsr), 4),
            "faults_found": int(faults_found),
            "total_faults": int(total_faults),
            "budget_used": budget,
            "selected_test_ids": selected_df["test_id"].tolist(),
        }
