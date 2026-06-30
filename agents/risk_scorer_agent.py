"""
agents/risk_scorer_agent.py
----------------------------
RiskScorerAgent: Applies VRTQ scoring to produce the RL state vector.

Takes git diff features from ChangeAnalyzerAgent and the test suite,
returns a full state matrix ready for the PPO model.

Pure computation — no LLM call needed. Zero token cost.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from environment.state_builder import build_state_matrix, STATE_DIM


class RiskScorerAgent:
    """
    AutoGen-style agent that builds VRTQ state vectors.

    Bridges ChangeAnalyzerAgent output → RL environment input.
    No LLM required — purely deterministic VRTQ computation.
    """

    def __init__(self):
        self.name = "RiskScorerAgent"

    def score(
        self,
        diff_features: Dict,
        test_suite_df: pd.DataFrame,
    ) -> Dict:
        """
        Build state matrix from diff features + test suite.

        Args:
            diff_features: Output from ChangeAnalyzerAgent.analyze()
            test_suite_df: Full test suite DataFrame

        Returns:
            {
                state_matrix: np.ndarray (n_tests x STATE_DIM),
                top_risk_modules: list[str],
                risk_summary: dict,
                git_context: dict,
            }
        """
        # Build full state matrix
        state_matrix = build_state_matrix(test_suite_df, diff_features)

        # Identify top risk modules (for ReportAgent)
        affected = set(diff_features.get("modules_affected", []))
        module_risk = (
            test_suite_df[test_suite_df["module"].isin(affected)]
            .groupby("module")["vrtq_risk_score"]
            .mean()
            .sort_values(ascending=False)
        )
        top_risk_modules = module_risk.index.tolist()[:3]

        # Compute risk summary statistics
        affected_mask = test_suite_df["module"].isin(affected)
        n_affected_tests = int(affected_mask.sum())
        avg_risk = float(
            test_suite_df.loc[affected_mask, "vrtq_risk_score"].mean()
            if n_affected_tests > 0
            else 0.0
        )
        high_risk_count = int(
            (test_suite_df.loc[affected_mask, "vrtq_risk_score"] > 0.7).sum()
        )

        risk_summary = {
            "modules_affected": list(affected),
            "n_affected_tests": n_affected_tests,
            "avg_risk_score": round(avg_risk, 3),
            "high_risk_tests": high_risk_count,
            "churn_score": diff_features.get("churn_score", 0.0),
            "dependency_depth": diff_features.get("dependency_depth", 1),
        }

        print(f"[{self.name}] State matrix: {state_matrix.shape} | "
              f"Affected tests: {n_affected_tests} | "
              f"Avg risk: {avg_risk:.3f} | "
              f"High-risk: {high_risk_count}")

        return {
            "state_matrix": state_matrix,
            "top_risk_modules": top_risk_modules,
            "risk_summary": risk_summary,
            "git_context": diff_features,
        }

    def get_high_risk_tests(
        self,
        test_suite_df: pd.DataFrame,
        diff_features: Dict,
        threshold: float = 0.7,
    ) -> pd.DataFrame:
        """
        Return tests above risk threshold in affected modules.
        Used by ReportAgent for explanation generation.
        """
        affected = set(diff_features.get("modules_affected", []))
        mask = (
            test_suite_df["module"].isin(affected)
            & (test_suite_df["vrtq_risk_score"] > threshold)
        )
        return test_suite_df[mask].sort_values(
            "vrtq_composite", ascending=False
        ).head(10)


if __name__ == "__main__":
    from data.fault_injection import create_training_dataset
    from agents.change_analyzer_agent import ChangeAnalyzerAgent, simulate_git_diff

    df = create_training_dataset()
    diff = simulate_git_diff(modules=["payment_service", "auth_service"])

    analyzer = ChangeAnalyzerAgent(use_llm=False)
    scorer = RiskScorerAgent()

    features = analyzer.analyze(diff)
    result = scorer.score(features, df)

    print(f"\nState matrix shape: {result['state_matrix'].shape}")
    print(f"Top risk modules: {result['top_risk_modules']}")
    print(f"Risk summary: {result['risk_summary']}")
