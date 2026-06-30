"""
agents/test_selector_agent.py
------------------------------
TestSelectorAgent: Queries trained PPO model to rank tests.

Takes state matrix from RiskScorerAgent, runs greedy RL inference,
returns ordered list of test IDs with confidence scores.

Falls back to VRTQ heuristic if model not available.

Author: Anthony Vallente
Project: VRTQ-RL
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/ppo_vrtq_rl.zip")


class TestSelectorAgent:
    """
    AutoGen-style agent that runs RL inference for test selection.

    Loads trained PPO model and runs greedy selection over the
    full test suite given the current state matrix.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.name = "TestSelectorAgent"
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained MaskablePPO model. Falls back gracefully if not found."""
        if os.path.exists(self.model_path):
            try:
                from sb3_contrib import MaskablePPO
                self.model = MaskablePPO.load(self.model_path)
                print(f"[{self.name}] PPO model loaded from {self.model_path}")
            except Exception as e:
                print(f"[{self.name}] Could not load PPO model: {e}")
                self.model = None
        else:
            print(f"[{self.name}] Model not found at {self.model_path} — "
                  f"will use VRTQ heuristic fallback")

    def select(
        self,
        scorer_output: Dict,
        test_suite_df: pd.DataFrame,
        budget: int = 50,
    ) -> Dict:
        """
        Run test selection using PPO model or VRTQ fallback.

        Args:
            scorer_output: Output from RiskScorerAgent.score()
            test_suite_df: Full test suite DataFrame
            budget: Maximum tests to select

        Returns:
            {
                selected_indices: list[int],
                selected_test_ids: list[str],
                selection_scores: list[float],
                method_used: str,
                budget: int,
            }
        """
        state_matrix = scorer_output["state_matrix"]

        if self.model is not None:
            result = self._select_with_ppo(state_matrix, test_suite_df, budget)
        else:
            result = self._select_with_vrtq(test_suite_df, budget)

        # Attach test metadata for downstream agents
        selected_df = test_suite_df.iloc[result["selected_indices"]]
        result["selected_test_ids"] = selected_df["test_id"].tolist()
        result["selected_modules"] = selected_df["module"].tolist()
        result["selected_types"] = selected_df["test_type"].tolist()
        result["budget"] = budget

        print(f"[{self.name}] Selected {len(result['selected_indices'])} tests "
              f"via {result['method_used']} | "
              f"Top module: {result['selected_modules'][0] if result['selected_modules'] else 'N/A'}")

        return result

    def _select_with_ppo(
        self,
        state_matrix: np.ndarray,
        df: pd.DataFrame,
        budget: int,
    ) -> Dict:
        """
        Greedy, action-masked PPO inference — no exploration, and the model
        can never select an already-picked test (real masking via
        MaskablePPO's action_masks param, not a manual tiebreak fallback).
        """
        from environment.state_builder import apply_selection_mask

        n_tests = len(df)
        selected = []
        scores = []
        current_state = state_matrix.copy()

        for step in range(budget):
            obs = current_state.flatten().astype(np.float32)
            mask = np.ones(n_tests, dtype=bool)
            mask[selected] = False

            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)

            selected.append(action)
            score = float(np.max(current_state[action]))
            scores.append(score)

            # Mask out selected test
            current_state = apply_selection_mask(current_state, selected)

        return {
            "selected_indices": selected,
            "selection_scores": scores,
            "method_used": "PPO (VRTQ-RL)",
        }

    def _select_with_vrtq(
        self,
        df: pd.DataFrame,
        budget: int,
    ) -> Dict:
        """VRTQ heuristic fallback when model not available."""
        from rl.baselines.vrtq_heuristic import VRTQHeuristicSelector
        selector = VRTQHeuristicSelector()
        indices = selector.select(df, budget=budget)
        scores = df.iloc[indices]["vrtq_composite"].tolist()
        return {
            "selected_indices": indices,
            "selection_scores": scores,
            "method_used": "VRTQ Heuristic (fallback)",
        }


if __name__ == "__main__":
    from data.fault_injection import create_training_dataset
    from agents.change_analyzer_agent import ChangeAnalyzerAgent, simulate_git_diff
    from agents.risk_scorer_agent import RiskScorerAgent

    df = create_training_dataset()
    diff = simulate_git_diff(modules=["payment_service"])

    analyzer = ChangeAnalyzerAgent(use_llm=False)
    scorer = RiskScorerAgent()
    selector = TestSelectorAgent()

    features = analyzer.analyze(diff)
    scored = scorer.score(features, df)
    selected = selector.select(scored, df, budget=50)

    print(f"\nTop 5 selected tests:")
    for i, (tid, mod) in enumerate(
        zip(selected["selected_test_ids"][:5], selected["selected_modules"][:5])
    ):
        print(f"  {i+1}. {tid} ({mod})")
    print(f"\nMethod: {selected['method_used']}")
