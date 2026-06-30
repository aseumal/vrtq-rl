"""
agents/agentic/tools.py
-------------------------
PipelineToolkit: wraps the existing, unmodified agent methods
(ChangeAnalyzerAgent.analyze, RiskScorerAgent.score, TestSelectorAgent.select,
ReportAgent.generate) behind JSON-safe functions an AutoGen agent can call.

The LLM never sees a raw DataFrame or ndarray — the toolkit holds that state
in session-scoped instance attributes and only exposes/accepts primitives
(str, int, list[str], dict of scalars) across the tool-call boundary.

Author: Anthony Seumal
Project: VRTQ-RL
"""

from collections import Counter
from typing import Dict, List, Optional

import pandas as pd


class PipelineToolkit:
    """
    Session-scoped wrapper around the four pipeline stages. One instance per
    agentic pipeline run (not shared across requests).
    """

    def __init__(self, test_suite_df: pd.DataFrame, budget: int, diff_text: str):
        self.test_suite_df = test_suite_df
        self.budget = budget
        self._diff_text = diff_text

        self.diff_features: Optional[Dict] = None
        self.scorer_output: Optional[Dict] = None
        self.selector_output: Optional[Dict] = None
        self.report: Optional[Dict] = None

        from agents.change_analyzer_agent import ChangeAnalyzerAgent
        from agents.risk_scorer_agent import RiskScorerAgent
        from agents.test_selector_agent import TestSelectorAgent
        from agents.report_agent import ReportAgent

        # The toolkit's own calls into these agents stay deterministic
        # (use_llm=False) — the LLM reasoning happens one level up, in the
        # Supervisor/Critic conversation, not duplicated inside each stage.
        self._analyzer = ChangeAnalyzerAgent(use_llm=False)
        self._scorer = RiskScorerAgent()
        self._selector = TestSelectorAgent()
        self._reporter = ReportAgent(use_llm=False)

    def analyze_diff(self) -> Dict:
        """Parse the currently loaded git diff into structured change features (modules, churn, change type)."""
        self.diff_features = self._analyzer.analyze(self._diff_text)
        return self.diff_features

    def score_risk(self) -> Dict:
        """Compute the VRTQ risk summary for the most recently analyzed diff. Call after analyze_diff."""
        if self.diff_features is None:
            raise ValueError("Call analyze_diff first")
        self.scorer_output = self._scorer.score(self.diff_features, self.test_suite_df)
        return self.scorer_output["risk_summary"]

    def select_tests(self, budget: Optional[int] = None, focus_modules: Optional[List[str]] = None) -> Dict:
        """
        Run RL test selection. Call after score_risk. Can be re-invoked with
        a different budget or focus_modules to refine the result (e.g. after
        Critic feedback).
        """
        if self.scorer_output is None:
            raise ValueError("Call score_risk first")

        effective_budget = budget or self.budget
        scorer_output = self.scorer_output
        if focus_modules:
            # Rebuild the state matrix with the requested module focus.
            # RiskScorerAgent.score() uses diff_features["modules_affected"]
            # to compute the git-diff-overlap features baked into the state
            # matrix — TestSelectorAgent only ever reads state_matrix, not
            # git_context, so patching the output dict alone would be a
            # no-op; the matrix must actually be rebuilt via re-scoring.
            adjusted_features = {**self.diff_features, "modules_affected": focus_modules}
            scorer_output = self._scorer.score(adjusted_features, self.test_suite_df)

        self.selector_output = self._selector.select(scorer_output, self.test_suite_df, budget=effective_budget)
        self.budget = effective_budget
        return {
            "method_used": self.selector_output["method_used"],
            "n_selected": len(self.selector_output["selected_indices"]),
            "top_5_test_ids": self.selector_output["selected_test_ids"][:5],
            "selected_modules_distribution": dict(Counter(self.selector_output["selected_modules"])),
        }

    def get_report_metrics(self) -> Dict:
        """Generate the final report (FDR/TTFF/TSR metrics + top tests) from the current selection. Call after select_tests."""
        if self.selector_output is None:
            raise ValueError("Call select_tests first")
        self.report = self._reporter.generate(
            self.selector_output, self.scorer_output, self.test_suite_df, budget=self.budget
        )
        return self.report["metrics"]
