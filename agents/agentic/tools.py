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

    # ── Scenario 2: module-mapping ambiguity (Phase 0, before score_risk) ──

    def get_module_ambiguity_report(self) -> Dict:
        """
        Flag any file in the analyzed diff that matches 2+ modules' keyword
        lists, AND whose candidate modules are still both present in the
        current modules_affected. Call after analyze_diff, before score_risk
        (correcting modules_affected after score_risk would require a full
        re-score).

        The second condition matters: find_ambiguous_files() is pure
        keyword matching with no notion of prior corrections, so without
        filtering by current modules_affected, a file whose ambiguity was
        already resolved via reexamine_module_match() would keep reporting
        as ambiguous forever, since its keywords never change — this tool
        is meant to reflect current truth, not raw/stale keyword matches.
        """
        if self.diff_features is None:
            raise ValueError("Call analyze_diff first")
        ambiguous_files = self._analyzer.find_ambiguous_files(self.diff_features.get("files_changed", []))
        affected = set(self.diff_features.get("modules_affected", []))
        still_ambiguous = [
            af for af in ambiguous_files
            if len(set(af["candidate_modules"]) & affected) >= 2
        ]
        return {
            "ambiguous_files": still_ambiguous,
            "modules_affected": self.diff_features.get("modules_affected", []),
        }

    def reexamine_module_match(self, file: str, module_to_confirm_or_retract: str, keep: bool) -> Dict:
        """
        Apply a decision about one ambiguous file->module match: remove or
        keep `module_to_confirm_or_retract` in modules_affected. Idempotent.
        Call after get_module_ambiguity_report, before score_risk (score_risk
        hasn't run yet in Phase 0, so no re-scoring is needed here, unlike
        select_tests's focus_modules path which corrects an already-scored run).
        """
        if self.diff_features is None:
            raise ValueError("Call analyze_diff first")
        modules = set(self.diff_features.get("modules_affected", []))
        if keep:
            modules.add(module_to_confirm_or_retract)
        else:
            modules.discard(module_to_confirm_or_retract)
        self.diff_features["modules_affected"] = list(modules)
        return {"modules_affected": self.diff_features["modules_affected"]}

    # ── Scenario 1: execution-time SLA negotiation (Phase 2) ──

    def get_selection_time_budget_check(self, sla_seconds: int = 1200) -> Dict:
        """
        Sum execution_time_seconds for the current selection. Call after
        select_tests. Pure aggregation — no re-selection happens here.
        """
        if self.selector_output is None:
            raise ValueError("Call select_tests first")
        selected_df = self.test_suite_df.iloc[self.selector_output["selected_indices"]]
        total_seconds = float(selected_df["execution_time_seconds"].sum())
        n_selected = len(self.selector_output["selected_indices"])
        return {
            "total_seconds": round(total_seconds, 1),
            "sla_seconds": sla_seconds,
            "over_sla": total_seconds > sla_seconds,
            "n_selected": n_selected,
            "avg_seconds_per_test": round(total_seconds / max(1, n_selected), 2),
        }

    def select_tests_by_time_budget(
        self, time_budget_seconds: int, focus_modules: Optional[List[str]] = None
    ) -> Dict:
        """
        Re-rank by truncating the FULL PPO/VRTQ ranking (budget=len(df), i.e.
        "rank everything") by cumulative execution_time_seconds instead of by
        count. Does not modify TestSelectorAgent.select()'s signature/behavior
        at all — "rank everything, then truncate differently" is exactly what
        budget=len(df) already gives.
        """
        if self.scorer_output is None:
            raise ValueError("Call score_risk first")

        scorer_output = self.scorer_output
        if focus_modules:
            adjusted_features = {**self.diff_features, "modules_affected": focus_modules}
            scorer_output = self._scorer.score(adjusted_features, self.test_suite_df)

        full_n = len(self.test_suite_df)
        full_ranking = self._selector.select(scorer_output, self.test_suite_df, budget=full_n)

        selected_indices, selected_test_ids = [], []
        selected_modules, selected_types, selection_scores = [], [], []
        cumulative = 0.0
        for idx, tid, mod, ttype, score in zip(
            full_ranking["selected_indices"], full_ranking["selected_test_ids"],
            full_ranking["selected_modules"], full_ranking["selected_types"],
            full_ranking["selection_scores"],
        ):
            test_time = float(self.test_suite_df.iloc[idx]["execution_time_seconds"])
            if cumulative + test_time > time_budget_seconds and selected_indices:
                break
            cumulative += test_time
            selected_indices.append(idx)
            selected_test_ids.append(tid)
            selected_modules.append(mod)
            selected_types.append(ttype)
            selection_scores.append(score)

        self.selector_output = {
            "selected_indices": selected_indices,
            "selected_test_ids": selected_test_ids,
            "selected_modules": selected_modules,
            "selected_types": selected_types,
            "selection_scores": selection_scores,
            "method_used": full_ranking["method_used"],
            "budget": len(selected_indices),
        }
        self.budget = len(selected_indices)
        return {
            "method_used": self.selector_output["method_used"],
            "n_selected": len(selected_indices),
            "total_seconds": round(cumulative, 1),
            "time_budget_seconds": time_budget_seconds,
            "top_5_test_ids": selected_test_ids[:5],
            "selected_modules_distribution": dict(Counter(selected_modules)),
        }

    # ── Scenario 3: confidence-based substitution (Phase 2.5) ──

    def get_selection_confidence_report(self, low_confidence_threshold: float = 0.15) -> Dict:
        """
        For a PPO-based selection, recompute per-step action probabilities by
        replaying the same masked greedy loop _select_with_ppo() already ran
        (cheap, sub-second for 200 tests) — does not modify _select_with_ppo()
        itself. Flags a step only if its action probability is both below
        threshold AND not meaningfully above what picking uniformly among the
        remaining valid actions would give (this policy's probabilities run
        uniformly low in a 200-way space, so threshold alone would over-flag).
        """
        if self.selector_output is None:
            raise ValueError("Call select_tests first")
        if self.selector_output["method_used"] != "PPO (VRTQ-RL)":
            return {"available": False, "method_used": self.selector_output["method_used"],
                    "flagged": [], "threshold": low_confidence_threshold}

        import numpy as np
        from environment.state_builder import apply_selection_mask

        model = self._selector.model
        state_matrix = self.scorer_output["state_matrix"]
        n_tests = len(self.test_suite_df)
        current_state = state_matrix.copy()
        selected: List[int] = []
        flagged = []

        for step, action in enumerate(self.selector_output["selected_indices"]):
            obs = current_state.flatten().astype(np.float32)
            mask = np.ones(n_tests, dtype=bool)
            mask[selected] = False

            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            dist = model.policy.get_distribution(obs_tensor, action_masks=mask)
            probs = dist.distribution.probs[0]
            action_prob = float(probs[action].item())
            n_remaining_valid = int(mask.sum())
            uniform_baseline = 1.0 / max(1, n_remaining_valid)

            if action_prob < low_confidence_threshold and action_prob < uniform_baseline * 1.5:
                flagged.append({
                    "test_id": self.selector_output["selected_test_ids"][step],
                    "step": step,
                    "action_prob": round(action_prob, 4),
                    "uniform_baseline": round(uniform_baseline, 4),
                    "n_remaining_valid": n_remaining_valid,
                })

            selected.append(action)
            current_state = apply_selection_mask(current_state, selected)

        return {"available": True, "method_used": "PPO (VRTQ-RL)",
                "flagged": flagged, "threshold": low_confidence_threshold}

    def substitute_test_with_vrtq_heuristic(self, test_id_to_replace: str) -> Dict:
        """
        Replace one test in the current selection with the next-best
        VRTQ-ranked test not already selected, restricted to affected
        modules if any (else whole-suite). Preserves budget size. Pure list
        surgery — reuses VRTQHeuristicSelector.compute_composite, already
        used elsewhere as the deterministic fallback, not a new dependency.
        """
        if self.selector_output is None:
            raise ValueError("Call select_tests first")
        if test_id_to_replace not in self.selector_output["selected_test_ids"]:
            raise ValueError(f"{test_id_to_replace} is not in the current selection")

        from rl.baselines.vrtq_heuristic import VRTQHeuristicSelector

        pos = self.selector_output["selected_test_ids"].index(test_id_to_replace)
        selected_set = set(self.selector_output["selected_indices"])

        affected = set(self.diff_features.get("modules_affected", [])) if self.diff_features else set()
        mask = ~self.test_suite_df.index.isin(selected_set)
        if affected:
            module_mask = mask & self.test_suite_df["module"].isin(affected)
            if module_mask.any():
                mask = module_mask

        candidates = self.test_suite_df[mask]
        scores = VRTQHeuristicSelector().compute_composite(candidates)
        new_idx = int(scores.idxmax())
        new_row = self.test_suite_df.loc[new_idx]

        self.selector_output["selected_indices"][pos] = new_idx
        self.selector_output["selected_test_ids"][pos] = new_row["test_id"]
        self.selector_output["selected_modules"][pos] = new_row["module"]
        self.selector_output["selected_types"][pos] = new_row["test_type"]
        self.selector_output["selection_scores"][pos] = float(new_row["vrtq_composite"])

        return {
            "replaced_test_id": test_id_to_replace,
            "new_test_id": new_row["test_id"],
            "new_module": new_row["module"],
        }
