"""
agents/report_agent.py
-----------------------
ReportAgent: Generates human-readable prioritization report.

Uses LLM for plain-English summary only (one focused call).
All metrics computed deterministically — no hallucination risk on numbers.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import os
import json
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class ReportAgent:
    """
    AutoGen-style agent that generates the final prioritization report.

    Structure:
    - Metrics: computed deterministically (no LLM)
    - Top tests: ranked list with rationale
    - Plain-English summary: one LLM call (optional)
    """

    def __init__(self, use_llm: bool = True):
        self.name = "ReportAgent"
        self.use_llm = use_llm

        if use_llm:
            self._init_autogen()

    def _init_autogen(self):
        """Initialize AutoGen AssistantAgent for summary generation."""
        try:
            import autogen
            config = [{
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", ""),
            }]
            self.agent = autogen.AssistantAgent(
                name=self.name,
                llm_config={"config_list": config, "max_tokens": 200},
                system_message=(
                    "You are a QA intelligence system. Given test prioritization data, "
                    "write a 3-sentence plain-English summary for a QA engineer. "
                    "Be specific about which modules are at risk and why. "
                    "Do not repeat numbers already in the data — add insight."
                ),
            )
        except Exception:
            self.use_llm = False

    def generate(
        self,
        selector_output: Dict,
        scorer_output: Dict,
        test_suite_df: pd.DataFrame,
        budget: int = 50,
    ) -> Dict:
        """
        Generate complete prioritization report.

        Args:
            selector_output: Output from TestSelectorAgent.select()
            scorer_output: Output from RiskScorerAgent.score()
            test_suite_df: Full test suite DataFrame
            budget: Tests budget

        Returns:
            Full report dict with metrics, top tests, and summary
        """
        selected_indices = selector_output["selected_indices"]
        selected_df = test_suite_df.iloc[selected_indices].reset_index(drop=True)
        risk_summary = scorer_output["risk_summary"]
        git_context = scorer_output["git_context"]

        # Compute metrics
        metrics = self._compute_metrics(selected_df, test_suite_df, budget)

        # Build top-10 test details
        top_tests = self._build_top_tests(
            selected_df.head(10),
            selector_output["selection_scores"][:10],
            git_context,
        )

        # Generate plain-English summary
        summary = self._generate_summary(
            metrics, risk_summary, git_context, top_tests
        )

        report = {
            "method": selector_output["method_used"],
            "metrics": metrics,
            "top_tests": top_tests,
            "risk_summary": risk_summary,
            "git_context": {
                "modules_affected": git_context.get("modules_affected", []),
                "change_type": git_context.get("change_type", "unknown"),
                "churn_score": git_context.get("churn_score", 0.0),
            },
            "summary": summary,
            "budget": budget,
            "total_tests": len(test_suite_df),
        }

        print(f"[{self.name}] Report generated | "
              f"FDR@50%={metrics['fdr_50']:.1%} | "
              f"TTFF={metrics['ttff']} | "
              f"Method={selector_output['method_used']}")

        return report

    def _compute_metrics(
        self,
        selected_df: pd.DataFrame,
        full_df: pd.DataFrame,
        budget: int,
    ) -> Dict:
        """Compute FDR, TTFF, TSR deterministically."""
        total_faults = int(full_df["has_fault"].sum())
        faults_found = int(selected_df["has_fault"].sum())

        k25 = max(1, int(len(selected_df) * 0.25))
        k50 = max(1, int(len(selected_df) * 0.50))

        fdr_25 = int(selected_df.iloc[:k25]["has_fault"].sum()) / max(1, total_faults)
        fdr_50 = int(selected_df.iloc[:k50]["has_fault"].sum()) / max(1, total_faults)
        fdr_100 = faults_found / max(1, total_faults)

        fault_mask = selected_df["has_fault"].values
        ttff = int(np.argmax(fault_mask)) + 1 if fault_mask.any() else budget + 1

        cumulative = selected_df["has_fault"].cumsum().values
        target = int(0.8 * total_faults)
        tsr_idx = np.where(cumulative >= target)[0]
        tests_for_80 = int(tsr_idx[0]) + 1 if len(tsr_idx) > 0 else budget
        tsr = 1.0 - (tests_for_80 / len(full_df))

        time_saved = full_df["execution_time_seconds"].sum() - selected_df["execution_time_seconds"].sum()

        return {
            "fdr_25": round(fdr_25, 4),
            "fdr_50": round(fdr_50, 4),
            "fdr_100": round(fdr_100, 4),
            "ttff": ttff,
            "tsr": round(tsr, 4),
            "faults_found": faults_found,
            "total_faults": total_faults,
            "tests_run": len(selected_df),
            "total_tests": len(full_df),
            "estimated_time_saved_seconds": round(time_saved, 1),
        }

    def _build_top_tests(
        self,
        top_df: pd.DataFrame,
        scores: List[float],
        git_context: Dict,
    ) -> List[Dict]:
        """Build structured list of top prioritized tests with rationale."""
        affected = set(git_context.get("modules_affected", []))
        result = []

        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            reasons = []
            if row["module"] in affected:
                reasons.append("in changed module")
            if row["vrtq_risk_score"] > 0.7:
                reasons.append(f"high risk ({row['vrtq_risk_score']:.2f})")
            if row["historical_failure_rate"] > 0.2:
                reasons.append(f"historical failure rate {row['historical_failure_rate']:.0%}")
            if row["days_since_last_run"] > 7:
                reasons.append(f"not run in {row['days_since_last_run']} days")

            result.append({
                "rank": rank,
                "test_id": row["test_id"],
                "test_name": row["test_name"],
                "module": row["module"],
                "test_type": row["test_type"],
                "vrtq_composite": round(float(row["vrtq_composite"]), 3),
                "selection_score": round(float(scores[rank - 1]), 3) if rank <= len(scores) else 0.0,
                "rationale": ", ".join(reasons) if reasons else "VRTQ score",
                "execution_time_seconds": row["execution_time_seconds"],
            })

        return result

    def _generate_summary(
        self,
        metrics: Dict,
        risk_summary: Dict,
        git_context: Dict,
        top_tests: List[Dict],
    ) -> str:
        """Generate plain-English summary. Uses LLM if available."""
        if self.use_llm and hasattr(self, "agent"):
            try:
                import autogen
                proxy = autogen.UserProxyAgent(
                    name="proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False,
                )
                prompt = (
                    f"Summarize this test prioritization result for a QA engineer:\n"
                    f"- Changed modules: {risk_summary['modules_affected']}\n"
                    f"- Change type: {git_context.get('change_type')}\n"
                    f"- Tests selected: {metrics['tests_run']} of {metrics['total_tests']}\n"
                    f"- FDR@50%: {metrics['fdr_50']:.1%}\n"
                    f"- Time to first failure: test #{metrics['ttff']}\n"
                    f"- High-risk tests: {risk_summary['high_risk_tests']}\n"
                    f"- Top test: {top_tests[0]['test_name'] if top_tests else 'N/A'}\n"
                )
                proxy.initiate_chat(
                    self.agent,
                    message=prompt,
                    silent=True,
                )
                last = proxy.last_message()
                if last and last.get("content"):
                    return last["content"].strip()
            except Exception:
                pass

        # Deterministic fallback summary
        modules = risk_summary.get("modules_affected", [])
        change = git_context.get("change_type", "change")
        return (
            f"This {change} affects {len(modules)} module(s) "
            f"({', '.join(modules[:2])}{'...' if len(modules) > 2 else ''}). "
            f"VRTQ-RL selected {metrics['tests_run']} tests, "
            f"projected to detect {metrics['fdr_50']:.0%} of faults at the 50% mark "
            f"with first failure expected at test #{metrics['ttff']}. "
            f"Estimated time saved: "
            f"{metrics['estimated_time_saved_seconds']:.0f}s vs running the full suite."
        )

    def print_report(self, report: Dict) -> None:
        """Pretty-print report to console."""
        print("\n" + "="*60)
        print("VRTQ-RL PRIORITIZATION REPORT")
        print("="*60)
        print(f"Method: {report['method']}")
        print(f"Tests: {report['metrics']['tests_run']} / {report['metrics']['total_tests']}")
        print(f"\nMetrics:")
        m = report["metrics"]
        print(f"  FDR@25%:  {m['fdr_25']:.1%}")
        print(f"  FDR@50%:  {m['fdr_50']:.1%}")
        print(f"  FDR@100%: {m['fdr_100']:.1%}")
        print(f"  TTFF:     Test #{m['ttff']}")
        print(f"  TSR:      {m['tsr']:.1%}")
        print(f"  Time saved: {m['estimated_time_saved_seconds']:.0f}s")
        print(f"\nTop 5 Tests:")
        for t in report["top_tests"][:5]:
            print(f"  {t['rank']}. {t['test_id']} | {t['module']} | "
                  f"VRTQ={t['vrtq_composite']:.3f} | {t['rationale']}")
        print(f"\nSummary:\n{report['summary']}")
        print("="*60)


if __name__ == "__main__":
    from data.fault_injection import create_training_dataset
    from agents.change_analyzer_agent import ChangeAnalyzerAgent, simulate_git_diff
    from agents.risk_scorer_agent import RiskScorerAgent
    from agents.test_selector_agent import TestSelectorAgent

    df = create_training_dataset()
    diff = simulate_git_diff(modules=["payment_service", "auth_service"], churn="medium")

    analyzer = ChangeAnalyzerAgent(use_llm=False)
    scorer = RiskScorerAgent()
    selector = TestSelectorAgent()
    reporter = ReportAgent(use_llm=False)

    features = analyzer.analyze(diff)
    scored = scorer.score(features, df)
    selected = selector.select(scored, df, budget=50)
    report = reporter.generate(selected, scored, df, budget=50)

    reporter.print_report(report)
