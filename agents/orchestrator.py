"""
agents/orchestrator.py
-----------------------
VRTQ-RL Orchestrator: Runs all four agents in sequence (Option B).

Direct sequential calling — no GroupChat overhead.
Each agent receives only what it needs, minimizing token usage.

Usage:
    python -m agents.orchestrator
    python -m agents.orchestrator --modules payment_service auth_service --churn high

Author: Anthony Seumal
Project: VRTQ-RL
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

TEST_BUDGET = int(os.getenv("TEST_BUDGET", 50))


def run_pipeline(
    diff_input,
    test_suite_df=None,
    budget: int = TEST_BUDGET,
    use_llm: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run the full VRTQ-RL agent pipeline.

    Flow:
        diff → ChangeAnalyzerAgent
             → RiskScorerAgent
             → TestSelectorAgent
             → ReportAgent
             → Final report

    Args:
        diff_input: Raw git diff string OR dict with diff features
        test_suite_df: Test suite DataFrame. If None, generates fresh dataset.
        budget: Max tests to select
        use_llm: Whether to use LLM for classification + summary
        verbose: Print step-by-step progress

    Returns:
        Complete report dictionary
    """
    from agents.change_analyzer_agent import ChangeAnalyzerAgent
    from agents.risk_scorer_agent import RiskScorerAgent
    from agents.test_selector_agent import TestSelectorAgent
    from agents.report_agent import ReportAgent

    if test_suite_df is None:
        from data.fault_injection import create_training_dataset
        test_suite_df = create_training_dataset()

    if verbose:
        print("\n" + "="*60)
        print("VRTQ-RL PIPELINE STARTING")
        print("="*60)

    # Step 1: Analyze git diff
    if verbose:
        print("\n[Step 1/4] ChangeAnalyzerAgent — parsing diff...")
    analyzer = ChangeAnalyzerAgent(use_llm=use_llm)
    diff_features = analyzer.analyze(diff_input)

    # Step 2: Score risks using VRTQ
    if verbose:
        print("\n[Step 2/4] RiskScorerAgent — computing VRTQ state vector...")
    scorer = RiskScorerAgent()
    scorer_output = scorer.score(diff_features, test_suite_df)

    # Step 3: Select tests using PPO model
    if verbose:
        print("\n[Step 3/4] TestSelectorAgent — running RL inference...")
    selector = TestSelectorAgent()
    selector_output = selector.select(scorer_output, test_suite_df, budget=budget)

    # Step 4: Generate report
    if verbose:
        print("\n[Step 4/4] ReportAgent — generating report...")
    reporter = ReportAgent(use_llm=use_llm)
    report = reporter.generate(selector_output, scorer_output, test_suite_df, budget=budget)

    if verbose:
        reporter.print_report(report)

    return report


def save_report(report: Dict, path: str = "evaluation/last_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy types for JSON serialization
    def convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(report), f, indent=2)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VRTQ-RL pipeline")
    parser.add_argument("--modules", nargs="+", default=["payment_service"],
                        help="Modules to simulate as changed")
    parser.add_argument("--churn", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--budget", type=int, default=TEST_BUDGET)
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LLM calls (requires OPENAI_API_KEY)")
    parser.add_argument("--agentic", action="store_true",
                        help="Use the opt-in agentic mode (Supervisor+Critic AutoGen "
                             "conversation, real tool calls) instead of the plain "
                             "sequential pipeline. Requires OPENAI_API_KEY; falls back "
                             "to the deterministic pipeline if not configured.")
    parser.add_argument("--save", action="store_true",
                        help="Save report to evaluation/last_report.json")
    args = parser.parse_args()

    from agents.change_analyzer_agent import simulate_git_diff
    diff = simulate_git_diff(modules=args.modules, churn=args.churn)

    if args.agentic:
        from agents.agentic_orchestrator import run_agentic_pipeline
        report = run_agentic_pipeline(
            diff_input=diff,
            budget=args.budget,
            verbose=True,
        )
    else:
        report = run_pipeline(
            diff_input=diff,
            budget=args.budget,
            use_llm=args.use_llm,
            verbose=True,
        )

    if args.save:
        save_report(report)
