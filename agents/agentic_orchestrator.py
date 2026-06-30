"""
agents/agentic_orchestrator.py
---------------------------------
Opt-in "agentic mode": a Supervisor agent drives the same four pipeline
stages as agents/orchestrator.py::run_pipeline() via real AutoGen tool
calls, with a Critic agent that can request one re-run with adjusted
parameters. Falls back to the plain deterministic run_pipeline() on any
missing config, error, or timeout — never raises to the caller.

This is strictly additive: run_pipeline() itself is never modified, and
this module is never imported by the RL evaluation scripts.

Usage:
    python -m agents.agentic_orchestrator

Author: Anthony Vallente
Project: VRTQ-RL
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

TEST_BUDGET = int(os.getenv("TEST_BUDGET", 50))
MAX_CRITIC_ROUNDS = 1
TIMEOUT_SECONDS = 90


def run_agentic_pipeline(
    diff_input,
    test_suite_df=None,
    budget: int = TEST_BUDGET,
    max_critic_rounds: int = MAX_CRITIC_ROUNDS,
    timeout_seconds: int = TIMEOUT_SECONDS,
    verbose: bool = True,
) -> Dict:
    """
    Run the pipeline in agentic mode. Returns the same report dict shape as
    agents.orchestrator.run_pipeline(), plus:
      - agentic_trace: list of {role, content, tool_call} transcript entries
      - fallback_used: bool
      - fallback_reason: str | None
      - rounds_used: int (Critic-triggered re-runs actually performed)
    """
    from agents.orchestrator import run_pipeline
    from agents.agentic.llm_config import get_llm_config

    if test_suite_df is None:
        from data.fault_injection import create_training_dataset
        test_suite_df = create_training_dataset()

    diff_text = diff_input if isinstance(diff_input, str) else str(diff_input)

    llm_config = get_llm_config()
    if llm_config is None:
        if verbose:
            print("[agentic_orchestrator] OPENAI_API_KEY not configured — falling back to deterministic pipeline")
        report = run_pipeline(diff_input, test_suite_df=test_suite_df, budget=budget,
                               use_llm=False, verbose=verbose)
        return {**report, "agentic_trace": [], "fallback_used": True,
                "fallback_reason": "OPENAI_API_KEY not configured", "rounds_used": 0}

    from agents.agentic.tools import PipelineToolkit
    from agents.agentic.conversation import run_conversation

    toolkit = PipelineToolkit(test_suite_df, budget, diff_text)

    def _run():
        return run_conversation(toolkit, llm_config, max_critic_rounds=max_critic_rounds)

    try:
        if verbose:
            print(f"\n{'='*50}")
            print("VRTQ-RL: Agentic Pipeline (Supervisor + Critic)")
            print(f"{'='*50}")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as pool:
            conv_result = pool.submit(_run).result(timeout=timeout_seconds)
        elapsed = time.time() - t0

        if toolkit.report is None:
            raise RuntimeError("Agentic conversation ended without producing a report")

        if verbose:
            print(f"[agentic_orchestrator] Conversation complete in {elapsed:.1f}s, "
                  f"{conv_result['rounds_used']} Critic-triggered re-run(s)")

        return {
            **toolkit.report,
            "agentic_trace": conv_result["transcript"],
            "fallback_used": False,
            "fallback_reason": None,
            "rounds_used": conv_result["rounds_used"],
        }

    except FutureTimeoutError:
        reason = f"Agentic conversation exceeded {timeout_seconds}s timeout"
        if verbose:
            print(f"[agentic_orchestrator] {reason} — falling back to deterministic pipeline")
    except Exception as e:
        reason = f"Agentic conversation failed: {e}"
        if verbose:
            print(f"[agentic_orchestrator] {reason} — falling back to deterministic pipeline")

    report = run_pipeline(diff_input, test_suite_df=test_suite_df, budget=budget,
                           use_llm=False, verbose=verbose)
    return {**report, "agentic_trace": [], "fallback_used": True,
            "fallback_reason": reason, "rounds_used": 0}


if __name__ == "__main__":
    from agents.change_analyzer_agent import simulate_git_diff

    diff = simulate_git_diff(modules=["payment_service", "auth_service"], churn="medium")
    report = run_agentic_pipeline(diff_input=diff, budget=50, verbose=True)

    print(f"\nfallback_used: {report['fallback_used']}")
    if report["fallback_used"]:
        print(f"fallback_reason: {report['fallback_reason']}")
    else:
        print(f"rounds_used: {report['rounds_used']}")
        print(f"trace length: {len(report['agentic_trace'])} messages")
    print(f"\nMetrics: {report['metrics']}")
