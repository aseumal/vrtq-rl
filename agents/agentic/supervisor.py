"""
agents/agentic/supervisor.py
-------------------------------
Supervisor agent: an AutoGen AssistantAgent that drives the pipeline via
real tool/function calls (analyze_diff -> score_risk -> select_tests ->
get_report_metrics), paired with a UserProxyAgent executor that actually
runs the Python functions.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import autogen

from agents.agentic.tools import PipelineToolkit

SUPERVISOR_SYSTEM_MESSAGE = """You are the Supervisor agent for a test-prioritization pipeline.

You have four tools, which MUST be called in this order on first pass:
  1. analyze_diff() - parse the currently loaded git diff (no arguments needed)
  2. score_risk() - compute VRTQ risk summary
  3. select_tests(budget, focus_modules) - run RL-based test selection
  4. get_report_metrics() - compute final FDR/TTFF/TSR metrics

Call exactly one tool per turn, and never call the same tool twice in a row -
each tool's result is already final the moment it returns, there is no need
to call it again to "confirm" the result. The instant get_report_metrics()
returns, your very next message must start with the literal text "SUMMARY:"
followed by a 2-3 sentence plain-English summary of the metrics it returned,
and you must not call any more tools after that.

Never invent metrics or test IDs - only report numbers actually returned
by your tools.
"""


def build_supervisor(llm_config: dict, toolkit: PipelineToolkit):
    """Returns (supervisor, executor) with the toolkit's four methods registered as tools."""
    supervisor = autogen.AssistantAgent(
        name="Supervisor",
        llm_config=llm_config,
        system_message=SUPERVISOR_SYSTEM_MESSAGE,
    )
    executor = autogen.UserProxyAgent(
        name="ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        is_termination_msg=lambda m: (
            isinstance(m.get("content"), str) and m["content"].strip().startswith("SUMMARY:")
        ),
    )

    @supervisor.register_for_llm(name="analyze_diff", description="Parse the currently loaded git diff into structured change features (modules, churn, change type). Takes no arguments.")
    @executor.register_for_execution(name="analyze_diff")
    def analyze_diff() -> dict:
        return toolkit.analyze_diff()

    @supervisor.register_for_llm(name="score_risk", description="Compute the VRTQ risk summary for the analyzed diff. Call after analyze_diff.")
    @executor.register_for_execution(name="score_risk")
    def score_risk() -> dict:
        return toolkit.score_risk()

    @supervisor.register_for_llm(name="select_tests", description="Run RL test selection. Call after score_risk. Re-callable with a different budget or focus_modules to refine results.")
    @executor.register_for_execution(name="select_tests")
    def select_tests(budget: int = None, focus_modules: list = None) -> dict:
        return toolkit.select_tests(budget=budget, focus_modules=focus_modules)

    @supervisor.register_for_llm(name="get_report_metrics", description="Generate final FDR/TTFF/TSR metrics from the current selection. Call after select_tests.")
    @executor.register_for_execution(name="get_report_metrics")
    def get_report_metrics() -> dict:
        return toolkit.get_report_metrics()

    return supervisor, executor
