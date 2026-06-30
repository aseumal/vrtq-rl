"""
agents/agentic/supervisor.py
-------------------------------
Supervisor agent: an AutoGen AssistantAgent that drives the pipeline via
real tool/function calls, paired with a UserProxyAgent executor that
actually runs the Python functions. Ten tools total: the original four
pipeline stages, plus six added for genuine agent-to-agent negotiation
(module-ambiguity, execution-time SLA, and RL-confidence scenarios).

Author: Anthony Seumal
Project: VRTQ-RL
"""

import autogen

from agents.agentic.tools import PipelineToolkit

SUPERVISOR_SYSTEM_MESSAGE = """You are the Supervisor agent for a test-prioritization pipeline.

Your tools, grouped by when they're used:

PHASE 0 - module classification:
  1. analyze_diff() - parse the currently loaded git diff (no arguments)
  2. get_module_ambiguity_report() - check for ambiguous file-to-module matches
  3. reexamine_module_match(file, module_to_confirm_or_retract, keep) - apply a
     RiskReviewer-requested correction (only call if asked)

PHASE 1 - scoring + selection (the normal "run the pipeline" path):
  4. score_risk() - compute VRTQ risk summary
  5. select_tests(budget, focus_modules) - run RL-based test selection
  6. get_report_metrics() - compute FDR/TTFF/TSR metrics

PHASE 2/2.5 - negotiation tools (only call if a reviewer's feedback asks you to):
  7. get_selection_time_budget_check(sla_seconds) - check execution time vs an SLA ceiling
  8. select_tests_by_time_budget(time_budget_seconds, focus_modules) - re-select truncated by time, not count
  9. get_selection_confidence_report(low_confidence_threshold) - check the policy's per-pick confidence
  10. substitute_test_with_vrtq_heuristic(test_id_to_replace) - swap one low-confidence pick for a heuristic-ranked one

Call exactly one tool per turn, and never call the same tool twice in a row -
each tool's result is already final the moment it returns, there is no need
to call it again to "confirm" the result.

When asked to "run the full pipeline", call tools 1, 4, 5, 6 in exactly that
order. When a reviewer's feedback message asks you to apply a specific
correction, call ONLY the tool(s) it names - do not re-run earlier phases
unless explicitly asked.

The instant the last tool call needed for your current task returns, your
very next message must start with the literal text "SUMMARY:" followed by a
2-3 sentence plain-English summary, and you must not call any more tools
after that.

Never invent metrics or test IDs - only report numbers actually returned
by your tools.
"""


def build_supervisor(llm_config: dict, toolkit: PipelineToolkit):
    """Returns (supervisor, executor) with the toolkit's ten methods registered as tools."""
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

    @supervisor.register_for_llm(name="get_module_ambiguity_report", description="Check the analyzed diff for files matching 2+ modules' keyword lists. Call after analyze_diff, before score_risk.")
    @executor.register_for_execution(name="get_module_ambiguity_report")
    def get_module_ambiguity_report() -> dict:
        return toolkit.get_module_ambiguity_report()

    @supervisor.register_for_llm(name="reexamine_module_match", description="Keep or retract one ambiguous file-to-module match. Call only if RiskReviewer challenges a specific match.")
    @executor.register_for_execution(name="reexamine_module_match")
    def reexamine_module_match(file: str, module_to_confirm_or_retract: str, keep: bool) -> dict:
        return toolkit.reexamine_module_match(file, module_to_confirm_or_retract, keep)

    @supervisor.register_for_llm(name="score_risk", description="Compute the VRTQ risk summary for the analyzed diff. Call after analyze_diff (and after any module-ambiguity corrections).")
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

    @supervisor.register_for_llm(name="get_selection_time_budget_check", description="Check the current selection's total execution time against a CI SLA ceiling (seconds). Call after select_tests.")
    @executor.register_for_execution(name="get_selection_time_budget_check")
    def get_selection_time_budget_check(sla_seconds: int = 1200) -> dict:
        return toolkit.get_selection_time_budget_check(sla_seconds=sla_seconds)

    @supervisor.register_for_llm(name="select_tests_by_time_budget", description="Re-select tests truncated by cumulative execution time instead of count. Call only if SLAReviewer requests it.")
    @executor.register_for_execution(name="select_tests_by_time_budget")
    def select_tests_by_time_budget(time_budget_seconds: int, focus_modules: list = None) -> dict:
        return toolkit.select_tests_by_time_budget(time_budget_seconds=time_budget_seconds, focus_modules=focus_modules)

    @supervisor.register_for_llm(name="get_selection_confidence_report", description="Check the PPO policy's action-probability confidence on each selected test. Call after select_tests.")
    @executor.register_for_execution(name="get_selection_confidence_report")
    def get_selection_confidence_report(low_confidence_threshold: float = 0.15) -> dict:
        return toolkit.get_selection_confidence_report(low_confidence_threshold=low_confidence_threshold)

    @supervisor.register_for_llm(name="substitute_test_with_vrtq_heuristic", description="Replace one low-confidence test pick with the next-best VRTQ-heuristic-ranked test. Call only if ConfidenceReviewer requests it.")
    @executor.register_for_execution(name="substitute_test_with_vrtq_heuristic")
    def substitute_test_with_vrtq_heuristic(test_id_to_replace: str) -> dict:
        return toolkit.substitute_test_with_vrtq_heuristic(test_id_to_replace)

    return supervisor, executor
