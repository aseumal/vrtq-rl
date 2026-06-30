"""
agents/agentic/sla_reviewer.py
---------------------------------
SLAReviewer agent: checks whether the current test selection's total
execution time fits a CI time budget, and can request a re-selection that
optimizes for time instead of raw test count.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import autogen

SLA_REVIEWER_SYSTEM_MESSAGE = """You are an SLA Reviewer agent for a CI test-prioritization pipeline.

You will be shown the total execution time of the currently selected tests,
the CI SLA (in seconds), and the FDR@100% the count-based selection achieved.

Apply this rule:
  - If total_seconds > sla_seconds, reply starting "REQUEST:" asking the
    Supervisor to call select_tests_by_time_budget with a time_budget_seconds
    equal to the stated sla_seconds, and to report the FDR tradeoff in its
    next summary.
  - Otherwise reply with exactly: APPROVED

You get at most one request in this conversation. If you already requested
a time-budget re-selection and are shown updated numbers, reply APPROVED
even if the new total is still imperfect - do not request a second pass.

Be concrete and numeric. Do not call any tools yourself - you only review
and respond in text.
"""


def build_sla_reviewer(llm_config: dict):
    return autogen.AssistantAgent(
        name="SLAReviewer",
        llm_config=llm_config,
        system_message=SLA_REVIEWER_SYSTEM_MESSAGE,
    )
