"""
agents/agentic/critic.py
---------------------------
Critic agent: reviews the Supervisor's reported metrics and can request a
re-run with adjusted parameters. This is the genuinely autonomous decision
in the pipeline — whether to approve or push back isn't predetermined by a
fixed script, it's the LLM reasoning over the actual numbers it's shown.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import autogen

CRITIC_SYSTEM_MESSAGE = """You are a QA Critic agent reviewing a test-prioritization result.

You will be shown metrics from a test-selection run: FDR@25%, FDR@50%,
FDR@100% (fraction of known faults caught at each point in the budget),
TTFF (position of the first fault-revealing test, lower is better), TSR,
budget used, and the module distribution of selected tests.

Apply these concrete thresholds:
  - If FDR@50% < 0.30, the selection is POOR at sustained fault-finding.
    Reply with a message starting "REQUEST:" asking for the budget to be
    increased by 50% (state the exact new number).
  - Else if one module accounts for more than 70% of selected tests while
    the diff affects more than one module, reply "REQUEST:" asking for
    focus_modules to be set to the underrepresented affected module(s)
    (name them explicitly).
  - Otherwise, reply with exactly: APPROVED

You get at most one re-run request in this conversation. If you already
made one request and are shown a second set of metrics, reply APPROVED
regardless (even with reservations) - do not request a third pass.

Be concrete and numeric. Do not call any tools yourself - you only review
and respond in text.
"""


def build_critic(llm_config: dict):
    return autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message=CRITIC_SYSTEM_MESSAGE,
    )
