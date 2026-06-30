"""
agents/agentic/risk_reviewer.py
----------------------------------
RiskReviewer agent: challenges ambiguous module-mapping decisions from
ChangeAnalyzerAgent BEFORE RiskScorerAgent consumes them. Unlike the
Critic (which reviews a finished result), this is a pre-emptive challenge
to a classification decision — genuinely different agents disagreeing on
what the diff even means, not just whether the final numbers look good.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import autogen

RISK_REVIEWER_SYSTEM_MESSAGE = """You are a Risk Reviewer agent checking a ChangeAnalyzer's module classification before risk scoring runs.

You will be shown a list of ambiguous files: each file matched 2+ modules'
keyword lists (e.g. a filename containing both "payment" and "auth").

For each ambiguous file, decide whether BOTH matched modules genuinely
belong in modules_affected, or whether one is a false positive (e.g. the
keyword appears in the filename but the change is clearly scoped to one
module based on the file's primary name component before the first
underscore/path segment).

If you believe a specific module match is a false positive for a specific
file, reply starting "REQUEST:" naming exactly one file and exactly one
module to retract, with a one-sentence reason.
Otherwise reply with exactly: APPROVED

You get at most one challenge per ambiguous file, and at most TWO challenges
total in this conversation regardless of how many ambiguous files exist.
After your challenge(s) are addressed, reply APPROVED even with reservations.

Be concrete - name the exact file and module. Do not call any tools
yourself - you only review and respond in text.
"""


def build_risk_reviewer(llm_config: dict):
    return autogen.AssistantAgent(
        name="RiskReviewer",
        llm_config=llm_config,
        system_message=RISK_REVIEWER_SYSTEM_MESSAGE,
    )
