"""
agents/agentic/conversation.py
---------------------------------
Orchestrates the full agentic conversation across five phases:

  Phase 0   - analyze_diff + module-ambiguity check, RiskReviewer can
              challenge a classification BEFORE score_risk consumes it
              (state-matrix dependency: must resolve before scoring).
  Phase 1   - score_risk -> select_tests (count-based) -> get_report_metrics.
  Phase 2   - execution-time SLA check; SLAReviewer can request a
              time-budgeted re-selection.
  Phase 2.5 - RL policy confidence check; ConfidenceReviewer can request a
              single low-confidence-pick substitution.
  Phase 3   - Critic reviews final outcome quality (FDR/TTFF/distribution),
              now SLA-aware so a budget-increase request must be justified
              against the same time ceiling Phase 2 uses.

Every reviewer sub-chat is a neutral, executor-less proxy talking to a
tool-less reviewer persona (never the tool-bearing Supervisor) so a
reviewer can never attempt a tool call with nothing able to run it. Every
phase's round cap is enforced by an independent hard Python counter,
regardless of what the LLM says — this was empirically necessary: the
Critic has been observed asking for a re-run beyond its stated cap.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import json
from collections import Counter
from typing import Dict, List

import autogen

from agents.agentic.supervisor import build_supervisor
from agents.agentic.critic import build_critic
from agents.agentic.risk_reviewer import build_risk_reviewer
from agents.agentic.sla_reviewer import build_sla_reviewer
from agents.agentic.confidence_reviewer import build_confidence_reviewer
from agents.agentic.tools import PipelineToolkit


def _normalize_messages(chat_history: List[Dict], default_sender: str) -> List[Dict]:
    """Flatten an AutoGen chat_history into a simple, JSON-safe transcript."""
    out = []
    for m in chat_history:
        sender = m.get("name") or default_sender
        tool_calls = m.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except (TypeError, ValueError):
                    pass
                out.append({"role": sender, "content": None,
                            "tool_call": {"name": fn.get("name"), "args": args}})
        elif m.get("content"):
            out.append({"role": sender, "content": m["content"], "tool_call": None})
    return out


def _build_ambiguity_prompt(ambiguous_files: List[Dict]) -> str:
    lines = ["Review these ambiguous module classifications:"]
    for af in ambiguous_files:
        lines.append(
            f"- File: {af['file']} | Candidate modules: {af['candidate_modules']} "
            f"| Matched keywords: {af['matched_keywords']}"
        )
    return "\n".join(lines)


def _build_sla_prompt(toolkit: PipelineToolkit, time_check: Dict) -> str:
    metrics = toolkit.report["metrics"]
    return (
        "Review this selection's execution-time budget:\n"
        f"- Total execution time: {time_check['total_seconds']}s\n"
        f"- CI SLA ceiling: {time_check['sla_seconds']}s\n"
        f"- Over SLA: {time_check['over_sla']}\n"
        f"- Tests selected: {time_check['n_selected']}\n"
        f"- FDR@100% achieved: {metrics['fdr_100']:.1%}\n"
    )


def _build_confidence_prompt(confidence_report: Dict) -> str:
    if not confidence_report["available"]:
        return f"Selection used {confidence_report['method_used']} (not PPO) - no confidence data available."
    lines = [f"Review the PPO policy's confidence (threshold={confidence_report['threshold']}):"]
    if not confidence_report["flagged"]:
        lines.append("No picks flagged as low-confidence.")
    for f in confidence_report["flagged"]:
        lines.append(
            f"- {f['test_id']} (step {f['step']}): action_prob={f['action_prob']}, "
            f"uniform_baseline={f['uniform_baseline']}, n_remaining_valid={f['n_remaining_valid']}"
        )
    return "\n".join(lines)


def _build_review_prompt(toolkit: PipelineToolkit, sla_seconds: int) -> str:
    metrics = toolkit.report["metrics"]
    distribution = dict(Counter(toolkit.selector_output["selected_modules"]))
    modules_affected = toolkit.diff_features.get("modules_affected", [])
    # Python-level enrichment, not an LLM tool call - the Critic needs this
    # context to reason about budget tradeoffs, but fetching it isn't itself
    # a Supervisor decision.
    time_check = toolkit.get_selection_time_budget_check(sla_seconds=sla_seconds)
    return (
        "Review this test-selection result:\n"
        f"- FDR@25%: {metrics['fdr_25']:.1%}\n"
        f"- FDR@50%: {metrics['fdr_50']:.1%}\n"
        f"- FDR@100%: {metrics['fdr_100']:.1%}\n"
        f"- TTFF: test #{metrics['ttff']}\n"
        f"- Budget used: {metrics['tests_run']}\n"
        f"- Modules affected by diff: {modules_affected}\n"
        f"- Selected-tests module distribution: {distribution}\n"
        f"- total_seconds: {time_check['total_seconds']}\n"
        f"- sla_seconds: {time_check['sla_seconds']}\n"
        f"- avg_seconds_per_test: {time_check['avg_seconds_per_test']}\n"
    )


def run_conversation(
    toolkit: PipelineToolkit,
    llm_config: dict,
    max_critic_rounds: int = 1,
    max_ambiguity_rounds: int = 2,
    sla_seconds: int = 1200,
    confidence_threshold: float = 0.15,
) -> Dict:
    """
    Returns {"transcript": [...], "rounds_used": int}. "rounds_used" counts
    only Critic-triggered re-runs (Phase 3), for backward-compatible API
    shape with the dashboard; ambiguity/SLA/confidence round counts are
    folded into the transcript itself, not surfaced as separate top-level
    fields, to keep agents/agentic_orchestrator.py's report shape stable.

    Raises on any AutoGen-level failure - caller (agentic_orchestrator.py)
    is responsible for catching and falling back to the deterministic
    pipeline.
    """
    supervisor, executor = build_supervisor(llm_config, toolkit)
    critic = build_critic(llm_config)
    risk_reviewer = build_risk_reviewer(llm_config)
    sla_reviewer = build_sla_reviewer(llm_config)
    confidence_reviewer = build_confidence_reviewer(llm_config)
    transcript: List[Dict] = []

    def _reviewer_proxy(name: str = "Reviewer") -> autogen.UserProxyAgent:
        return autogen.UserProxyAgent(
            name=name, human_input_mode="NEVER",
            max_consecutive_auto_reply=0, code_execution_config=False,
        )

    # ── Phase 0: classify + module-ambiguity check ──────────────────────
    phase0 = executor.initiate_chat(
        supervisor,
        message=(
            "Call analyze_diff(), then call get_module_ambiguity_report(). "
            "After get_module_ambiguity_report() returns, send a SUMMARY: "
            "message describing what you found. Do not call score_risk yet."
        ),
        max_turns=6,
    )
    transcript.extend(_normalize_messages(phase0.chat_history, default_sender="Supervisor"))
    last_phase = phase0

    ambiguity_rounds_used = 0
    ambiguity = toolkit.get_module_ambiguity_report()
    while ambiguity["ambiguous_files"]:
        review_chat = _reviewer_proxy().initiate_chat(
            risk_reviewer, message=_build_ambiguity_prompt(ambiguity["ambiguous_files"]), max_turns=1,
        )
        transcript.extend(_normalize_messages(review_chat.chat_history, default_sender="RiskReviewer"))
        reply = review_chat.chat_history[-1].get("content", "") if review_chat.chat_history else ""

        if "REQUEST:" not in reply.upper() or ambiguity_rounds_used >= max_ambiguity_rounds:
            break

        ambiguity_rounds_used += 1
        followup = (
            f"The RiskReviewer challenged your module classification:\n{reply}\n\n"
            "Apply this feedback: call reexamine_module_match ONCE with the file and "
            "module named above, keep=false. Do not call get_module_ambiguity_report "
            "again - the correction is applied immediately by reexamine_module_match. "
            "Immediately after reexamine_module_match returns, send a new SUMMARY: "
            "message and stop - do not call score_risk yet."
        )
        prior_len = len(last_phase.chat_history)
        phase0_n = executor.initiate_chat(
            supervisor, message=followup, max_turns=4, clear_history=False,
        )
        transcript.extend(_normalize_messages(phase0_n.chat_history[prior_len:], default_sender="Supervisor"))
        last_phase = phase0_n

        ambiguity = toolkit.get_module_ambiguity_report()

    # ── Phase 1: score + select + report (existing tool order, unchanged) ──
    prior_len = len(last_phase.chat_history)
    phase1 = executor.initiate_chat(
        supervisor,
        message=(
            "Continue the pipeline on the current (possibly corrected) module "
            "classification: call score_risk(), then select_tests(), then "
            "get_report_metrics(). Then send a SUMMARY: message and stop."
        ),
        max_turns=8,
        clear_history=False,
    )
    transcript.extend(_normalize_messages(phase1.chat_history[prior_len:], default_sender="Supervisor"))
    last_phase = phase1

    # ── Phase 2: execution-time SLA negotiation ─────────────────────────
    time_check = toolkit.get_selection_time_budget_check(sla_seconds=sla_seconds)
    if time_check["over_sla"]:
        sla_chat = _reviewer_proxy().initiate_chat(
            sla_reviewer, message=_build_sla_prompt(toolkit, time_check), max_turns=1,
        )
        transcript.extend(_normalize_messages(sla_chat.chat_history, default_sender="SLAReviewer"))
        reply = sla_chat.chat_history[-1].get("content", "") if sla_chat.chat_history else ""

        if "REQUEST:" in reply.upper():
            followup = (
                f"The SLAReviewer reviewed execution time and responded:\n{reply}\n\n"
                f"Apply this feedback: call select_tests_by_time_budget ONCE with "
                f"time_budget_seconds={sla_seconds}, then call get_report_metrics ONCE. "
                "Do not call either tool more than once - each result is already final "
                "the moment it returns. Immediately after get_report_metrics returns, "
                "send a new SUMMARY: message and stop."
            )
            prior_len = len(last_phase.chat_history)
            phase2 = executor.initiate_chat(
                supervisor, message=followup, max_turns=4, clear_history=False,
            )
            transcript.extend(_normalize_messages(phase2.chat_history[prior_len:], default_sender="Supervisor"))
            last_phase = phase2

    # ── Phase 2.5: RL policy confidence negotiation ─────────────────────
    confidence_report = toolkit.get_selection_confidence_report(low_confidence_threshold=confidence_threshold)
    if confidence_report["available"] and confidence_report["flagged"]:
        conf_chat = _reviewer_proxy().initiate_chat(
            confidence_reviewer, message=_build_confidence_prompt(confidence_report), max_turns=1,
        )
        transcript.extend(_normalize_messages(conf_chat.chat_history, default_sender="ConfidenceReviewer"))
        reply = conf_chat.chat_history[-1].get("content", "") if conf_chat.chat_history else ""

        if "REQUEST:" in reply.upper():
            followup = (
                f"The ConfidenceReviewer flagged a low-confidence pick and responded:\n{reply}\n\n"
                "Apply this feedback: call substitute_test_with_vrtq_heuristic ONCE with the "
                "exact test_id named above, then call get_report_metrics ONCE. Do not call "
                "either tool more than once - each result is already final the moment it "
                "returns. Immediately after get_report_metrics returns, send a new SUMMARY: "
                "message and stop."
            )
            prior_len = len(last_phase.chat_history)
            phase2_5 = executor.initiate_chat(
                supervisor, message=followup, max_turns=4, clear_history=False,
            )
            transcript.extend(_normalize_messages(phase2_5.chat_history[prior_len:], default_sender="Supervisor"))
            last_phase = phase2_5

    # ── Phase 3: final outcome-quality review (existing Critic, SLA-aware) ──
    rounds_used = 0
    while toolkit.report is not None:
        critic_chat = _reviewer_proxy().initiate_chat(
            critic, message=_build_review_prompt(toolkit, sla_seconds), max_turns=1,
        )
        transcript.extend(_normalize_messages(critic_chat.chat_history, default_sender="Critic"))
        critic_reply = critic_chat.chat_history[-1].get("content", "") if critic_chat.chat_history else ""

        if "REQUEST:" not in critic_reply.upper() or rounds_used >= max_critic_rounds:
            break

        rounds_used += 1
        followup = (
            f"The Critic reviewed your last result and responded:\n{critic_reply}\n\n"
            "Apply this feedback: call select_tests ONCE with the requested adjustment, "
            "then call get_report_metrics ONCE. Do not call either tool more than once. "
            "Immediately after get_report_metrics returns, send a new SUMMARY: message "
            "and stop."
        )
        prior_len = len(last_phase.chat_history)
        phase3 = executor.initiate_chat(
            supervisor, message=followup, max_turns=5, clear_history=False,
        )
        transcript.extend(_normalize_messages(phase3.chat_history[prior_len:], default_sender="Supervisor"))
        last_phase = phase3

    return {"transcript": transcript, "rounds_used": rounds_used}
