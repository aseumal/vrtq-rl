"""
agents/agentic/conversation.py
---------------------------------
Runs the two-phase agentic conversation:
  1. Supervisor <-> ToolExecutor: drives the four pipeline-stage tool calls.
  2. Supervisor's results <-> Critic: a neutral review exchange. The Critic
     has no tool access (it only ever talks to a plain, executor-less proxy)
     so it can never attempt a tool call with nothing able to run it. If the
     Critic requests a change, control goes back to phase 1 with a followup
     instruction, up to max_critic_rounds times (hard Python counter, not
     just a prompt instruction).

Author: Anthony Seumal
Project: VRTQ-RL
"""

import json
from collections import Counter
from typing import Dict, List

import autogen

from agents.agentic.supervisor import build_supervisor
from agents.agentic.critic import build_critic
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


def _build_review_prompt(toolkit: PipelineToolkit) -> str:
    metrics = toolkit.report["metrics"]
    distribution = dict(Counter(toolkit.selector_output["selected_modules"]))
    modules_affected = toolkit.diff_features.get("modules_affected", [])
    return (
        "Review this test-selection result:\n"
        f"- FDR@25%: {metrics['fdr_25']:.1%}\n"
        f"- FDR@50%: {metrics['fdr_50']:.1%}\n"
        f"- FDR@100%: {metrics['fdr_100']:.1%}\n"
        f"- TTFF: test #{metrics['ttff']}\n"
        f"- Budget used: {metrics['tests_run']}\n"
        f"- Modules affected by diff: {modules_affected}\n"
        f"- Selected-tests module distribution: {distribution}\n"
    )


def run_conversation(
    toolkit: PipelineToolkit,
    llm_config: dict,
    max_critic_rounds: int = 1,
) -> Dict:
    """
    Returns {"transcript": [...], "rounds_used": int}. Raises on any
    AutoGen-level failure — caller (agents/agentic_orchestrator.py) is
    responsible for catching and falling back to the deterministic pipeline.
    """
    supervisor, executor = build_supervisor(llm_config, toolkit)
    critic = build_critic(llm_config)
    transcript: List[Dict] = []

    # The diff text itself is never put in the chat message — it's already
    # loaded into the toolkit, and analyze_diff() takes no arguments, so the
    # LLM never has to see/repeat the raw diff (saves tokens, keeps the
    # trace readable).
    phase1 = executor.initiate_chat(
        supervisor,
        message="Run the full pipeline on the currently loaded diff.",
        max_turns=12,
    )
    transcript.extend(_normalize_messages(phase1.chat_history, default_sender="Supervisor"))

    rounds_used = 0
    while toolkit.report is not None:
        reviewer_proxy = autogen.UserProxyAgent(
            name="Reviewer", human_input_mode="NEVER",
            max_consecutive_auto_reply=0, code_execution_config=False,
        )
        critic_chat = reviewer_proxy.initiate_chat(
            critic, message=_build_review_prompt(toolkit), max_turns=1,
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
        # clear_history=False intentionally keeps the Supervisor's own memory of
        # phase 1 (so it has context for the re-run), but that also means the
        # ChatResult it returns reflects the FULL cumulative conversation, not
        # just this round's new messages — slice to only the new tail so the
        # transcript doesn't re-include phase 1's tool calls a second time.
        prior_len = len(phase1.chat_history) if rounds_used == 1 else len(phase_n.chat_history)
        phase_n = executor.initiate_chat(
            supervisor, message=followup, max_turns=5, clear_history=False,
        )
        transcript.extend(_normalize_messages(phase_n.chat_history[prior_len:], default_sender="Supervisor"))

    return {"transcript": transcript, "rounds_used": rounds_used}
