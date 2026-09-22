"""Build the self-teacher reprompt context f_u (Section 5) from a
CriticalStepRecord (docs/DATA_FORMAT.md schema), and turn it into chat
messages appended after observation_context for the EMA-teacher forward pass.
"""
from __future__ import annotations

from string import Template
from typing import Any

_REPROMPT_TEMPLATE = Template(
    "[Hindsight review] You previously took the action below at this exact "
    "point in the task. Before answering again, reconsider it:\n\n"
    "Action you took: $action_taken\n"
    "What happened afterwards: $outcome_summary (outcome type: $outcome_type)\n"
    "$alternatives_block"
    "\nFirst, briefly reconsider why this outcome occurred given the action "
    "you took. Then reason about what the safer alternative would have been. "
    "Finally, answer again as you would at this step -- your answer should "
    "reflect this reconsideration."
)

# Used when a record has no outcome_summary (optional field, see
# docs/DATA_FORMAT.md §1.2) -- a generic fallback beats no narrative at all,
# though a real outcome_summary produces a noticeably better reprompt.
_GENERIC_OUTCOME_SUMMARY = {
    "benign_success": "The task was completed safely with no adverse effects.",
    "attack_success": "This action allowed a malicious/injected instruction to succeed.",
    "benign_fail": "The task was not completed, but no attack occurred.",
    "truncated": "The trajectory was cut off before reaching a clear outcome.",
}


def _format_alternatives(record: dict[str, Any]) -> str:
    """Uses structured `sibling_branches` if present; omits the block
    entirely otherwise (this repo's own CriticalStepRecord producers don't
    currently populate it -- see opd_masalign/extract_critical_steps.py).
    """
    alt_lines = []
    for s in record.get("sibling_branches") or []:
        if s.get("is_taken"):
            continue
        summary = s.get("action_summary") or "(no description provided)"
        alt_lines.append(f"  - {summary} (led to a safer outcome)" if s.get("child_lrs", 1.0) < 0 else f"  - {summary}")
    if not alt_lines:
        return ""
    return "Alternative actions available at this point:\n" + "\n".join(alt_lines) + "\n"


def build_reprompt_text(record: dict[str, Any]) -> str:
    action_taken = record.get("action_taken", {}).get("raw_text", "")
    outcome_type = record.get("outcome_type", "unknown")
    outcome_summary = record.get("outcome_summary") or _GENERIC_OUTCOME_SUMMARY.get(
        outcome_type, "(no summary provided)"
    )
    alternatives_block = _format_alternatives(record)

    return _REPROMPT_TEMPLATE.substitute(
        action_taken=action_taken,
        outcome_summary=outcome_summary,
        outcome_type=outcome_type,
        alternatives_block=alternatives_block,
    )


def build_teacher_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """observation_context.messages + one hindsight-review user turn.

    The EMA teacher is then asked to continue from here -- i.e. we
    teacher-force the ORIGINAL action_taken.raw_text again under this new
    context and read off its log-prob as q_theta_ema(a_u | u, f_u), per the
    reprompting template described in Section 5 of the proposal.
    """
    ctx = record["observation_context"]
    if ctx.get("format") != "chat_messages":
        raise ValueError(
            "build_teacher_messages currently only supports observation_context.format == 'chat_messages'; "
            "for 'raw_text' contexts, prepend build_reprompt_text(record) to the raw text directly instead."
        )
    messages = list(ctx["messages"])
    messages.append({"role": "user", "content": build_reprompt_text(record)})
    return messages
