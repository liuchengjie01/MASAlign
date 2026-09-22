"""Data schema for MASAlign / OPD.

CriticalStepRecord: the training-time hand-off format, matching
docs/DATA_FORMAT.md. This is what opd_trainer.py consumes directly -- there
is no upstream tree/graph/risk pipeline in this package; critical steps are
located and labeled upstream of this schema (e.g. against a collaborator's
raw rollout logs) and handed off already in this format.

Plain dataclasses/dicts (JSON-serializable) rather than pydantic models.
"""
from __future__ import annotations

TERMINAL_TYPES = ("benign_success", "attack_success", "benign_fail", "truncated")

# The only 4 fields a collaborator must provide -- see docs/DATA_FORMAT.md §1.2.
# Everything else (outcome_summary, sibling_branches, ...) is optional: the
# trainer falls back to sane defaults when a record omits them.
# old_action_logprob is NOT in this list and is not read from records at all
# -- opd_trainer.py self-computes it from the model being trained (see
# OPDTrainer.precompute_old_log_probs), since IS-clip only means anything
# when "old" refers to the exact checkpoint/tokenizer training starts from.
REQUIRED_CRITICAL_STEP_FIELDS = [
    "task_id",
    "observation_context",
    "action_taken",
    "outcome_type",
]


def validate_critical_step_record(d: dict) -> list[str]:
    """Return a list of human-readable errors; empty list == valid.

    Only checks the 4 required fields as hard errors. Missing optional
    fields (outcome_summary, sibling_branches) produce advisory notes, not
    errors, since the trainer has a defined fallback for each -- see
    docs/DATA_FORMAT.md §1.2.
    """
    errors = []
    for key in REQUIRED_CRITICAL_STEP_FIELDS:
        if key not in d or d[key] is None:
            errors.append(f"missing required field: {key}")

    ctx = d.get("observation_context")
    if isinstance(ctx, dict):
        fmt = ctx.get("format")
        if fmt == "chat_messages" and not ctx.get("messages"):
            errors.append("observation_context.format == chat_messages but 'messages' is empty")
        elif fmt == "raw_text" and not ctx.get("raw_text"):
            errors.append("observation_context.format == raw_text but 'raw_text' is empty")
        elif fmt not in ("chat_messages", "raw_text"):
            errors.append(f"observation_context.format must be chat_messages|raw_text, got {fmt!r}")

    action = d.get("action_taken")
    if isinstance(action, dict) and not action.get("raw_text"):
        errors.append("action_taken.raw_text is required (needed to teacher-force recompute log-probs)")

    outcome = d.get("outcome_type")
    if outcome is not None and outcome not in TERMINAL_TYPES:
        errors.append(f"outcome_type must be one of {TERMINAL_TYPES}, got {outcome!r}")

    if not d.get("outcome_summary"):
        errors.append(
            "note: outcome_summary is missing -- the self-teacher reprompt will fall back to a "
            "generic sentence derived from outcome_type (optional field, not an error)"
        )
    if not d.get("sibling_branches"):
        errors.append(
            "note: sibling_branches is missing -- the reprompt will omit the 'safer alternative' "
            "hint (optional field, not an error)"
        )

    return errors
