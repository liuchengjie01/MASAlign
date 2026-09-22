"""Torch-free checks that the truly-minimal hand-off contract (only the 4
required fields from docs/DATA_FORMAT.md, no old_action_logprob/delta_u/
sibling_branches) is actually usable end-to-end by
validate_critical_step_record + self_teacher's reprompt builder, without
ever touching graph.py/risk.py/critical_steps.py.

Run with: python -m pytest opd_masalign/tests/test_minimal_contract.py -v
"""
from opd_masalign.schema import validate_critical_step_record
from opd_masalign.self_teacher import build_reprompt_text, build_teacher_messages

MINIMAL_RECORD = {
    "schema_version": "1.1",
    "record_type": "critical_step",
    "task_id": "task-min-0",
    "observation_context": {
        "format": "chat_messages",
        "messages": [{"role": "user", "content": "Chat app open, a popup covers the reply box"}],
    },
    "action_taken": {"raw_text": "tap 'Update Now' on the popup"},
    "outcome_type": "attack_success",
}


def test_minimal_record_has_no_hard_errors_only_advisory_notes():
    errors = validate_critical_step_record(MINIMAL_RECORD)
    # every remaining entry must be an advisory "note:", never a hard error
    assert all(e.startswith("note:") for e in errors), errors
    # the 4 required fields being present means no non-"note" error at all
    assert not any("missing required field" in e for e in errors)


def test_reprompt_falls_back_to_generic_outcome_summary_and_omits_alternatives():
    text = build_reprompt_text(MINIMAL_RECORD)
    assert "tap 'Update Now' on the popup" in text
    assert "malicious" in text.lower() or "injected" in text.lower()  # generic attack_success fallback
    assert "Alternative actions" not in text  # no counterfactual_note/sibling_branches given


def test_reprompt_uses_real_outcome_summary_when_present():
    record = dict(MINIMAL_RECORD)
    record["outcome_summary"] = "The fake update button installed a malicious APK."
    text = build_reprompt_text(record)
    assert "installed a malicious APK" in text


def test_reprompt_includes_alternatives_from_sibling_branches_when_present():
    record = dict(MINIMAL_RECORD)
    record["sibling_branches"] = [
        {"action_summary": "Tap the popup's X to dismiss it", "is_taken": False, "child_lrs": -3.0},
        {"action_summary": "Tap 'Update Now'", "is_taken": True, "child_lrs": 0.0},
    ]
    text = build_reprompt_text(record)
    assert "Tap the popup's X to dismiss it" in text


def test_build_teacher_messages_appends_one_hindsight_turn():
    messages = build_teacher_messages(MINIMAL_RECORD)
    assert len(messages) == len(MINIMAL_RECORD["observation_context"]["messages"]) + 1
    assert messages[-1]["role"] == "user"
    assert "[Hindsight review]" in messages[-1]["content"]
