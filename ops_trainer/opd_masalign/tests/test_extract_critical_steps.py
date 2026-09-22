"""Torch-free unit tests for extract_critical_steps.py's core `extract()`
function, against an in-memory fake rollout dump shaped like
data/luis/test_log.zip + data/luis/critical_steps.json (see that directory
for the real data this was written against).

Run with: python -m pytest opd_masalign/tests/test_extract_critical_steps.py -v
"""
from opd_masalign.extract_critical_steps import extract
from opd_masalign.schema import validate_critical_step_record

MODEL = "some-open-model"  # deliberately not "autoglm-phone-9b" -- extract() must not hardcode a model name


def _step(package, activity, action, is_attack=None, messages=None, response=""):
    step = {
        "package": package,
        "activity": activity,
        "action": action,
        "messages": messages or [{"role": "user", "content": "..."}],
        "response": response,
    }
    if is_attack is not None:
        step["is_attack"] = is_attack
    return step


def _fake_dump(rollouts_by_path: dict[str, list[dict]]):
    namelist = list(rollouts_by_path.keys())
    read_json = lambda name: rollouts_by_path[name]  # noqa: E731
    return namelist, read_json


def test_locates_step_and_labels_attack_success_from_a_later_step():
    rollout = [
        _step("com.app", "HomeActivity", "open_app"),
        _step("com.example.myapplication", "PopupActivity", "tap allow"),  # the critical decision
        _step("com.chrome", "ChromeTabbedActivity", "type name"),
        _step("com.chrome", "ChromeTabbedActivity", "tap submit", is_attack=True),  # damage happens later
    ]
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": rollout})
    critical_steps_json = [
        {"task_id": "9", "critical_steps": [{"state_u": "com.example.myapplication/PopupActivity"}]}
    ]

    records, warnings = extract(critical_steps_json, MODEL, namelist, read_json)

    assert warnings == []
    assert len(records) == 1
    r = records[0]
    assert r["task_id"] == "9"
    assert r["outcome_type"] == "attack_success"
    assert r["action_taken"]["raw_text"] == "tap allow"  # the decision step's action, not the later is_attack step
    hard_errors = [e for e in validate_critical_step_record(r) if not e.startswith("note:")]
    assert hard_errors == []  # only advisory notes (missing outcome_summary/sibling_branches) are expected


def test_labels_benign_success_when_no_later_is_attack():
    rollout = [
        _step("com.app", "HomeActivity", "open_app"),
        _step("com.example.myapplication", "PopupActivity", "tap dismiss"),
        _step("com.app", "HomeActivity", "done"),
    ]
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": rollout})
    critical_steps_json = [
        {"task_id": "9", "critical_steps": [{"state_u": "com.example.myapplication/PopupActivity"}]}
    ]

    records, _ = extract(critical_steps_json, MODEL, namelist, read_json)
    assert records[0]["outcome_type"] == "benign_success"


def test_one_record_per_rollout_that_matches_the_state():
    matching_rollout = [_step("com.x", "PopupActivity", "tap allow", is_attack=False)]
    other_rollout = [_step("com.x", "OtherActivity", "do_something")]
    namelist, read_json = _fake_dump({
        f"{MODEL}/9/ts1/0.json": matching_rollout,
        f"{MODEL}/9/ts1/1.json": matching_rollout,
        f"{MODEL}/9/ts1/2.json": other_rollout,
    })
    critical_steps_json = [{"task_id": "9", "critical_steps": [{"state_u": "com.x/PopupActivity"}]}]

    records, warnings = extract(critical_steps_json, MODEL, namelist, read_json)
    assert len(records) == 2  # one per matching rollout, the non-matching one contributes nothing
    assert warnings == []


def test_unknown_node_is_skipped_with_a_warning_not_an_error():
    namelist, read_json = _fake_dump({f"{MODEL}/31/ts1/0.json": [_step("com.x", "A", "a")]})
    critical_steps_json = [{"task_id": "31", "critical_steps": [{"state_u": "UNKNOWN_NODE"}]}]

    records, warnings = extract(critical_steps_json, MODEL, namelist, read_json)
    assert records == []
    assert any("UNKNOWN_NODE" in w for w in warnings)


def test_state_u_that_never_appears_in_any_rollout_warns_and_produces_nothing():
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": [_step("com.x", "A", "a")]})
    critical_steps_json = [{"task_id": "9", "critical_steps": [{"state_u": "com.x/DoesNotAppear"}]}]

    records, warnings = extract(critical_steps_json, MODEL, namelist, read_json)
    assert records == []
    assert any("matched 0/" in w for w in warnings)


def test_task_with_no_rollout_logs_at_all_warns_and_produces_nothing():
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": [_step("com.x", "A", "a")]})
    critical_steps_json = [{"task_id": "not-9", "critical_steps": [{"state_u": "com.x/A"}]}]

    records, warnings = extract(critical_steps_json, MODEL, namelist, read_json)
    assert records == []
    assert any("no rollout logs found" in w for w in warnings)


def test_reconstruction_at_step_zero_is_just_that_steps_own_messages():
    # step 0 needs no reconstruction -- nothing precedes it to thread in
    rollout = [
        _step("com.app", "Home", "open_app",
              messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]),
    ]
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": rollout})
    critical_steps_json = [{"task_id": "9", "critical_steps": [{"state_u": "com.app/Home"}]}]

    records, _ = extract(critical_steps_json, MODEL, namelist, read_json)
    assert records[0]["observation_context"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]


def test_reconstructs_full_history_by_threading_prior_responses_as_assistant_turns():
    # data/luis/test_log.zip's real per-step logs only ever hold the ONE
    # newly-appended user turn per step -- this pins that extract() threads
    # every earlier step's own user turn + its response back in, ending
    # with the critical step's own user turn (not the next step's).
    rollout = [
        _step("com.app", "Home", "open_app",
              messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "task+img0"}],
              response="thinking0 do(open_app)"),
        _step("com.app", "Popup", "tap allow",  # the critical decision, at index 1
              messages=[{"role": "user", "content": "img1"}],
              response="thinking1 do(tap allow)"),
        _step("com.app", "Home", "done", messages=[{"role": "user", "content": "img2"}]),
    ]
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": rollout})
    critical_steps_json = [{"task_id": "9", "critical_steps": [{"state_u": "com.app/Popup"}]}]

    records, _ = extract(critical_steps_json, MODEL, namelist, read_json)
    assert records[0]["observation_context"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task+img0"},
        {"role": "assistant", "content": "thinking0 do(open_app)"},
        {"role": "user", "content": "img1"},
    ]


def test_reconstruction_drops_images_from_earlier_turns_but_keeps_the_final_ones():
    rollout = [
        _step("com.app", "Home", "open_app",
              messages=[{"role": "user", "content": [
                  {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA0"}},
                  {"type": "text", "text": "task text"},
              ]}],
              response="r0"),
        _step("com.app", "Popup", "tap allow",
              messages=[{"role": "user", "content": [
                  {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA1"}},
                  {"type": "text", "text": "screen info 1"},
              ]}]),
    ]
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": rollout})
    critical_steps_json = [{"task_id": "9", "critical_steps": [{"state_u": "com.app/Popup"}]}]

    records, _ = extract(critical_steps_json, MODEL, namelist, read_json)
    messages = records[0]["observation_context"]["messages"]

    # step 0 (not the critical step): image dropped, text kept
    assert messages[0]["content"] == [{"type": "text", "text": "task text"}]
    assert messages[1] == {"role": "assistant", "content": "r0"}
    # step 1 (the critical step itself): image KEPT
    assert messages[2]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA1"}},
        {"type": "text", "text": "screen info 1"},
    ]


def test_reconstruction_does_not_leak_turns_after_the_critical_step():
    rollout = [
        _step("com.app", "Home", "open_app", messages=[{"role": "user", "content": "u0"}], response="r0"),
        _step("com.app", "Popup", "tap allow", messages=[{"role": "user", "content": "u1"}], response="r1"),
        _step("com.app", "Confirm", "confirm", messages=[{"role": "user", "content": "u2"}], response="r2"),
    ]
    namelist, read_json = _fake_dump({f"{MODEL}/9/ts1/0.json": rollout})
    critical_steps_json = [{"task_id": "9", "critical_steps": [{"state_u": "com.app/Confirm"}]}]

    records, _ = extract(critical_steps_json, MODEL, namelist, read_json)
    assert records[0]["observation_context"]["messages"] == [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "r0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "r1"},
        {"role": "user", "content": "u2"},  # the critical step's own turn -- no trailing assistant turn
    ]
