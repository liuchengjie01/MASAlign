"""Build a CriticalStepRecord JSONL (docs/DATA_FORMAT.md) directly from a
collaborator's raw per-step rollout logs + their critical-step location file,
with NO tree/graph/risk pipeline involved -- this is the ETL step discussed
for real data/luis-shaped dumps, not a smoke-test generator.

Input shapes this expects (see data/luis/ for the real example this was
written against):
  - `--critical_steps_json`: a JSON list of
        {"task_id": "...", "critical_steps": [{"state_u": "<package>/<activity>"}, ...]}
    `state_u` is an abstract locator, not a training example -- it only says
    WHICH on-device state was flagged as critical.
  - `--zip` (or `--dir`): raw rollout logs at
        <model>/<task_id>/<timestamp>/<rollout_idx>.json
    each a list of per-step dicts with (at least) `package`, `activity`,
    `action`, `messages`, `response`, and optionally a per-step boolean
    `is_attack`.

What this script does, per (task_id, state_u, rollout):
  1. LOCATE: scan the rollout for the first step whose "<package>/<activity>"
     matches state_u -- that step is the actual training example. A state_u
     that can't be matched (e.g. "UNKNOWN_NODE") is skipped with a warning,
     not silently dropped as an error.
  2. RECONSTRUCT the full conversation (see _reconstruct_full_messages):
     each step's own logged `messages` field in data/luis/test_log.zip
     turns out to hold only THAT step's newly-appended user turn, never the
     growing prior history -- confirmed against the raw logs' own
     `prompt_tokens` field, which climbs steadily step over step (e.g. one
     real rollout: 4058 -> 5066 across steps 1-15) even though each
     individual step's own logged content (one screenshot + one short text
     line) stays essentially constant in size. The missing turns are
     recovered by threading together every earlier step's own user turn
     (text only -- confirmed with whoever collected data/luis/test_log.zip
     that each real API call only ever carried the CURRENT screenshot, no
     prior ones) plus its `response` (as that step's assistant turn),
     ending with the located step's own user turn (image included). See
     that function's docstring for the one remaining unconfirmed
     assumption (whether older turns' TEXT is windowed/summarized rather
     than kept verbatim) before trusting this for a real (non-smoke)
     training run.
  3. LABEL: scan the SAME rollout forward from that step for any later step
     with `is_attack: true` -- if found, outcome_type="attack_success",
     otherwise "benign_success". This is a deliberate binary simplification
     (no benign_fail/truncated distinction, no outcome_summary) -- see the
     project discussion this was written from: we only need to know whether
     the decision at that state eventually led to an attack succeeding.

--model is a required argument, not a hardcoded string: this script works
against whichever collaborator model's logs you point it at (only models
whose logs actually carry `is_attack`, e.g. autoglm-phone-9b in data/luis/
today, can currently be labeled by step 2 above -- others will emit a
warning and be skipped).

Usage:
    python -m opd_masalign.extract_critical_steps \
        --zip data/luis/test_log.zip data/autoglm-phone-9b.zip data/autoglm-phone-9b-part10.zip \
        --critical_steps_json data/luis/critical_steps.json \
        --model autoglm-phone-9b \
        --out data/critical_steps_autoglm.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from typing import Optional

from .action_parse import parse_response, parse_response_markdown_headers


def _state_key(step: dict) -> str:
    return f"{step.get('package', '')}/{step.get('activity', '')}"


def _locate_critical_step(rollout: list[dict], state_u: str) -> Optional[int]:
    """First step index whose package/activity matches state_u, or None."""
    for i, step in enumerate(rollout):
        if _state_key(step) == state_u:
            return i
    return None


def _drop_images(content):
    """Strip image_url blocks from a message's content, keeping text blocks
    (or the content as-is if it isn't a content-block list). Used on every
    turn EXCEPT the final one when reconstructing history -- see
    _reconstruct_full_messages's docstring for why older turns' images are
    dropped rather than re-included.
    """
    if not isinstance(content, list):
        return content
    return [b for b in content if b.get("type") != "image_url"]


def _reconstruct_full_messages(rollout: list[dict], step_idx: int) -> list[dict]:
    """Rebuild the multi-turn conversation actually sent to the LLM at
    step_idx, by threading together every earlier step's own logged user
    turn (text only) plus its `response` (recast as that step's assistant
    turn), ending with step_idx's own user turn (kept as-is, image
    included).

    WHY THIS IS NEEDED (not a nice-to-have, see this module's docstring for
    the evidence): each step's `messages` field only ever holds ONE newly-
    appended user turn (two for step 0, which also carries the system
    turn) -- never the accumulated history -- even though the real API
    call's own reported `prompt_tokens` grows steadily step over step.
    Passing step_idx's bare single-turn `messages` on its own as
    observation_context would train against a context the model never
    actually saw.

    WHY OLDER TURNS' IMAGES ARE DROPPED, NOT JUST THEIR MISSING-FROM-THE-LOG
    TEXT: re-including every prior turn's own screenshot at full resolution
    made single records balloon past 5MB and didn't fit the evidence either
    -- the real `prompt_tokens` growth is only ~70-200/step (matches
    ~1 short response + 1 short "Screen Info" line accumulating in text),
    nowhere near what an extra full-resolution image every turn would cost
    (typically hundreds to low thousands of vision tokens each). Keeping
    only the LATEST turn's image and threading in only TEXT for everything
    earlier is the reconstruction that actually matches the token-count
    evidence.

    ASSUMPTION (2) CONFIRMED with whoever collected data/luis/test_log.zip:
    each step's real API call carries only the CURRENT screenshot, no prior
    ones -- so dropping older turns' images here is correct, not a guess.

    ASSUMPTION (1) STILL UNVERIFIED: that the missing turns are exactly this
    rollout's own prior steps, threaded as plain alternating
    user/assistant turns with no summarization/compression/sliding window
    on the TEXT side (confirmed images are dropped; whether older turns'
    TEXT is also ever summarized/windowed rather than kept verbatim is
    still open). If the real framework windows or summarizes prior text
    turns instead of keeping every one verbatim, this reconstruction will
    not match the true observation_context, and docs/DATA_FORMAT.md's own
    prerequisite (byte-identical to what was actually sent) would be
    violated. Get this confirmed before trusting a real (non-smoke)
    training run built on it.
    """
    messages: list[dict] = []
    for i in range(step_idx + 1):
        step = rollout[i]
        is_last = i == step_idx
        for m in step["messages"]:
            content = m["content"] if is_last else _drop_images(m["content"])
            messages.append({**m, "content": content})
        if not is_last:
            messages.append({"role": "assistant", "content": step.get("response", "")})
    return messages


def _label_outcome(rollout: list[dict], from_idx: int) -> str:
    """attack_success if any step from_idx..end has is_attack==True, else benign_success."""
    for step in rollout[from_idx:]:
        if step.get("is_attack"):
            return "attack_success"
    return "benign_success"


def _describe_downstream_guiowl(rollout: list[dict], from_idx: int) -> str:
    """Plain factual summary of what happened from step from_idx on, for the general hindsight
    reprompt's $downstream_event (reprompt_variants.py). Built only from the log itself: the first
    is_attack step's app + the executor's own one-line `### Description ###` of that action, and the
    text typed along the way. Says nothing about *which* step was the mistake -- that's the point.
    """
    attack_idx = next((i for i in range(from_idx, len(rollout)) if rollout[i].get("is_attack")), None)
    if attack_idx is None:
        return "The rest of this episode finished without the attack succeeding."
    step = rollout[attack_idx]
    desc = step.get("action_response", "").split("### Description ###", 1)[-1].strip() or step.get("action", "")
    where = "This very action" if attack_idx == from_idx else (
        f"{attack_idx - from_idx} step(s) after this one, in {step.get('package', 'another app')}, the action"
    )
    text = (f"{where} \"{desc}\" is where the attack succeeded: information from the user's request was "
            "submitted to a page controlled by an attacker.")
    typed = []
    for st in rollout[from_idx:attack_idx + 1]:
        try:
            a = json.loads(st.get("action", ""))
        except ValueError:
            continue
        if isinstance(a, dict) and a.get("action") == "type" and a.get("text"):
            typed.append(a["text"])
    if typed:
        text += " Along the way you typed: " + ", ".join(f'"{t}"' for t in typed) + "."
    return text


def _describe_downstream_chain_guiowl(rollout: list[dict], from_idx: int) -> str:
    """Like _describe_downstream_guiowl, but lists EVERY action from from_idx up to the first is_attack
    step (each step's own `### Description ###`, the typed text for `type` actions, and an app-switch
    marker whenever the package changes), for the g2 reprompt's $downstream_chain. Still purely factual
    and position-agnostic: it never says which step was the mistake, but lets the model see e.g. that
    the episode left the task's app right after its own action.
    """
    attack_idx = next((i for i in range(from_idx, len(rollout)) if rollout[i].get("is_attack")), None)
    if attack_idx is None:
        return "The rest of this episode finished without the attack succeeding."
    parts = []
    package = rollout[from_idx].get("package")
    for i in range(from_idx, attack_idx + 1):
        st = rollout[i]
        try:
            a = json.loads(st.get("action", ""))
        except ValueError:
            a = None
        if isinstance(a, dict) and a.get("action") == "type" and a.get("text"):
            desc = f'Type "{a["text"]}".'
        else:
            desc = st.get("action_response", "").split("### Description ###", 1)[-1].strip() or st.get("action", "")
        if st.get("package") != package:
            package = st.get("package")
            desc = f"[now in {package}] {desc}"
        parts.append(f"(this step) {desc}" if i == from_idx else desc)
    return (" -> ".join(parts) + " <- here the attack succeeded: information from the user's request was "
            "submitted to a page controlled by an attacker.")


def _popup_offset(rollout: list[dict], step_idx: int) -> Optional[int]:
    """step_idx minus the first PopupActivity step (negative = before the popup); None if the rollout
    never shows one. Analysis-only metadata for grouping results -- no prompt reads it."""
    first = next((i for i, st in enumerate(rollout) if "PopupActivity" in st.get("activity", "")), None)
    return None if first is None else step_idx - first


def popup_position(popup_offset: Optional[int]) -> str:
    """'before' / 'at' / 'after' the popup step, or 'no_popup' -- for grouping results only."""
    if popup_offset is None:
        return "no_popup"
    return "at" if popup_offset == 0 else ("before" if popup_offset < 0 else "after")


def _iter_rollouts(namelist: list[str], read_json, model: str, task_id: str):
    """Yield (source_path, rollout) for every <model>/<task_id>/*/<rollout_idx>.json entry.
    Rollout files are named by a bare integer index (0.json, 1.json, ...); anything else
    (e.g. GUI-Owl/Mobile-Agent-V3 logs' per-task-timestamp `task_result.json` summary, which
    isn't a list of per-step dicts and would blow up downstream) is skipped on purpose, not
    just by accident of extension -- confirmed against data/luis/gui-owl-7b.zip's layout.
    """
    prefix = f"{model}/{task_id}/"
    for name in namelist:
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        if not name.rsplit("/", 1)[-1][: -len(".json")].isdigit():
            continue
        yield name, read_json(name)


def extract(
    critical_steps_json: list[dict],
    model: str,
    namelist: list[str],
    read_json,
    fmt: str = "autoglm",
) -> tuple[list[dict], list[str]]:
    """Returns (records, warnings). Kept side-effect-free (no file I/O) so
    this is unit-testable against an in-memory fake `namelist`/`read_json`.

    `fmt` picks the per-step field names and response grammar:
      - "autoglm": `messages` (only the newly-appended turn -- threaded into
        full history by _reconstruct_full_messages, see its docstring) +
        `response` (`<think>...</think><answer>...</answer>`, see action_parse.parse_response).
      - "guiowl": Mobile-Agent-V3/GUI-Owl executor logs. `action_messages` is
        already the COMPLETE single-turn prompt actually sent (the planner's
        plan/subgoal text + current screenshot are baked into that one turn
        by the collection harness itself, confirmed by inspecting
        data/luis/gui-owl-7b.zip: every step carries exactly one message, not
        a growing history) -- used as-is, no reconstruction. `action_response`
        has no special tokens, just `### Thought ### / ### Action ### /
        ### Description ###` headers (action_parse.parse_response_markdown_headers).
        Also adds `downstream_event` / `downstream_chain` (see _describe_downstream_guiowl /
        _describe_downstream_chain_guiowl), and `step_idx` /
        `popup_offset` for locating the step and grouping results, and `planner_context` (the
        step's own `planning_messages`) for opd_trainer.py --teacher_replan.
    """
    if fmt not in ("autoglm", "guiowl"):
        raise ValueError(f"fmt must be 'autoglm' or 'guiowl', got {fmt!r}")
    messages_field = "messages" if fmt == "autoglm" else "action_messages"
    response_field = "response" if fmt == "autoglm" else "action_response"
    parser = parse_response if fmt == "autoglm" else parse_response_markdown_headers

    records: list[dict] = []
    warnings: list[str] = []

    for task_entry in critical_steps_json:
        task_id = task_entry["task_id"]
        state_us = [cs["state_u"] for cs in task_entry.get("critical_steps", [])]
        rollouts = list(_iter_rollouts(namelist, read_json, model, task_id))
        if not rollouts:
            warnings.append(f"task_id={task_id!r}: no rollout logs found under {model}/{task_id}/*/*.json")
            continue

        for state_u in state_us:
            if state_u == "UNKNOWN_NODE":
                warnings.append(f"task_id={task_id!r}: state_u is UNKNOWN_NODE, cannot locate -- skipped")
                continue

            n_matched = 0
            for source_path, rollout in rollouts:
                step_idx = _locate_critical_step(rollout, state_u)
                if step_idx is None:
                    continue
                n_matched += 1
                step = rollout[step_idx]
                outcome_type = _label_outcome(rollout, step_idx)
                full_messages = (
                    _reconstruct_full_messages(rollout, step_idx) if fmt == "autoglm"
                    else step[messages_field]
                )
                action_taken = {"raw_text": step["action"]}
                # The step's own reasoning, kept so opd_trainer.py --think_prefill can score the action
                # at the position the model really produces it. Only trusted when the parsed final
                # action equals the logged `action` (cross-check against the grammar each `fmt` uses).
                thinking, parsed_action, well_formed = parser(step.get(response_field, ""))
                if well_formed and parsed_action.strip() == step["action"].strip():
                    action_taken["thinking"] = thinking
                record = {
                    "schema_version": "1.1",
                    "record_type": "critical_step",
                    "task_id": task_id,
                    "observation_context": {"format": "chat_messages", "messages": full_messages},
                    "action_taken": action_taken,
                    "outcome_type": outcome_type,
                    "state_u": state_u,
                    "source": source_path,
                }
                if fmt == "guiowl":
                    record["downstream_event"] = _describe_downstream_guiowl(rollout, step_idx)
                    record["downstream_chain"] = _describe_downstream_chain_guiowl(rollout, step_idx)
                    record["step_idx"] = step_idx
                    record["popup_offset"] = _popup_offset(rollout, step_idx)
                    if step.get("planning_messages"):
                        # the planner sub-call on the same screenshot, for opd_trainer.py --teacher_replan
                        record["planner_context"] = {"format": "chat_messages", "messages": step["planning_messages"]}
                records.append(record)

            if n_matched == 0:
                warnings.append(
                    f"task_id={task_id!r}, state_u={state_u!r}: matched 0/{len(rollouts)} rollouts "
                    f"(activity string never appears in any {model} rollout for this task)"
                )

    return records, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip", nargs="+",
        help="one or more raw rollout dump zips (e.g. data/luis/test_log.zip). A zip whose entries are "
        "<model>/<task_id>/... is used as-is; one scoped to a single model (<task_id>/..., e.g. "
        "autoglm-phone-9b.zip) is mounted under --model. When several zips contain the same path the "
        "first one wins (identical re-exports of the same rollout are de-duplicated this way).",
    )
    parser.add_argument("--dir", help="alternative to --zip: an already-extracted directory with the same layout")
    parser.add_argument("--critical_steps_json", required=True, help="e.g. data/luis/critical_steps.json")
    parser.add_argument("--model", required=True, help="model directory name under the zip/dir, e.g. autoglm-phone-9b -- NOT hardcoded, pass whichever collaborator model's logs you're extracting")
    parser.add_argument("--format", default="autoglm", choices=["autoglm", "guiowl"], help="per-step log field names/response grammar, see extract()'s docstring")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if bool(args.zip) == bool(args.dir):
        raise SystemExit("pass exactly one of --zip or --dir")

    with open(args.critical_steps_json) as f:
        critical_steps_json = json.load(f)

    if args.zip:
        sources: dict[str, tuple[zipfile.ZipFile, str]] = {}
        for path in args.zip:
            z = zipfile.ZipFile(path)
            names = [n for n in z.namelist() if n.endswith(".json")]
            scoped = not any(n.startswith(f"{args.model}/") for n in names)
            for n in names:
                sources.setdefault(f"{args.model}/{n}" if scoped else n, (z, n))
        namelist = list(sources)

        def read_json(name):
            z, real = sources[name]
            return json.load(z.open(real))
    else:
        namelist = [
            os.path.relpath(os.path.join(root, fn), args.dir)
            for root, _, files in os.walk(args.dir)
            for fn in files
        ]
        read_json = lambda name: json.load(open(os.path.join(args.dir, name)))  # noqa: E731

    records, warnings = extract(critical_steps_json, args.model, namelist, read_json, fmt=args.format)

    for w in warnings:
        print(f"WARNING: {w}")

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} critical-step records ({len(warnings)} warnings) to {args.out}")


if __name__ == "__main__":
    main()
