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
        --zip data/luis/test_log.zip \
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


def _iter_rollouts(namelist: list[str], read_json, model: str, task_id: str):
    """Yield (source_path, rollout) for every <model>/<task_id>/*/*.json entry."""
    prefix = f"{model}/{task_id}/"
    for name in namelist:
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        yield name, read_json(name)


def extract(
    critical_steps_json: list[dict],
    model: str,
    namelist: list[str],
    read_json,
) -> tuple[list[dict], list[str]]:
    """Returns (records, warnings). Kept side-effect-free (no file I/O) so
    this is unit-testable against an in-memory fake `namelist`/`read_json`.
    """
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
                full_messages = _reconstruct_full_messages(rollout, step_idx)
                records.append({
                    "schema_version": "1.1",
                    "record_type": "critical_step",
                    "task_id": task_id,
                    "observation_context": {"format": "chat_messages", "messages": full_messages},
                    "action_taken": {"raw_text": step["action"]},
                    "outcome_type": outcome_type,
                })

            if n_matched == 0:
                warnings.append(
                    f"task_id={task_id!r}, state_u={state_u!r}: matched 0/{len(rollouts)} rollouts "
                    f"(activity string never appears in any {model} rollout for this task)"
                )

    return records, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", help="path to the raw rollout dump zip (e.g. data/luis/test_log.zip)")
    parser.add_argument("--dir", help="alternative to --zip: an already-extracted directory with the same layout")
    parser.add_argument("--critical_steps_json", required=True, help="e.g. data/luis/critical_steps.json")
    parser.add_argument("--model", required=True, help="model directory name under the zip/dir, e.g. autoglm-phone-9b -- NOT hardcoded, pass whichever collaborator model's logs you're extracting")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if bool(args.zip) == bool(args.dir):
        raise SystemExit("pass exactly one of --zip or --dir")

    with open(args.critical_steps_json) as f:
        critical_steps_json = json.load(f)

    if args.zip:
        z = zipfile.ZipFile(args.zip)
        namelist = z.namelist()
        read_json = lambda name: json.load(z.open(name))  # noqa: E731
    else:
        namelist = [
            os.path.relpath(os.path.join(root, fn), args.dir)
            for root, _, files in os.walk(args.dir)
            for fn in files
        ]
        read_json = lambda name: json.load(open(os.path.join(args.dir, name)))  # noqa: E731

    records, warnings = extract(critical_steps_json, args.model, namelist, read_json)

    for w in warnings:
        print(f"WARNING: {w}")

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} critical-step records ({len(warnings)} warnings) to {args.out}")


if __name__ == "__main__":
    main()
