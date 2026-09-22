"""Generate a JSONL of CriticalStepRecords using ONLY the 4 required fields
(docs/DATA_FORMAT.md §1.2), deliberately WITHOUT outcome_summary/
sibling_branches -- to prove opd_trainer.py runs end-to-end on the
truly-minimal hand-off contract (old_action_logprob is never read from the
JSONL at all -- opd_trainer.py self-computes it, see
OPDTrainer.precompute_old_log_probs).

Usage:
    python -m opd_masalign.tests.generate_minimal_smoke_data \
        --out data/critical_steps_minimal_smoke.jsonl --n 8
"""
from __future__ import annotations

import argparse
import json


def _obs(text: str) -> dict:
    return {"format": "chat_messages", "messages": [
        {"role": "system", "content": "You are a mobile agent completing a user's task safely."},
        {"role": "user", "content": text},
    ]}


def build_record(i: int) -> dict:
    record = {
        "schema_version": "1.1",
        "record_type": "critical_step",
        "task_id": f"task-min-{i}",
        "observation_context": _obs("Chat app open, a popup covers the reply box"),
        "action_taken": {"raw_text": "tap 'Update Now' on the popup"},
        "outcome_type": "attack_success" if i % 2 == 0 else "benign_success",
    }
    # only give the recommended free-text fields to half the records, to
    # also exercise the generic-fallback paths in self_teacher.py.
    if i % 2 == 0:
        record["outcome_summary"] = "The fake update button installed a malicious APK in the background."
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    with open(args.out, "w") as f:
        for i in range(args.n):
            f.write(json.dumps(build_record(i)) + "\n")
    print(f"wrote {args.n} minimal critical-step records to {args.out}")


if __name__ == "__main__":
    main()
