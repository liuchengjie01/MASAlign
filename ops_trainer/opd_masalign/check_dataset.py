"""Sanity-check a CriticalStepRecord JSONL before training: schema validity, outcome mix, whether
`action_taken.thinking` is present (needed by opd_trainer.py --think_prefill), and exactly one
image per record (the VLM Collator requires it).

    python -m opd_masalign.check_dataset data/critical_steps_0925.jsonl
"""
import argparse
import json
import sys
from collections import Counter

from opd_masalign.schema import validate_critical_step_record


def _count_images(messages) -> int:
    n = 0
    for m in messages:
        if isinstance(m.get("content"), list):
            n += sum(1 for b in m["content"] if b.get("type") == "image_url")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    n = 0
    outcomes, image_counts = Counter(), Counter()
    missing_thinking = invalid = 0
    for line in open(args.path):
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        errors = [e for e in validate_critical_step_record(rec) if not e.startswith("note:")]
        if errors:
            invalid += 1
            print(f"record {n} (task_id={rec.get('task_id')!r}) invalid: {errors}")
        outcomes[rec.get("outcome_type")] += 1
        if rec.get("action_taken", {}).get("thinking") is None:
            missing_thinking += 1
        image_counts[_count_images(rec.get("observation_context", {}).get("messages", []))] += 1

    print(f"{args.path}: {n} records")
    print(f"  outcome_type: {dict(outcomes)}")
    print(f"  images per record: {dict(image_counts)}  (the --vlm Collator needs exactly 1)")
    print(f"  missing action_taken.thinking: {missing_thinking}/{n}  (needed by --think_prefill)")
    print(f"  schema-invalid: {invalid}/{n}")
    ok = n > 0 and not invalid and not missing_thinking and set(image_counts) == {1}
    print("OK" if ok else "NOT READY for --vlm --think_prefill training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
