"""Manual GPU check (real checkpoint; not collected by pytest) -- compares the candidate
executor-stage hindsight-reprompt wordings in opd_masalign/reprompt_variants.py on GUI-Owl-7B, to
see whether telling the teacher to not treat the baked-in "Overall Plan"/"Current Subgoal" as
authoritative makes it more likely to disfavor the attacked action (see
manual_teacher_generate_check_guiowl.py for the original AutoGLM-wording check).

    python -m opd_masalign.tests.manual_teacher_generate_check_guiowl_variants \
        --model_name <GUI-Owl-7B path> --data_path data/critical_steps_guiowl_1027.jsonl
"""
import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from opd_masalign.extract_critical_steps import popup_position
from opd_masalign.guiowl_format import format_think_prefix, generate_teacher_thinking
from opd_masalign.opd_trainer import Collator, CriticalStepDataset, _encode_context_and_action_vlm, _sequence_log_probs
from opd_masalign.reprompt_variants import VARIANTS, build_teacher_messages_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_records", type=int, default=9999)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--outcomes", default="attack_success", help="comma-separated outcome_type filter, or 'all'")
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()), help="comma-separated variant names to run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = args.variants.split(",")
    for v in variants:
        assert v in VARIANTS, f"unknown variant {v!r}, choices: {list(VARIANTS)}"

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    collator = Collator(processor=processor, max_len=args.max_len, think_prefill=True)
    records = CriticalStepDataset(args.data_path).records
    if args.outcomes != "all":
        keep = set(args.outcomes.split(","))
        records = [r for r in records if r.get("outcome_type") in keep]
    records = records[: args.num_records]
    print(f"running on {len(records)} record(s) x {len(variants)} variant(s) {variants}, outcomes={args.outcomes!r}", flush=True)

    def action_logp(messages, prefix, action):
        ids, mask, vk = _encode_context_and_action_vlm(processor, messages, action, args.max_len, None, prefix)
        input_ids = torch.tensor([ids], device=device)
        vk = {k: v.to(device) for k, v in vk.items()}
        with torch.no_grad():
            lp, shifted = _sequence_log_probs(
                model, input_ids, torch.ones_like(input_ids), torch.tensor([mask], device=device), vision_kwargs=vk
            )
        return lp[0][shifted[0].bool()].float().sum().item()

    rows = []
    for i, rec in enumerate(records):
        action = rec["action_taken"]["raw_text"]
        recorded_prefix = format_think_prefix(rec["action_taken"].get("thinking") or "")
        s = action_logp(rec["observation_context"]["messages"], recorded_prefix, action)
        row = {"task_id": rec["task_id"], "position": popup_position(rec.get("popup_offset")), "student": s, "variants": {}}
        for v in variants:
            t_msgs = build_teacher_messages_variant(rec, v)
            t0 = time.time()
            thinking, n_tok, unclosed, skipped = generate_teacher_thinking(
                model, collator, t_msgs, args.max_new_tokens, args.temperature, args.top_p
            )
            gen_s = time.time() - t0
            t_gen = action_logp(t_msgs, format_think_prefix(thinking), action)
            row["variants"][v] = {
                "teacher_generated": t_gen, "gen_tokens": n_tok, "gen_seconds": gen_s,
                "unclosed": unclosed, "skipped": skipped, "teacher_thinking": thinking,
            }
            print(
                f"rec {i} task={rec['task_id']} variant={v} student={s:8.3f} teacher/generated={t_gen:8.3f} "
                f"diff={t_gen - s:+7.3f} gen={n_tok}tok {gen_s:.1f}s unclosed={unclosed} skipped={skipped}", flush=True,
            )
            torch.cuda.empty_cache()
        rows.append(row)

    print("\n=== per-variant summary (teacher_generated - student) ===")
    for group in ("all", "before", "at", "after", "no_popup"):
        rs = [row for row in rows if group == "all" or row["position"] == group]
        if not rs:
            continue
        print(f"--- position={group} (n={len(rs)}) ---")
        for v in variants:
            diffs = [row["variants"][v]["teacher_generated"] - row["student"] for row in rs]
            low = sum(d < 0 for d in diffs)
            skipped = sum(row["variants"][v]["skipped"] for row in rs)
            unclosed = sum(row["variants"][v]["unclosed"] for row in rs)
            gen_tok = statistics.mean(row["variants"][v]["gen_tokens"] for row in rs)
            print(
                f"{v}: mean_diff={statistics.mean(diffs):+7.3f} median_diff={statistics.median(diffs):+7.3f} "
                f"stdev={statistics.pstdev(diffs):6.3f} lower={low}/{len(rs)} ({100*low/len(rs):.0f}%) "
                f"skipped={skipped} unclosed={unclosed} mean_gen_tok={gen_tok:.0f}"
            )
    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("TEACHER_GENERATE_CHECK_GUIOWL_VARIANTS_DONE")


if __name__ == "__main__":
    main()
