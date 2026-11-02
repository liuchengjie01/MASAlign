"""Manual GPU check (real checkpoint; not collected by pytest) -- GUI-Owl-7B variant of
manual_teacher_generate_check.py. Same question (does the EMA teacher's self-generated,
hindsight-reprompted reasoning actually make it disfavor the attacked action more than the
student does?), on a different model/data format (see extract_critical_steps.py --format guiowl
and opd_masalign/guiowl_format.py for what differs and why).

    python -m opd_masalign.tests.manual_teacher_generate_check_guiowl \
        --model_name <GUI-Owl-7B path> --data_path data/critical_steps_guiowl_1027.jsonl
"""
import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from opd_masalign.guiowl_format import format_think_prefix, generate_teacher_thinking
from opd_masalign.opd_trainer import Collator, CriticalStepDataset, _encode_context_and_action_vlm, _sequence_log_probs
from opd_masalign.self_teacher import build_teacher_messages


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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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
    print(f"running on {len(records)} record(s), outcomes={args.outcomes!r}", flush=True)

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
        t_msgs = build_teacher_messages(rec)
        t0 = time.time()
        thinking, n_tok, unclosed, skipped = generate_teacher_thinking(
            model, collator, t_msgs, args.max_new_tokens, args.temperature, args.top_p
        )
        gen_s = time.time() - t0
        s = action_logp(rec["observation_context"]["messages"], format_think_prefix(rec["action_taken"].get("thinking") or ""), action)
        t_rec = action_logp(t_msgs, format_think_prefix(rec["action_taken"].get("thinking") or ""), action)
        t_gen = action_logp(t_msgs, format_think_prefix(thinking), action)
        rows.append({
            "task_id": rec["task_id"], "student": s, "teacher_recorded": t_rec, "teacher_generated": t_gen,
            "gen_tokens": n_tok, "gen_seconds": gen_s, "unclosed": unclosed, "skipped": skipped, "teacher_thinking": thinking,
            "recorded_thinking": rec["action_taken"].get("thinking"),
        })
        print(
            f"rec {i} task={rec['task_id']} student={s:8.3f} teacher/recorded={t_rec:8.3f} "
            f"teacher/generated={t_gen:8.3f} gen={n_tok}tok {gen_s:.1f}s unclosed={unclosed} skipped={skipped}", flush=True,
        )
        print(f"    teacher thinking: {thinking[:300]!r}", flush=True)
        torch.cuda.empty_cache()

    def mean(k):
        return statistics.mean(r[k] for r in rows)

    print("\n=== mean action log-prob (sum over action tokens) ===")
    print(f"student                     {mean('student'):8.3f}")
    print(f"teacher, recorded reasoning {mean('teacher_recorded'):8.3f}")
    print(f"teacher, generated reasoning{mean('teacher_generated'):8.3f}")
    low = sum(r["teacher_generated"] < r["student"] for r in rows)
    print(f"teacher(generated) lower than student on {low}/{len(rows)} records")
    print(f"generation: mean {mean('gen_tokens'):.0f} tokens, {mean('gen_seconds'):.1f}s each, unclosed {sum(r['unclosed'] for r in rows)}, skipped(empty reasoning) {sum(r['skipped'] for r in rows)}")
    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("TEACHER_GENERATE_CHECK_GUIOWL_DONE")


if __name__ == "__main__":
    main()
