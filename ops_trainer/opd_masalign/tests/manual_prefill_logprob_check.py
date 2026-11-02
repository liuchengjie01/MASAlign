"""Manual GPU check (needs the real checkpoint; not collected by pytest):
does scoring the action AFTER the model's real `<think>..</think><answer>` prefix
(opd_trainer.py --think_prefill) put the action tokens at sensible log-probs,
compared with the legacy position straight after `<|assistant|>\\n`?

    python -m opd_masalign.tests.manual_prefill_logprob_check \
        --model_name <AutoGLM-Phone-9B path> --data_path data/critical_steps_autoglm_popup_test20.jsonl
"""
import argparse
import json
import statistics

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from opd_masalign.opd_trainer import (
    CriticalStepDataset,
    _encode_context_and_action_vlm,
    _sequence_log_probs,
    _think_prefix,
)
from opd_masalign.self_teacher import build_teacher_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_records", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--out", default=None, help="optional JSON path for per-record numbers")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    records = CriticalStepDataset(args.data_path).records[: args.num_records]

    stats = {}
    per_record = []
    for i, record in enumerate(records):
        action = record["action_taken"]["raw_text"]
        for ctx_name, messages in (
            ("student", record["observation_context"]["messages"]),
            ("teacher", build_teacher_messages(record)),
        ):
            for mode, prefix in (("legacy", ""), ("think_prefill", _think_prefix(record))):
                ids, mask, vk = _encode_context_and_action_vlm(
                    processor, messages, action, args.max_len, None, prefix
                )
                input_ids = torch.tensor([ids], device=device)
                attn = torch.ones_like(input_ids)
                rmask = torch.tensor([mask], device=device)
                vk = {k: v.to(device) for k, v in vk.items()}
                with torch.no_grad():
                    lp, shifted = _sequence_log_probs(model, input_ids, attn, rmask, vision_kwargs=vk)
                resp = lp[0][shifted[0].bool()].float().cpu()
                stats.setdefault((ctx_name, mode), []).append((resp[0].item(), resp.sum().item(), len(ids)))
                print(
                    f"rec {i} task={record['task_id']} {ctx_name:<7} {mode:<13} seq_len={len(ids):>5} "
                    f"first_action_token_logp={resp[0].item():8.3f} action_sum_logp={resp.sum().item():8.3f}",
                    flush=True,
                )
        torch.cuda.empty_cache()
        s_sum = stats[("student", "think_prefill")][-1][1]
        t_sum = stats[("teacher", "think_prefill")][-1][1]
        per_record.append({"task_id": record["task_id"], "student_sum_logp": s_sum, "teacher_sum_logp": t_sum,
                           "teacher_minus_student": t_sum - s_sum})
        print(f"rec {i} task={record['task_id']} teacher - student action logp (think_prefill) = {t_sum - s_sum:+.3f}", flush=True)

    print("\n=== mean over records ===")
    for (ctx_name, mode), rows in sorted(stats.items()):
        print(
            f"{ctx_name:<7} {mode:<13} first_token_logp={statistics.mean(r[0] for r in rows):8.3f} "
            f"action_sum_logp={statistics.mean(r[1] for r in rows):8.3f}"
        )
    diffs = [r["teacher_minus_student"] for r in per_record]
    print(
        f"\nteacher - student (think_prefill): mean={statistics.mean(diffs):+.3f} median={statistics.median(diffs):+.3f} "
        f"teacher lower on {sum(d < 0 for d in diffs)}/{len(diffs)} records"
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(per_record, f, indent=1)
    print("PREFILL_LOGPROB_CHECK_DONE")


if __name__ == "__main__":
    main()
