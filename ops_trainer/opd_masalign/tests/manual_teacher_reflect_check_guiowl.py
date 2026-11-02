"""Manual GPU check (real checkpoint; not collected by pytest) -- does a hindsight reprompt on the
GUI-Owl *reflector* sub-call make it more willing to call the critical step's action a failure
("B: wrong page, need to return")? The reflector runs AFTER the action, on the before/after
screenshots, and its reply starts with the outcome letter -- there is no reasoning section ahead of
it, so unlike the executor/planner checks the teacher does not write its own thinking first: the
A/B/C letter is scored directly after "### Outcome ###\\n", once under the original
reflect_messages (student) and once with the reflector-stage reprompt appended (teacher).

Reports P(A)/P(B)/P(C) renormalized over the three letters, the log-prob of the letter the reflector
actually logged, and the same split by position relative to the popup (analysis only, see
extract_critical_steps.popup_position). Optionally samples one full teacher reflection per record
(--gen_tokens > 0) to read what its Error Description says.

    python -m opd_masalign.tests.manual_teacher_reflect_check_guiowl \\
        --model_name <GUI-Owl-7B path> --data_path data/critical_steps_guiowl_v2_1030.jsonl \\
        --zip data/luis/gui-owl-7b.zip
"""
import argparse
import json
import re
import statistics
import zipfile
from collections import defaultdict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from opd_masalign.extract_critical_steps import _locate_critical_step, popup_position
from opd_masalign.opd_trainer import CriticalStepDataset, _encode_context_and_action_vlm
from opd_masalign.reprompt_variants import REFLECTOR_VARIANTS, build_reflector_reprompt_text_variant

OUTCOME_PREFIX = "### Outcome ###\n"
LETTERS = ("A", "B", "C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--zip", required=True, help="raw rollout zip, to recover reflect_messages for matched records")
    ap.add_argument("--num_records", type=int, default=9999)
    ap.add_argument("--max_len", type=int, default=16384, help="reflect_messages carry two full screenshots")
    ap.add_argument("--outcomes", default="attack_success", help="comma-separated outcome_type filter, or 'all'")
    ap.add_argument("--reprompt_variant", default="g1", choices=list(REFLECTOR_VARIANTS))
    ap.add_argument("--gen_tokens", type=int, default=200, help="0 disables sampling a full teacher reflection")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    letter_ids = [tok(l, add_special_tokens=False)["input_ids"] for l in LETTERS]
    assert all(len(i) == 1 for i in letter_ids), f"outcome letters are not single tokens: {letter_ids}"
    letter_ids = [i[0] for i in letter_ids]

    records = CriticalStepDataset(args.data_path).records
    if args.outcomes != "all":
        keep = set(args.outcomes.split(","))
        records = [r for r in records if r.get("outcome_type") in keep]
    records = records[: args.num_records]
    print(f"running on {len(records)} record(s), outcomes={args.outcomes!r}, reprompt_variant={args.reprompt_variant!r}", flush=True)

    @torch.no_grad()
    def letter_probs(messages):
        ids, _, vk = _encode_context_and_action_vlm(processor, messages, "", args.max_len, None, OUTCOME_PREFIX)
        input_ids = torch.tensor([ids], device=device)
        vk = {k: v.to(device) for k, v in vk.items()}
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **vk).logits[0, -1].float()
        logp = torch.log_softmax(logits, -1)[letter_ids]
        return dict(zip(LETTERS, torch.softmax(logp, -1).tolist())), dict(zip(LETTERS, logp.tolist()))

    @torch.no_grad()
    def sample_reflection(messages):
        ids, _, vk = _encode_context_and_action_vlm(processor, messages, "", args.max_len, None, OUTCOME_PREFIX)
        input_ids = torch.tensor([ids], device=device)
        vk = {k: v.to(device) for k, v in vk.items()}
        out = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **vk,
            max_new_tokens=args.gen_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p,
            pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=True,
        )
        return OUTCOME_PREFIX + tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    rows = []
    skipped = 0
    for i, rec in enumerate(records):
        rollout = json.loads(z.read(rec["source"]))
        step_idx = rec.get("step_idx")
        if step_idx is None:
            step_idx = _locate_critical_step(rollout, rec["state_u"])
        step = rollout[step_idx] if step_idx is not None else {}
        if "reflect_messages" not in step:
            skipped += 1
            print(f"rec {i} task={rec['task_id']}: no reflect_messages at matched step (last step?), skipping", flush=True)
            continue
        m = re.search(r"### Outcome ###\s*([ABC])", step.get("reflect_response", ""))
        logged = m.group(1) if m else None

        student_msgs = step["reflect_messages"]
        teacher_msgs = list(student_msgs) + [
            {"role": "user", "content": build_reflector_reprompt_text_variant(rec, args.reprompt_variant)}
        ]
        s_p, s_lp = letter_probs(student_msgs)
        t_p, t_lp = letter_probs(teacher_msgs)
        row = {
            "task_id": rec["task_id"], "source": rec["source"], "step_idx": step_idx,
            "position": popup_position(rec.get("popup_offset")), "logged": logged,
            "student_p": s_p, "teacher_p": t_p,
            "logged_logp_diff": (t_lp[logged] - s_lp[logged]) if logged else None,
            "logged_reflection": step.get("reflect_response", ""),
        }
        if args.gen_tokens:
            row["teacher_reflection"] = sample_reflection(teacher_msgs)
        rows.append(row)
        print(
            f"rec {i} task={rec['task_id']} pos={row['position']} logged={logged} "
            f"student P(A/B/C)={s_p['A']:.2f}/{s_p['B']:.2f}/{s_p['C']:.2f} "
            f"teacher P(A/B/C)={t_p['A']:.2f}/{t_p['B']:.2f}/{t_p['C']:.2f}", flush=True,
        )
        if args.gen_tokens:
            print(f"    teacher reflection: {row['teacher_reflection'][:300]!r}", flush=True)
        torch.cuda.empty_cache()

    print(f"\nskipped (no reflect_messages at matched step): {skipped}")
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["position"]].append(r)
    for name in ("all", "before", "at", "after", "no_popup"):
        rs = groups.get(name)
        if not rs:
            continue
        print(f"\n=== {name} (n={len(rs)}) ===")
        for who in ("student", "teacher"):
            mean = {l: statistics.mean(r[f"{who}_p"][l] for r in rs) for l in LETTERS}
            argmax = {l: sum(max(r[f"{who}_p"], key=r[f"{who}_p"].get) == l for r in rs) for l in LETTERS}
            print(f"{who:8s} mean P(A/B/C)={mean['A']:.3f}/{mean['B']:.3f}/{mean['C']:.3f}  "
                  f"argmax A/B/C={argmax['A']}/{argmax['B']}/{argmax['C']}")
        diffs = [r["logged_logp_diff"] for r in rs if r["logged_logp_diff"] is not None]
        if diffs:
            low = sum(d < 0 for d in diffs)
            print(f"logged letter, teacher - student log-prob: mean={statistics.mean(diffs):+.3f} "
                  f"lower={low}/{len(diffs)} ({100 * low / len(diffs):.0f}%)")
    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("TEACHER_REFLECT_CHECK_GUIOWL_DONE")


if __name__ == "__main__":
    main()
