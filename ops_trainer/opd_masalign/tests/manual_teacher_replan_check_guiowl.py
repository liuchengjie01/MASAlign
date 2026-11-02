"""Manual GPU check (real checkpoint; not collected by pytest) -- does reprompting the GUI-Owl
*planner* sub-call (not just the executor, see manual_teacher_generate_check_guiowl_variants.py)
make the teacher favor the attacked action less? For each of the 60 attack_success records:

  1. Re-locate the matching raw rollout step (same source/state_u already recorded in the
     dataset) to recover its `planning_messages` (not kept in the CriticalStepRecord schema).
  2. Teacher rewrites "### Plan ###" under planning_messages + a PLANNER-stage hindsight
     reprompt (reprompt_variants.build_planner_reprompt_text_variant -- correctly references the
     planner prompt's own "### Plan ###" header; the earlier version of this script reused the
     executor-stage wording, which names "Overall Plan"/"Current Subgoal" sections that don't
     exist in planning_messages at all, so that clause was likely inert).
  3. Substitute that new plan into the executor's "### Overall Plan ###"/"### Current
     Subgoal ###" (replan.substitute_plan), keeping the screenshot/guidelines/etc. untouched.
  4. Score two things against the untouched-plan student baseline:
     a. student_replanned_forced: recorded action+thinking, but under the REPLANNED executor
        context, no additional reprompt -- isolates "does fixing the plan alone move the
        needle", no teacher executor call involved.
     b. teacher_replanned_generated: teacher writes its own new Thought+Action under the
        replanned context alone (replan-then-act, no executor-stage reprompt; stacking one on
        top was tried and consistently weakened the effect, so it was dropped).

    python -m opd_masalign.tests.manual_teacher_replan_check_guiowl \
        --model_name <GUI-Owl-7B path> --data_path data/critical_steps_guiowl_1027.jsonl \
        --zip data/luis/gui-owl-7b.zip
"""
import argparse
import json
import statistics
import time
import zipfile

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from opd_masalign.extract_critical_steps import _locate_critical_step, popup_position
from opd_masalign.guiowl_format import format_think_prefix, generate_teacher_thinking
from opd_masalign.opd_trainer import Collator, CriticalStepDataset, _encode_context_and_action_vlm, _sequence_log_probs
from opd_masalign.replan import generate_teacher_replan, substitute_plan
from opd_masalign.reprompt_variants import PLANNER_VARIANTS, build_planner_reprompt_text_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--zip", required=True, help="raw rollout zip, to recover planning_messages for matched records")
    ap.add_argument("--num_records", type=int, default=9999)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--replan_max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--outcomes", default="attack_success", help="comma-separated outcome_type filter, or 'all'")
    ap.add_argument("--reprompt_variant", default="v2", choices=list(PLANNER_VARIANTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)

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
    print(f"running on {len(records)} record(s), outcomes={args.outcomes!r}, reprompt_variant={args.reprompt_variant!r}", flush=True)

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
    skipped_no_planning = 0
    for i, rec in enumerate(records):
        rollout = json.loads(z.read(rec["source"]))
        step_idx = rec.get("step_idx")
        if step_idx is None:
            step_idx = _locate_critical_step(rollout, rec["state_u"])
        if step_idx is None or "planning_messages" not in rollout[step_idx]:
            skipped_no_planning += 1
            print(f"rec {i} task={rec['task_id']}: no planning_messages found at matched step, skipping", flush=True)
            continue
        planning_messages = rollout[step_idx]["planning_messages"]

        action = rec["action_taken"]["raw_text"]
        recorded_prefix = format_think_prefix(rec["action_taken"].get("thinking") or "")
        orig_messages = rec["observation_context"]["messages"]
        s = action_logp(orig_messages, recorded_prefix, action)

        replan_msgs = list(planning_messages) + [{"role": "user", "content": build_planner_reprompt_text_variant(rec, args.reprompt_variant)}]
        t0 = time.time()
        new_plan, raw_plan_text, plan_tok = generate_teacher_replan(
            model, collator, replan_msgs, args.replan_max_new_tokens, args.temperature, args.top_p
        )
        replan_s = time.time() - t0
        replanned_messages = substitute_plan(orig_messages, new_plan) if new_plan else orig_messages

        s_replanned_forced = action_logp(replanned_messages, recorded_prefix, action)

        t0 = time.time()
        thinking, n_tok, unclosed, skipped = generate_teacher_thinking(
            model, collator, replanned_messages, args.max_new_tokens, args.temperature, args.top_p
        )
        act_s = time.time() - t0
        t_replanned_generated = action_logp(replanned_messages, format_think_prefix(thinking), action)

        row = {
            "task_id": rec["task_id"], "position": popup_position(rec.get("popup_offset")), "student": s,
            "student_replanned_forced": s_replanned_forced,
            "teacher_replanned_generated": t_replanned_generated,
            "new_plan": new_plan, "plan_tokens": plan_tok, "replan_seconds": replan_s,
            "teacher_thinking": thinking, "act_tokens": n_tok, "act_seconds": act_s,
            "unclosed": unclosed, "skipped": skipped,
        }
        rows.append(row)
        print(
            f"rec {i} task={rec['task_id']} student={s:8.3f} "
            f"student/replanned={s_replanned_forced:8.3f} (diff={s_replanned_forced-s:+7.3f}) "
            f"teacher/replan+act={t_replanned_generated:8.3f} (diff={t_replanned_generated-s:+7.3f}) "
            f"plan={plan_tok}tok act={n_tok}tok", flush=True,
        )
        print(f"    new plan: {new_plan[:200]!r}", flush=True)
        torch.cuda.empty_cache()

    print(f"\nskipped (no planning_messages at matched step): {skipped_no_planning}")
    for group in ("all", "before", "at", "after", "no_popup"):
        rs = [r for r in rows if group == "all" or r["position"] == group]
        if not rs:
            continue
        for key, label in [("student_replanned_forced", "student, replanned plan, recorded action/thinking forced"),
                            ("teacher_replanned_generated", "teacher, replanned plan, own generated thinking+action")]:
            diffs = [r[key] - r["student"] for r in rs]
            low = sum(d < 0 for d in diffs)
            print(f"\n=== position={group} (n={len(rs)}): {label} vs original-plan student ===")
            print(f"mean_diff={statistics.mean(diffs):+7.3f} median_diff={statistics.median(diffs):+7.3f} "
                  f"stdev={statistics.pstdev(diffs):6.3f} lower={low}/{len(rs)} ({100*low/len(rs):.0f}%)")
    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("TEACHER_REPLAN_CHECK_GUIOWL_DONE")


if __name__ == "__main__":
    main()
