#!/bin/bash
# OPD training on GUI-Owl-7B (Mobile-Agent-V3 executor), FSDP2. Run from the project root.
# Data: extract_critical_steps.py --format guiowl (records must carry planner_context for --teacher_replan;
# data/critical_steps_guiowl_1027_train.jsonl is the 1027 set with it added).
# Teacher = planner replan under the t1 planner reprompt, then the teacher writes its own thinking in the
# replanned executor context (no executor-stage reprompt -- those raised the attacked action's log-prob on
# GUI-Owl-7B). Override with environment variables, e.g.
#   MODEL=/path/to/GUI-Owl-7B OUT=/tmp/opd_guiowl GPUS=4 bash scripts/train_opd_guiowl7b.sh
set -euo pipefail

MODEL=${MODEL:-$(find "$HOME/.cache/huggingface/hub/models--mPLUG--GUI-Owl-7B/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)}
DATA=${DATA:-data/critical_steps_guiowl_1027_train.jsonl}
OUT=${OUT:-/tmp/opd_guiowl7b}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-6}
GPUS=${GPUS:-8}
PLANNER=${PLANNER:-t1}   # planner-stage reprompt (reprompt_variants.PLANNER_VARIANTS)
REPROMPT=${REPROMPT:-t1} # executor-stage reprompt, only used if a replan comes out empty

python -m opd_masalign.check_dataset "$DATA"

torchrun --standalone --nproc_per_node="$GPUS" -m opd_masalign.opd_trainer --vlm \
  --model_name "$MODEL" --data_path "$DATA" --output_dir "$OUT" \
  --dtype bf16 --disable_image_splitting \
  --gradient_checkpointing --think_prefill \
  --model_format guiowl --teacher_replan --planner_variant "$PLANNER" --reprompt_variant "$REPROMPT" \
  --epochs "$EPOCHS" --max_steps 100000 --max_len 8192 --batch_size 1 \
  --lr "$LR" --max_grad_norm 1.0 --save_every 20
