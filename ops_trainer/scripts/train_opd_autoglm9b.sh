#!/bin/bash
# Full OPD training on AutoGLM-Phone-9B, 8 GPUs, FSDP2. Run from the project root after
# scripts/extract_critical_steps.sh (the data check must print OK).
# Override with environment variables, e.g.
#   MODEL=/juicefs/models/AutoGLM-Phone-9B DATA=data/critical_steps_0925.jsonl \
#   OUT=/tmp/opd_autoglm_epoch3_0925 EPOCHS=3 bash scripts/train_opd_autoglm9b.sh
set -euo pipefail

MODEL=${MODEL:-/juicefs/models/AutoGLM-Phone-9B}
DATA=${DATA:-data/critical_steps_0925.jsonl}
OUT=${OUT:-/tmp/opd_autoglm_epoch3_0925}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-6}
GPUS=${GPUS:-8}
REPROMPT=${REPROMPT:-t1}  # teacher hindsight reprompt (opd_masalign/reprompt_variants.py); v0 = old popup-only one

python -m opd_masalign.check_dataset "$DATA"

torchrun --standalone --nproc_per_node="$GPUS" -m opd_masalign.opd_trainer --vlm \
  --model_name "$MODEL" --data_path "$DATA" --output_dir "$OUT" \
  --dtype bf16 --disable_image_splitting --image_max_side 512 \
  --gradient_checkpointing --think_prefill \
  --epochs "$EPOCHS" --max_steps 100000 --max_len 8192 \
  --lr "$LR" --max_grad_norm 1.0 --save_every 20 \
  --model_format autoglm --reprompt_variant "$REPROMPT"
