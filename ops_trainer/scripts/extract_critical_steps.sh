#!/bin/bash
# Raw collaborator rollout zips -> CriticalStepRecord JSONL, then sanity-check the result.
# Run from the project root. Override the paths with environment variables, e.g.
#   ZIPS="data/a.zip data/b.zip" CRITICAL_STEPS_JSON=data/critical_steps.json \
#   OUT=data/critical_steps_0925.jsonl bash scripts/extract_critical_steps.sh
set -euo pipefail

ZIPS=${ZIPS:-"data/autoglm-phone-9b.zip data/autoglm-phone-9b-part10.zip"}
CRITICAL_STEPS_JSON=${CRITICAL_STEPS_JSON:-data/critical_steps.json}
MODEL=${MODEL:-autoglm-phone-9b}
OUT=${OUT:-data/critical_steps_0925.jsonl}

python -m opd_masalign.extract_critical_steps \
  --zip $ZIPS \
  --critical_steps_json "$CRITICAL_STEPS_JSON" \
  --model "$MODEL" \
  --out "$OUT"

python -m opd_masalign.check_dataset "$OUT"
