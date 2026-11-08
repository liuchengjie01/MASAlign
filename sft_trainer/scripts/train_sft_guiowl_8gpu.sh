#!/usr/bin/env bash
set -e

SFT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd -- "$SFT_DIR/.." && pwd)

cd "$REPO_ROOT"

torchrun --standalone --nproc_per_node=8 \
  "$SFT_DIR/train.py" \
  --config "$SFT_DIR/config.yaml"
