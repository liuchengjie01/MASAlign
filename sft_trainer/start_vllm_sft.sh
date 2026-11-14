#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${BASE_MODEL:-E:/luis/GUI-Owl-7B}"
ADAPTER_PATH="${ADAPTER_PATH:-${SFT_DIR}/guiowl_sft_adapter}"
DEVICE_PROFILE="${DEVICE_PROFILE:-dual_4090d}"
BASE_MODEL=$(python -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])) if os.path.isdir(os.path.expanduser(sys.argv[1])) else sys.argv[1])' "${BASE_MODEL}")
ADAPTER_PATH=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=True))' "${ADAPTER_PATH}")
python "${SFT_DIR}/inspect_adapter.py" "${ADAPTER_PATH}"
REQUIRED_RANK=$(python "${SFT_DIR}/inspect_adapter.py" "${ADAPTER_PATH}" --max-rank)
MAX_LORA_RANK="${MAX_LORA_RANK:-${REQUIRED_RANK}}"
if [[ ! "${MAX_LORA_RANK}" =~ ^[0-9]+$ ]] || (( MAX_LORA_RANK < REQUIRED_RANK )); then
  echo "MAX_LORA_RANK must be an integer at least ${REQUIRED_RANK}" >&2
  exit 2
fi
echo "vLLM executable: $(command -v vllm)"
echo "Serving base=${BASE_MODEL}, adapter=${ADAPTER_PATH}, profile=${DEVICE_PROFILE}"
export PYTHONUNBUFFERED=1

case "${DEVICE_PROFILE}" in
  dual_4090d)
    TP_SIZE=2
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    ;;
  pro6000_98g|rtx6000_96g|rtx6000_98g)
    TP_SIZE=1
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    ;;
  *) echo "Unknown DEVICE_PROFILE=${DEVICE_PROFILE}" >&2; exit 2 ;;
esac

exec vllm serve "${BASE_MODEL}" \
  --served-model-name gui-owl-base \
  --enable-lora \
  --lora-modules "gui-owl-sft=${ADAPTER_PATH}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --trust-remote-code \
  --port 4244
