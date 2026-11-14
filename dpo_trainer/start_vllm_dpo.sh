#!/usr/bin/env bash
set -euo pipefail

DPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${BASE_MODEL:-E:/luis/GUI-Owl-7B}"
ADAPTER_PATH="${ADAPTER_PATH:-${DPO_DIR}/guiowl_dpo_ckpt/adapter}"
DEVICE_PROFILE="${DEVICE_PROFILE:-dual_4090d}"

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

vllm serve "${BASE_MODEL}" \
    --served-model-name gui-owl-base \
    --enable-lora \
    --lora-modules "gui-owl-dpo=${ADAPTER_PATH}" \
    --max-lora-rank 32 \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --port 4245
