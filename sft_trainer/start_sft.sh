#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SFT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SFT_DIR}/config.yaml}"
DEVICE_PROFILE="${DEVICE_PROFILE:-dual_4090d}"

case "${DEVICE_PROFILE}" in
  dual_4090d)
    NPROC=2
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    ;;
  rtx6000_96g|rtx6000_98g)
    DEVICE_PROFILE="rtx6000_96g"
    NPROC=1
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    ;;
  *)
    echo "Unknown DEVICE_PROFILE=${DEVICE_PROFILE}" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

python -c 'import accelerate, peft, sys, torch, transformers, yaml; expected=int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA is unavailable"; assert torch.cuda.device_count() == expected, f"Expected {expected} visible GPUs, got {torch.cuda.device_count()}"; print(f"torch={torch.__version__}, transformers={transformers.__version__}, accelerate={accelerate.__version__}, peft={peft.__version__}"); print("GPUs:", [torch.cuda.get_device_name(i) for i in range(expected)])' "${NPROC}"

python -c 'import os, sys, yaml; cfg=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); missing=[p for p in (cfg["model"]["path"], cfg["data"]["train_jsonl"]) if not os.path.exists(p)]; assert not missing, "Missing configured paths (run from MASAlign root or edit config.yaml): " + ", ".join(missing)' "${CONFIG_PATH}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${SFT_DIR}/train.py" \
  --config "${CONFIG_PATH}" \
  --device_profile "${DEVICE_PROFILE}"
