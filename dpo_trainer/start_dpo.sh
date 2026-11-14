#!/usr/bin/env bash
set -euo pipefail

DPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${DPO_DIR}/config.yaml}"
cd "${DPO_DIR}/.."

PROFILE_ARGS=()
if [[ -n "${DEVICE_PROFILE:-}" ]]; then
  PROFILE_ARGS=(--device_profile "${DEVICE_PROFILE}")
fi
if ! NPROC=$(python "${DPO_DIR}/configuration.py" --config "${CONFIG_PATH}" "${PROFILE_ARGS[@]}" --num-processes); then
  echo "Failed to read GPU process count: config=${CONFIG_PATH}, profile=${DEVICE_PROFILE:-from YAML}" >&2
  exit 2
fi
if [[ "${NPROC}" != 1 && "${NPROC}" != 2 ]]; then
  echo "configuration.py --num-processes must print 1 or 2; got '${NPROC}'" >&2
  echo "Check the uploaded configuration.py CLI and hardware_profiles.*.num_processes in ${CONFIG_PATH}" >&2
  exit 2
fi
echo "Resolved DPO config=${CONFIG_PATH}, profile=${DEVICE_PROFILE:-from YAML}, processes=${NPROC}"
if [[ "${NPROC}" == 2 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Execute the actual uploaded train.py entry before importing torch or models.
if ! CHECK_OUTPUT=$(python -u "${DPO_DIR}/train.py" --config "${CONFIG_PATH}" "${PROFILE_ARGS[@]}" --check-config); then
  echo "${CHECK_OUTPUT}" >&2
  echo "DPO startup configuration check failed" >&2
  exit 2
fi
echo "${CHECK_OUTPUT}"
if [[ "${CHECK_OUTPUT}" != *"DPO configuration check passed"* ]]; then
  echo "train.py did not execute the startup check. Check the uploaded file and its main() entry: ${DPO_DIR}/train.py" >&2
  exit 2
fi

python -c 'import sys, torch; expected=int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count() == expected, f"Expected {expected} GPUs"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(expected)])' "${NPROC}"
echo "Starting DPO in foreground: config=${CONFIG_PATH}, processes=${NPROC}"
if [[ "${NPROC}" == 1 ]]; then
  TRAIN_COMMAND=(python -u "${DPO_DIR}/train.py")
else
  TRAIN_COMMAND=(python -u -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" "${DPO_DIR}/train.py")
fi
TRAIN_COMMAND+=(--config "${CONFIG_PATH}" "${PROFILE_ARGS[@]}")
echo "Command: ${TRAIN_COMMAND[*]}"
EXIT_CODE=0
"${TRAIN_COMMAND[@]}" || EXIT_CODE=$?
echo "DPO process exited with code ${EXIT_CODE}"
exit "${EXIT_CODE}"
