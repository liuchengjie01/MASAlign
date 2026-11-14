#!/usr/bin/env bash
set -euo pipefail

DPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${DPO_DIR}/config_long_context.yaml"

exec bash "${DPO_DIR}/start_dpo.sh"
