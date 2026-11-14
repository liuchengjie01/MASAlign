#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${SFT_DIR}/start_sft.sh"
