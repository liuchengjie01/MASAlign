#!/usr/bin/env bash
set -euo pipefail

# key = base model 路径, value = 该 base 对应的 adapter 目录列表（空格分隔）
declare -A ADAPTERS
ADAPTERS["/models/glm4v"]="/content/output/glm4v_lora_a /content/output/glm4v_lora_b"
ADAPTERS["/models/qwen2vl"]="/content/output/qwen2vl_lora_a /content/output/qwen2vl_lora_b"

OUTPUT_ROOT="/content/output/merged"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_ROOT"

for base in "${!ADAPTERS[@]}"; do
    base_name="$(basename "$base")"
    # shellcheck disable=SC2086
    for adapter in ${ADAPTERS[$base]}; do
        adapter_name="$(basename "$adapter")"
        out="${OUTPUT_ROOT}/${base_name}__${adapter_name}"

        echo "========================================"
        echo "base=$base"
        echo "adapter=$adapter"
        echo "output=$out"
        echo "========================================"

        python "$SCRIPT_DIR/merge_adapter.py" \
            --base_model "$base" \
            --adapter_dir "$adapter" \
            --output_dir "$out" \
            --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
            --device_map "${DEVICE_MAP:-cpu}"
    done
done

echo "All merges done."