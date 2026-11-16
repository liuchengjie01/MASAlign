import argparse
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--base_model", required=True, help="Base model path or HF id")
    parser.add_argument("--adapter_dir", required=True, help="LoRA adapter directory")
    parser.add_argument("--output_dir", required=True, help="Where to save the merged model")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", default="cpu", help="cpu or auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def main():
    args = parse_args()
    BASE_MODEL = args.base_model
    ADAPTER_DIR = args.adapter_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 1. 加载 base model（CPU 上合并更省显存，dtype 用 bf16/fp16 与训练一致）
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # 合并阶段不需要 GPU
        trust_remote_code=True,
    )

    # 2. 挂载 adapter
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    # 3. 合并并卸载 adapter
    model = model.merge_and_unload()

    # 4. 保存合并后的完整模型
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="30GB")

    # 5. processor/tokenizer 一起保存（推理需要）
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.save_pretrained(OUTPUT_DIR)
    print("merged model saved to", OUTPUT_DIR)



    base_model_path = BASE_MODEL
    merged_model_path = OUTPUT_DIR

    # 1. 加载基础模型（只加载到 CPU 看权重，不占显存）
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    # 2. 加载合并后的模型
    merged_model = AutoModelForImageTextToText.from_pretrained(
        merged_model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    # 3. 找一层被 LoRA 修改过的层，比如 q_proj
    target_layer = "model.language_model.layers.0.self_attn.q_proj"

    base_weight = dict(base_model.named_parameters())[f"{target_layer}.weight"]
    merged_weight = dict(merged_model.named_parameters())[f"{target_layer}.weight"]

    # 4. 计算两个权重的差异
    diff = (base_weight != merged_weight).sum().item()
    total_elements = base_weight.numel()

    print(f"目标层: {target_layer}")
    print(f"原始权重形状: {base_weight.shape}")
    print(f"合并后权重形状: {merged_weight.shape}")
    print(f"---")
    print(f"该层共有 {total_elements} 个参数，其中有 {diff} 个参数的值发生了变化。")

    if diff > 0:
        print("✅ 验证成功：权重数值发生了变化，LoRA 已经成功合并进原模型！")
    else:
        print("❌ 验证失败：权重完全一样，合并可能失败了（或者选错了层）。")