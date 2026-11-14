"""Read-only checks for a saved PEFT adapter before starting vLLM."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import struct
import sys


def inspect_adapter(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    with (directory / "adapter_config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Expected a LORA adapter, got {config.get('peft_type')!r}")
    weights = directory / "adapter_model.safetensors"
    with weights.open("rb") as file:
        header_size_bytes = file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {weights}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if not 0 < header_size <= min(weights.stat().st_size - 8, 16 * 1024 * 1024):
            raise ValueError(f"Invalid safetensors header size: {header_size}")
        tensors = json.loads(file.read(header_size))
    tensors.pop("__metadata__", None)
    if not tensors:
        raise ValueError("Adapter weights contain no tensors")
    payload_size = weights.stat().st_size - 8 - header_size
    for name, tensor in tensors.items():
        start, end = tensor["data_offsets"]
        if not 0 <= start <= end <= payload_size:
            raise ValueError(f"Invalid tensor offsets for {name}: weights may be truncated")
    ranks = [config["r"], *(config.get("rank_pattern") or {}).values()]
    if not all(isinstance(rank, int) and rank > 0 for rank in ranks):
        raise ValueError(f"Invalid LoRA ranks: {ranks}")
    max_rank = max(ranks)
    # vLLM accepts discrete rank capacities; round up to the nearest one.
    capacity = next((rank for rank in (8, 16, 32, 64, 128, 256, 320, 512) if rank >= max_rank), None)
    if capacity is None:
        raise ValueError(f"LoRA rank {max_rank} exceeds this launcher's supported capacities")
    return directory, config, weights, tensors, capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_path")
    parser.add_argument("--max-rank", action="store_true", help="Print only the required vLLM rank capacity")
    args = parser.parse_args()
    directory, config, weights, tensors, capacity = inspect_adapter(args.adapter_path)
    if args.max_rank:
        print(capacity)
        return
    print(f"Python: {sys.executable}")
    for package in ("vllm", "peft", "safetensors"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: not installed in this Python environment")
    print(f"Adapter directory: {directory}")
    print(f"Readable weights: {weights} ({weights.stat().st_size} bytes; {len(tensors)} tensors)")
    print(f"Recorded base model: {config.get('base_model_name_or_path')}")
    print(f"LoRA rank: {config['r']}; required --max-lora-rank: {capacity}")
    print(f"Target modules: {config.get('target_modules')}")
    print("First tensor names:")
    for name in list(tensors)[:4]:
        print(f"  {name}: {tensors[name]['shape']}")
    print("Adapter files are readable; vLLM worker compatibility still depends on its version and base model.")


if __name__ == "__main__":
    main()
