"""Read YAML and apply the selected GPU profile in one place."""

import copy
import inspect
import os
from pathlib import Path

import yaml


PROFILE_ALIASES = {"rtx6000_96g": "pro6000_98g", "rtx6000_98g": "pro6000_98g"}


def merge_config(config, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and value and isinstance(config.get(key), dict):
            merge_config(config[key], value)
        else:
            config[key] = copy.deepcopy(value)
    return config


def load_config(path, device_profile=None):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if "base_config" in config:
        with (Path(path).parent / config.pop("base_config")).open(encoding="utf-8") as file:
            config = merge_config(yaml.safe_load(file), config)
    profile = device_profile or os.environ.get("DEVICE_PROFILE") or config["device_profile"]
    profile = PROFILE_ALIASES.get(profile, profile)
    profiles = config.pop("hardware_profiles")
    if profile not in profiles:
        raise ValueError(f"Unknown device profile {profile!r}; choose from {list(profiles)}")
    merge_config(config, profiles[profile])
    config["device_profile"] = profile
    data = config["data"]
    if not isinstance(data["max_length"], int) or data["max_length"] <= 0:
        raise ValueError("data.max_length must be a positive integer")
    fsdp = config["training"].get("fsdp_config", {})
    if config["training"].get("fsdp") and (
        fsdp.get("version") != 2 or not isinstance(fsdp.get("reshard_after_forward"), bool)
    ):
        raise ValueError("FSDP requires version: 2 and boolean reshard_after_forward")
    return config


def build_training_args(config, layer_classes, config_class, transformers_version):
    """YAML training keys map directly to DPOConfig, without repeated assignments."""
    options = copy.deepcopy(config["training"])
    options.pop("resume_from_checkpoint", None)
    options.update(config["dpo"])
    options["max_length"] = config["data"]["max_length"]
    options["remove_unused_columns"] = False
    options["dataloader_persistent_workers"] = options.get("dataloader_num_workers", 0) > 0
    if options.get("fsdp"):
        options["fsdp_config"]["transformer_layer_cls_to_wrap"] = layer_classes
    parameters = inspect.signature(config_class.__init__).parameters
    # Transformers 4.x uses warmup_ratio for fractional warmup; current 5.x
    # accepts the fraction directly in warmup_steps and removed warmup_ratio.
    warmup = options.get("warmup_steps", 0)
    if "warmup_ratio" in parameters and 0 < warmup < 1:
        options["warmup_ratio"] = options.pop("warmup_steps")
    # Current Transformers uses a boolean FSDP switch and fsdp_config.version.
    # Older releases require the string switch and fsdp_version key instead.
    version = tuple(int(part) for part in transformers_version.split(".")[:2])
    if version < (5, 20):
        options["fsdp"] = "full_shard auto_wrap" if options.get("fsdp") else ""
        if options.get("fsdp"):
            fsdp = options["fsdp_config"]
            fsdp["fsdp_version"] = fsdp.pop("version")
            fsdp.pop("auto_wrap_policy", None)
    unsupported = sorted(set(options) - set(parameters))
    if unsupported:
        raise ValueError(
            f"Installed DPOConfig does not support: {', '.join(unsupported)}. "
            "Update these YAML keys to match your TRL/Transformers version."
        )
    return config_class(**options)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the resolved DPO hardware profile")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--device_profile", default=None)
    parser.add_argument("--num-processes", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.device_profile)
    print(config["num_processes"] if args.num_processes else yaml.safe_dump(config, sort_keys=False))
