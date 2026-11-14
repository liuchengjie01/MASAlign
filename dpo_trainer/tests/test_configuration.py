"""GPU switches and TRL options, without loading torch or a model."""

from pathlib import Path
import inspect
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dpo_trainer.configuration import build_training_args, load_config


CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


class FakeDPOConfig:
    # A strict constructor contract: removed length fields are absent, and
    # unsupported arguments fail before the constructor gets called.
    def __init__(self, **kwargs):
        self.options = kwargs


SUPPORTED_FIELDS = (
    "output_dir num_train_epochs per_device_train_batch_size gradient_accumulation_steps "
    "learning_rate warmup_steps lr_scheduler_type logging_steps save_strategy save_total_limit "
    "bf16 tf32 gradient_checkpointing gradient_checkpointing_kwargs optim max_grad_norm report_to "
    "fsdp fsdp_config remove_unused_columns dataloader_pin_memory seed dataloader_num_workers "
    "beta max_length truncation_mode dataloader_persistent_workers"
).split()


def constructor_signature(fields):
    return inspect.Signature([
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in fields),
    ])


FakeDPOConfig.__init__.__signature__ = constructor_signature(SUPPORTED_FIELDS)


class ConfigurationTests(unittest.TestCase):
    def test_dual_gpu_uses_fsdp_and_single_gpu_disables_it(self):
        dual = load_config(CONFIG, "dual_4090d")
        single = load_config(CONFIG, "pro6000_98g")
        self.assertEqual(dual["num_processes"], 2)
        self.assertEqual(single["num_processes"], 1)
        self.assertEqual(dual["data"]["max_length"], 4096)
        self.assertEqual(single["data"]["max_length"], 8192)
        self.assertTrue(dual["training"]["fsdp"])
        self.assertEqual(single["training"]["fsdp_config"], {})
        self.assertFalse(single["training"]["fsdp"])
        self.assertEqual(load_config(CONFIG, "rtx6000_96g"), single)

    def test_training_args_keep_yaml_keys_and_add_wrap_classes(self):
        for profile in ("dual_4090d", "pro6000_98g"):
            config = load_config(CONFIG, profile)
            options = build_training_args(config, ["DecoderBlock"], FakeDPOConfig, "5.22.2").options
            self.assertEqual(options["beta"], config["dpo"]["beta"])
            self.assertEqual(options["max_length"], config["data"]["max_length"])
            self.assertEqual(options["warmup_steps"], 0.1)
            self.assertNotIn("max_prompt_length", options)
            self.assertNotIn("max_completion_length", options)
            self.assertNotIn("resume_from_checkpoint", options)
            if profile == "dual_4090d":
                self.assertEqual(options["fsdp_config"]["transformer_layer_cls_to_wrap"], ["DecoderBlock"])
                self.assertTrue(options["fsdp"])
                self.assertEqual(options["fsdp_config"]["version"], 2)
                self.assertNotIn("transformer_layer_cls_to_wrap", config["training"]["fsdp_config"])

    def test_older_transformers_warmup_and_fsdp_are_translated(self):
        config = load_config(CONFIG, "dual_4090d")
        legacy = constructor_signature([*SUPPORTED_FIELDS, "warmup_ratio"])
        with patch.object(FakeDPOConfig.__init__, "__signature__", legacy):
            options = build_training_args(config, ["Block"], FakeDPOConfig, "4.57.6").options
        self.assertEqual(options["warmup_ratio"], 0.1)
        self.assertNotIn("warmup_steps", options)
        self.assertEqual(options["fsdp"], "full_shard auto_wrap")
        self.assertEqual(options["fsdp_config"]["fsdp_version"], 2)
        self.assertNotIn("version", options["fsdp_config"])
        config["training"]["warmup_steps"] = 100
        options = build_training_args(config, ["Block"], FakeDPOConfig, "5.22.2").options
        self.assertEqual(options["warmup_steps"], 100)

    def test_unknown_parameters_are_reported_instead_of_silently_dropped(self):
        config = load_config(CONFIG, "pro6000_98g")
        config["dpo"]["removed_parameter"] = True
        with self.assertRaisesRegex(ValueError, "does not support: removed_parameter"):
            build_training_args(config, [], FakeDPOConfig, "5.22.2")

    def test_long_context_only_overrides_single_gpu_limits(self):
        path = CONFIG.with_name("config_long_context.yaml")
        self.assertEqual(load_config(path, "pro6000_98g")["data"]["max_length"], 32768)
        self.assertEqual(load_config(path, "dual_4090d")["data"]["max_length"], 4096)
        with self.assertRaisesRegex(ValueError, "Unknown device profile"):
            load_config(CONFIG, "unknown")


if __name__ == "__main__":
    unittest.main()
