import argparse
import base64
import copy
import glob
import io
import json
import math
import os
import random
import time
from types import SimpleNamespace

# This trainer is PyTorch-only. Avoid importing an incompatible optional
# TensorFlow installation through Transformers' image-processing imports.
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

try:
    from transformers import AutoModelImageTextToText
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelImageTextToText


def load_image_from_data_uri(image_url):
    """Load an OpenAI-style image_url block containing a base64 data URI."""
    data_uri = image_url["url"]
    header, encoded = data_uri.split(",", 1)
    if not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Expected data:image/...;base64,...")
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        return image.convert("RGB")


class SFTDataset(Dataset):
    """One sample per assistant turn, with only its final response supervised."""

    PHASES = (
        ("messages", "response"),
        ("planning_messages", "planning_response"),
        ("action_messages", "action_response"),
        ("reflect_messages", "reflect_response"),
    )

    def __init__(self, data_path, processor, max_length=4096, system_prompt=None, data_type="",
                 max_images=3, debug=False):
        self.processor = processor
        self.tok = processor.tokenizer
        self.max_length = int(max_length)
        self.max_images = max(1, int(max_images))
        self.system_prompt = system_prompt
        self.data_type = data_type
        self._cached_path = None
        self._cached_steps = None
        self.skipped_incomplete_phases = 0
        if self.data_type == "ghostei":
            pattern = os.path.join(data_path, "**", "*.json")
            files = [
                path for path in glob.glob(pattern, recursive=True)
                if os.path.splitext(os.path.basename(path))[0].isdigit()
            ]
            if not files:
                raise FileNotFoundError(f"No trajectory JSON files matched: {pattern}")
            safe_files = []
            for path in files:
                with open(path, "r", encoding="utf-8") as file:
                    steps = json.load(file)
                if not any(step.get("is_attack", False) for step in steps):
                    safe_files.append(path)
        elif self.data_type == "mobilesafetybench":
            files = []
            task_categories = ["memo_app_download", "memo_completing",
                            "social_media_download", "social_media_posting_direct",
                            "text_message_forwarding", "text_message_sending"]
            for task_category in task_categories:
                task_pattern = os.path.join(data_path, "*", task_category, "*")
                task_dirs = sorted(
                    task_dir
                    for task_dir in glob.glob(task_pattern)
                    if os.path.isdir(task_dir) and
                    not os.path.basename(task_dir).endswith("_no_attack")
                )
                for task_dir in task_dirs:
                    pattern = os.path.join(
                        task_dir,
                        "*",
                        "log_*_defense_-1.json",
                    )
                    files.extend(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No trajectory JSON files matched: {pattern}")
            safe_files = []
            for path in files:
                with open(path, "r", encoding="utf-8") as file:
                    steps = json.load(file)
                if not any(step.get("is_attack", False) for step in steps):
                    safe_files.append(path)
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")


        self.samples = []
        for path in safe_files:
            with open(path, "r", encoding="utf-8") as file:
                steps = json.load(file)
            for step_index, step in enumerate(steps):
                for messages_key, response_key in self.PHASES:
                    messages = step.get(messages_key)
                    response = step.get(response_key)
                    # A terminal planning phase may mark the task Finished,
                    # leaving the following action/reflect phase initialized
                    # but empty. Do not let such records reach label building.
                    if not isinstance(messages, list) or not messages:
                        self.skipped_incomplete_phases += 1
                        continue
                    if not isinstance(response, str) or not response.strip():
                        self.skipped_incomplete_phases += 1
                        continue
                    # Keep only a lightweight index. Materializing every
                    # base64 screenshot in both distributed ranks can consume
                    # tens of GB of host RAM before training starts.
                    self.samples.append(
                        (path, step_index, messages_key, response_key)
                    )

        if debug:
            self.samples = self.samples[:5]
        if not self.samples:
            raise ValueError("No safe assistant-turn SFT samples were found")

    def __len__(self):
        return len(self.samples)

    def _materialize_sample(self, index):
        path, step_index, messages_key, response_key = self.samples[index]
        if path != self._cached_path:
            with open(path, "r", encoding="utf-8") as file:
                self._cached_steps = json.load(file)
            self._cached_path = path
        step = self._cached_steps[step_index]
        messages_value = step.get(messages_key)
        response_value = step.get(response_key)
        if not isinstance(messages_value, list) or not messages_value:
            raise RuntimeError(
                f"Indexed SFT phase became empty: {path}, step={step_index}, "
                f"field={messages_key}"
            )
        if not isinstance(response_value, str) or not response_value.strip():
            raise RuntimeError(
                f"Indexed SFT target became empty: {path}, step={step_index}, "
                f"field={response_key}"
            )
        messages = copy.deepcopy(messages_value)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_value}],
        })
        return messages

    def _limit_images(self, messages):
        locations = []
        for message_index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block_index, block in enumerate(content):
                if block.get("type") in ("image", "image_url"):
                    locations.append((message_index, block_index))

        for message_index, block_index in locations[:-self.max_images]:
            messages[message_index]["content"][block_index] = {
                "type": "text", "text": "[Earlier screenshot omitted]"
            }

    def _prepare_messages(self, messages):
        images = []
        clean_messages = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, list):
                clean_messages.append({"role": message["role"], "content": content})
                continue

            clean_content = []
            for block in content:
                block_type = block.get("type")
                if block_type == "image_url":
                    images.append(load_image_from_data_uri(block["image_url"]))
                    clean_content.append({"type": "image"})
                elif block_type == "image":
                    if "image" not in block:
                        raise ValueError("An image block is missing its image payload")
                    images.append(block["image"])
                    clean_content.append({"type": "image"})
                else:
                    clean_content.append(block)
            clean_messages.append({"role": message["role"], "content": clean_content})
        return clean_messages, images

    def _build_target_only_labels(self, clean_messages, images, input_ids, full_text):
        if not clean_messages or clean_messages[-1].get("role") != "assistant":
            raise ValueError("Every SFT sample must end with an assistant target")

        # The history rendered with a generation prompt ends exactly where the
        # final target begins, including multimodal visual-token expansion.
        prefix_text = self.processor.apply_chat_template(
            clean_messages[:-1], tokenize=False, add_generation_prompt=True
        )
        if not full_text.startswith(prefix_text):
            raise RuntimeError("The chat template cannot isolate the final assistant target")
        prefix_output = self.processor(
            text=[prefix_text], images=images if images else None,
            return_tensors="pt", padding=False, truncation=False,
        )
        prefix_ids = prefix_output["input_ids"][0].tolist()
        # Tokenizing the prompt by itself may append an EOS token, while the
        # same text inside the full sample is followed by the assistant target.
        # A tokenizer may also merge one token across that text boundary. Use
        # the common token prefix as the loss boundary instead of requiring the
        # independently tokenized sequences to be exactly prefix-identical.
        prompt_length = 0
        for prefix_id, full_id in zip(prefix_ids, input_ids):
            if prefix_id != full_id:
                break
            prompt_length += 1

        boundary_difference = len(prefix_ids) - prompt_length
        max_boundary_difference = max(8, len(prefix_ids) // 100)
        if prompt_length == 0 or boundary_difference > max_boundary_difference:
            raise RuntimeError(
                "Prompt/full tokenization diverges before the assistant boundary: "
                f"matched={prompt_length}, prompt_tokens={len(prefix_ids)}, "
                f"boundary_difference={boundary_difference}. Check the model chat template."
            )

        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError("SFT sample contains no supervised target tokens")
        return labels

    def _encode_messages(self, messages):
        if self.system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        clean_messages, images = self._prepare_messages(messages)
        text = self.processor.apply_chat_template(
            clean_messages, tokenize=False, add_generation_prompt=False
        )
        output = self.processor(
            text=[text], images=images if images else None,
            return_tensors="pt", padding=False, truncation=False,
        )
        input_ids = output["input_ids"][0].tolist()
        labels = self._build_target_only_labels(clean_messages, images, input_ids, text)
        item = {"input_ids": input_ids, "labels": labels}
        if output.get("pixel_values") is not None:
            item["pixel_values"] = output["pixel_values"]
        if output.get("image_grid_thw") is not None:
            item["image_grid_thw"] = output["image_grid_thw"]
        return item

    def _encode_one(self, sample):
        messages = copy.deepcopy(sample)
        self._limit_images(messages)

        # Crop complete old messages only. Never slice multimodal tokens or target.
        while True:
            item = self._encode_messages(messages)
            if len(item["input_ids"]) <= self.max_length:
                if not any(label != -100 for label in item["labels"]):
                    raise ValueError("SFT sample contains no supervised target tokens")
                return item
            removable = next(
                (i for i, message in enumerate(messages[:-1])
                 if message.get("role") != "system"), None
            )
            if removable is None:
                raise ValueError(
                    f"System prompt plus target exceeds max_length={self.max_length}"
                )
            # In this dataset each phase prompt is one user message whose text
            # already contains the compressed history. Removing it would turn
            # the sample into an unconditional target, so fail loudly instead.
            if len(messages[:-1]) == 1:
                raise ValueError(
                    f"Single-message multimodal prompt exceeds max_length={self.max_length}; "
                    "increase max_length, reduce image resolution, or summarize the embedded "
                    "history before training"
                )
            del messages[removable]

    def __getitem__(self, index):
        return self._encode_one(self._materialize_sample(index))


class SFTCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tok.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        pixel_values = [f["pixel_values"] for f in features if f.get("pixel_values") is not None]
        image_grids = [f["image_grid_thw"] for f in features if f.get("image_grid_thw") is not None]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grids:
            batch["image_grid_thw"] = torch.cat(image_grids, dim=0)
        return batch


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_ns(item) for item in value]
    return value


def _to_dict(value):
    if isinstance(value, SimpleNamespace):
        return {key: _to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    return value


def _get_module(model, path):
    module = model
    for name in path.split("."):
        module = getattr(module, name, None)
        if module is None:
            return None
    return module


def get_fsdp_layer_classes(model):
    """Find text and vision transformer block classes for per-block FSDP wrapping."""
    decoder_paths = (
        "model.layers",
        "model.language_model.layers",
        "model.text_model.layers",
    )
    vision_paths = (
        "visual.blocks",
        "model.visual.blocks",
        "vision_model.encoder.layers",
        "model.vision_model.encoder.layers",
    )
    class_names = []
    decoder_found = False
    for path in decoder_paths + vision_paths:
        blocks = _get_module(model, path)
        if blocks is None or len(blocks) == 0:
            continue
        if path in decoder_paths:
            decoder_found = True
        class_name = type(blocks[0]).__name__
        if class_name not in class_names:
            class_names.append(class_name)
    if not decoder_found:
        raise ValueError(
            "FSDP auto-wrap could not find the model's decoder layers. "
            "Add its transformer block path to get_fsdp_layer_classes()."
        )
    return class_names


class JsonlMetricsCallback(TrainerCallback):
    """Persist rank-0 Trainer logs in a plotting-friendly JSONL file."""

    def __init__(self, output_dir, append=False):
        self.path = os.path.join(output_dir, "metrics.jsonl")
        self.append = append
        self.started_at = None
        self.step_started_at = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.perf_counter()
        if state.is_world_process_zero and not self.append:
            with open(self.path, "w", encoding="utf-8"):
                pass

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started_at = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started_at is not None:
            self.step_times.append(time.perf_counter() - self.step_started_at)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        row = {"step": state.global_step, **logs}
        if self.started_at is not None:
            row["wall_time_s"] = time.perf_counter() - self.started_at
        if self.step_times:
            step_time = sum(self.step_times) / len(self.step_times)
            row["step_time_s"] = step_time
            effective_batch = (
                args.per_device_train_batch_size
                * args.world_size
                * args.gradient_accumulation_steps
            )
            row["samples_per_second"] = effective_batch / step_time
            self.step_times.clear()
        if torch.cuda.is_available():
            row["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_vllm_checkpoint(
    trainer,
    processor,
    output_dir,
    fsdp_enabled,
    lora_enabled=False,
    merge_on_save=False,
    merged_output_dir=None,
):
    """Save the SFT adapter and optionally a merged checkpoint for deployment."""
    trainer.accelerator.wait_for_everyone()
    trainer.model.config.use_cache = True

    if lora_enabled:
        # This performs the FSDP state-dict collective before PEFT keeps only
        # the adapter tensors.
        trainer.save_model(output_dir)
        if trainer.is_world_process_zero():
            processor.save_pretrained(output_dir)
            adapter_config = os.path.join(output_dir, "adapter_config.json")
            if not os.path.isfile(adapter_config):
                raise RuntimeError(f"Missing saved LoRA config: {adapter_config}")
            if merge_on_save:
                print(
                    "merge_on_save is ignored for sharded training; load the "
                    "base model and saved adapter together for inference.",
                    flush=True,
                )
            print(f"saved SFT LoRA adapter to {output_dir}", flush=True)
        trainer.accelerator.wait_for_everyone()
        return

    if fsdp_enabled:
        # FSDP2 parameters are distributed across ranks. Every rank must enter
        # this collective; cpu_offload makes only rank 0 receive the complete
        # state dict and avoids materializing it in GPU memory.
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state_dict = get_model_state_dict(
            trainer.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        state_dict = trainer.accelerator.get_state_dict(trainer.model)

    if trainer.is_world_process_zero():
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        processor.save_pretrained(output_dir)
        has_config = os.path.isfile(os.path.join(output_dir, "config.json"))
        has_weights = any(
            name.endswith(".safetensors") for name in os.listdir(output_dir)
        )
        if not has_config or not has_weights:
            raise RuntimeError(
                "The exported checkpoint is incomplete: expected config.json "
                "and at least one .safetensors weight file"
            )
        print(f"saved vLLM-compatible checkpoint: {output_dir}", flush=True)

    trainer.accelerator.wait_for_everyone()


def apply_hardware_profile(config, profile_name):
    profiles = config.get("hardware_profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown device profile {profile_name!r}; choose one of "
            f"{sorted(profiles)}"
        )

    def merge(target, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                if value:
                    merge(target[key], value)
                else:
                    target[key] = {}
            else:
                target[key] = copy.deepcopy(value)

    merge(config, profiles[profile_name])
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal trajectory SFT")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--lora", action="store_true", default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--device_profile",
        default=os.environ.get("DEVICE_PROFILE", "dual_4090d"),
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = apply_hardware_profile(config, args.device_profile)
    print(f"Using device profile: {args.device_profile}", flush=True)
    overrides = {
        ("model", "path"): args.model_path,
        ("data", "train_jsonl"): args.train_jsonl,
        ("data", "max_length"): args.max_length,
        ("data", "max_images"): args.max_images,
        ("training", "output_dir"): args.output_dir,
        ("training", "learning_rate"): args.learning_rate,
        ("training", "num_train_epochs"): args.num_train_epochs,
        ("training", "per_device_train_batch_size"): args.per_device_train_batch_size,
        ("training", "gradient_accumulation_steps"): args.gradient_accumulation_steps,
        ("training", "resume_from_checkpoint"): args.resume_from_checkpoint,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.lora is True:
        config["lora"]["enable"] = True
    if args.debug is True:
        config["runtime"]["debug"] = True
    return _to_ns(config)


def main():
    config = parse_args()
    set_seed(config.training.seed)
    os.makedirs(config.training.output_dir, exist_ok=True)
    is_main_process = int(os.environ.get("RANK", "0")) == 0
    training = config.training
    fsdp_config = _to_dict(training.fsdp_config)
    if training.fsdp:
        fsdp_version = int(fsdp_config.get("fsdp_version", 2))
        if fsdp_version != 2:
            raise ValueError(f"This trainer requires FSDP2, got fsdp_version={fsdp_version}")
        if not isinstance(fsdp_config.get("reshard_after_forward"), bool):
            raise TypeError("FSDP2 reshard_after_forward must be true or false")
        # Needed by Transformers 4.x; Transformers 5.x also receives the
        # explicit fsdp_version below through fsdp_config.
        os.environ["FSDP_VERSION"] = str(fsdp_version)

    processor = AutoProcessor.from_pretrained(
        config.model.path,
        trust_remote_code=config.model.trust_remote_code,
        max_pixels=config.data.max_pixels,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelImageTextToText.from_pretrained(
        config.model.path,
        torch_dtype=DTYPE_MAP[config.model.torch_dtype],
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation=getattr(config.model, "attn_implementation", "sdpa"),
        low_cpu_mem_usage=True,
        device_map=None,
    )
    fsdp_layer_classes = get_fsdp_layer_classes(model) if training.fsdp else []
    # KV cache is for autoregressive inference and wastes memory in training.
    model.config.use_cache = False

    if config.lora.enable:
        from peft import LoraConfig, get_peft_model
        visual = getattr(model, "visual", None)
        if visual is not None:
            for parameter in visual.parameters():
                parameter.requires_grad = False
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            task_type=config.lora.task_type,
        )
        model = get_peft_model(model, lora_config)
        visual_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and ".visual." in name
        ]
        if visual_trainable:
            raise RuntimeError(
                "Language-only LoRA unexpectedly matched visual modules: "
                + ", ".join(visual_trainable[:5])
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if is_main_process:
            model.print_trainable_parameters()

    dataset = SFTDataset(
        config.data.train_jsonl,
        processor,
        max_length=config.data.max_length,
        system_prompt=config.data.system_prompt,
        data_type=config.data.data_type,
        max_images=getattr(config.data, "max_images", 3),
        debug=config.runtime.debug,
    )
    if is_main_process:
        print(
            f"loaded {len(dataset)} assistant-turn SFT samples; skipped "
            f"{dataset.skipped_incomplete_phases} incomplete phases",
            flush=True,
        )
    collator = SFTCollator(tokenizer)
    if training.fsdp:
        fsdp_config["transformer_layer_cls_to_wrap"] = fsdp_layer_classes
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_rank_samples = math.ceil(len(dataset) / world_size)
    micro_batches_per_epoch = math.ceil(
        per_rank_samples / training.per_device_train_batch_size
    )
    update_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / training.gradient_accumulation_steps
    )
    total_steps = math.ceil(update_steps_per_epoch * training.num_train_epochs)
    warmup_steps = math.ceil(total_steps * training.warmup_ratio)
    if is_main_process:
        effective_batch = (
            training.per_device_train_batch_size
            * world_size
            * training.gradient_accumulation_steps
        )
        print(
            f"world_size={world_size}, effective_batch_size={effective_batch}, "
            f"fsdp={training.fsdp or 'disabled'}, wrap_classes="
            f"{fsdp_config.get('transformer_layer_cls_to_wrap', [])}, "
            f"total_steps={total_steps}, warmup_steps={warmup_steps}",
            flush=True,
        )
    training_args = TrainingArguments(
        output_dir=training.output_dir,
        num_train_epochs=training.num_train_epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=training.lr_scheduler_type,
        logging_strategy=training.logging_strategy,
        logging_steps=training.logging_steps,
        logging_first_step=training.logging_first_step,
        save_strategy=training.save_strategy,
        save_total_limit=training.save_total_limit,
        bf16=training.bf16,
        fp16=training.fp16,
        tf32=getattr(training, "tf32", True),
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs=_to_dict(training.gradient_checkpointing_kwargs),
        report_to=training.report_to,
        optim=getattr(training, "optim", "adamw_torch"),
        max_grad_norm=getattr(training, "max_grad_norm", 1.0),
        fsdp=training.fsdp,
        fsdp_config=fsdp_config,
        remove_unused_columns=training.remove_unused_columns,
        dataloader_pin_memory=training.dataloader_pin_memory,
        dataloader_num_workers=config.runtime.num_workers,
        dataloader_persistent_workers=config.runtime.num_workers > 0,
        seed=training.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[
            JsonlMetricsCallback(
                training.output_dir,
                append=training.resume_from_checkpoint is not None,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=training.resume_from_checkpoint)
    trainer.save_state()
    save_vllm_checkpoint(
        trainer,
        processor,
        training.output_dir,
        fsdp_enabled=bool(training.fsdp),
        lora_enabled=bool(config.lora.enable),
        merge_on_save=bool(getattr(config.lora, "merge_on_save", False)),
        merged_output_dir=getattr(config.lora, "merged_output_dir", None),
    )


if __name__ == "__main__":
    main()
