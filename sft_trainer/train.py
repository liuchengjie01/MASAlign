import argparse
import base64
import copy
import glob
import io
import json
import os
import random
from types import SimpleNamespace

# This trainer is PyTorch-only. Avoid importing an incompatible optional
# TensorFlow installation through Transformers' image-processing imports.
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments


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
        ("planning_messages", "planning_response"),
        ("action_messages", "action_response"),
        ("reflect_messages", "reflect_response"),
    )

    def __init__(self, data_path, processor, max_length=4096, system_prompt=None,
                 max_images=3, debug=False):
        self.processor = processor
        self.tok = processor.tokenizer
        self.max_length = int(max_length)
        self.max_images = max(1, int(max_images))
        self.system_prompt = system_prompt

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

        self.samples = []
        for path in safe_files:
            with open(path, "r", encoding="utf-8") as file:
                steps = json.load(file)
            for step in steps:
                for messages_key, response_key in self.PHASES:
                    if messages_key not in step or response_key not in step:
                        continue
                    # *_messages is the prompt actually sent to the target model.
                    # It already contains cumulative history and must not be
                    # appended to the previous phase again.
                    messages = copy.deepcopy(step[messages_key])
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": step[response_key]}],
                    })
                    self.samples.append(messages)

        if debug:
            self.samples = self.samples[:5]

    def __len__(self):
        return len(self.samples)

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
        if input_ids[:len(prefix_ids)] != prefix_ids:
            raise RuntimeError("The tokenized prompt is not a prefix of the full sample")
        return [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]

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
        return self._encode_one(self.samples[index])


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
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
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

    processor = AutoProcessor.from_pretrained(
        config.model.path, trust_remote_code=config.model.trust_remote_code
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.path,
        torch_dtype=DTYPE_MAP[config.model.torch_dtype],
        trust_remote_code=config.model.trust_remote_code,
        device_map=None,
    )
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if config.lora.enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=list(config.lora.target_modules),
            task_type=config.lora.task_type,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = SFTDataset(
        config.data.train_jsonl,
        processor,
        max_length=config.data.max_length,
        system_prompt=config.data.system_prompt,
        max_images=getattr(config.data, "max_images", 3),
        debug=config.runtime.debug,
    )
    print(f"loaded {len(dataset)} assistant-turn SFT samples")
    collator = SFTCollator(tokenizer)
    training = config.training
    training_args = TrainingArguments(
        output_dir=training.output_dir,
        num_train_epochs=training.num_train_epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        warmup_ratio=training.warmup_ratio,
        lr_scheduler_type=training.lr_scheduler_type,
        logging_steps=training.logging_steps,
        save_strategy=training.save_strategy,
        save_total_limit=training.save_total_limit,
        bf16=training.bf16,
        fp16=training.fp16,
        gradient_checkpointing=training.gradient_checkpointing,
        report_to=training.report_to,
        remove_unused_columns=training.remove_unused_columns,
        dataloader_pin_memory=training.dataloader_pin_memory,
        dataloader_num_workers=config.runtime.num_workers,
        seed=training.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=training.resume_from_checkpoint)
    trainer.save_model(training.output_dir)
    tokenizer.save_pretrained(training.output_dir)
    print("saved:", training.output_dir)


if __name__ == "__main__":
    main()
