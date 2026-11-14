import argparse
import base64
import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

os.environ.setdefault("USE_TF", "0")

import torch
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList

try:
    from transformers import AutoModelImageTextToText
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelImageTextToText


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: ChatMessage
    finish_reason: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI-style frequency penalty over tokens generated after the prompt."""

    def __init__(self, prompt_length: int, penalty: float):
        self.prompt_length = int(prompt_length)
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generated_ids = input_ids[:, self.prompt_length:]
        if generated_ids.numel() == 0 or self.penalty == 0:
            return scores
        counts = torch.zeros_like(scores)
        counts.scatter_add_(1, generated_ids, torch.ones_like(generated_ids, dtype=scores.dtype))
        return scores - self.penalty * counts


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str):
        raise TypeError("Image payload must be a PIL image, path, URL, or data URI")

    if value.startswith("data:image/"):
        header, encoded = value.split(",", 1)
        if ";base64" not in header:
            raise ValueError("Only base64 image data URIs are supported")
        data = base64.b64decode(encoded)
    elif value.startswith(("http://", "https://")):
        with urlopen(value, timeout=30) as response:
            data = response.read()
    else:
        path = value[7:] if value.startswith("file://") else value
        with open(path, "rb") as file:
            data = file.read()

    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _prepare_messages(messages: list[dict]) -> tuple[list[dict], list[Image.Image]]:
    normalized = []
    images = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            raise ValueError("Every message requires role and content")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise TypeError("Message content must be a string or a list of content blocks")

        clean_content = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                clean_content.append({"type": "text", "text": block["text"]})
            elif block_type == "image_url":
                images.append(_load_image(block["image_url"]))
                clean_content.append({"type": "image"})
            elif block_type == "image":
                images.append(_load_image(block.get("image")))
                clean_content.append({"type": "image"})
            else:
                raise ValueError(f"Unsupported content block type: {block_type!r}")
        normalized.append({"role": role, "content": clean_content})
    return normalized, images


class _Completions:
    def __init__(self, client: "OpenAICompatibleVLMClient"):
        self._client = client

    def create(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> ChatCompletion:
        if stream:
            raise NotImplementedError("Streaming responses are not implemented")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        clean_messages, images = _prepare_messages(messages)
        prompt = self._client.processor.apply_chat_template(
            clean_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._client.processor(
            text=[prompt],
            images=images if images else None,
            return_tensors="pt",
            padding=False,
        )
        prompt_length = inputs["input_ids"].shape[1]
        inputs = {key: value.to(self._client.input_device) for key, value in inputs.items()}

        generation_kwargs = dict(extra_body or {})
        generation_kwargs["max_new_tokens"] = int(max_tokens)
        generation_kwargs["do_sample"] = temperature > 0
        if temperature > 0:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)

        processors = generation_kwargs.pop("logits_processor", None)
        processors = LogitsProcessorList(processors or [])
        if frequency_penalty:
            processors.append(FrequencyPenaltyLogitsProcessor(prompt_length, frequency_penalty))
        if processors:
            generation_kwargs["logits_processor"] = processors

        with torch.inference_mode():
            output = self._client.model.generate(**inputs, **generation_kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        generated_ids = sequences[:, prompt_length:]
        content = self._client.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        completion_tokens = generated_ids.shape[1]
        eos_token_id = self._client.processor.tokenizer.eos_token_id
        stopped = completion_tokens < max_tokens
        if eos_token_id is not None and completion_tokens:
            stopped = stopped or generated_ids[0, -1].item() == eos_token_id

        return ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=model or self._client.model_name,
            choices=[Choice(0, ChatMessage("assistant", content), "stop" if stopped else "length")],
            usage=Usage(
                prompt_tokens=prompt_length,
                completion_tokens=completion_tokens,
                total_tokens=prompt_length + completion_tokens,
            ),
        )


class _Chat:
    def __init__(self, client: "OpenAICompatibleVLMClient"):
        self.completions = _Completions(client)


class OpenAICompatibleVLMClient:
    """Local GUI-VLM checkpoint with an OpenAI-compatible chat interface."""

    def __init__(
        self,
        checkpoint_path: str,
        adapter_path: str | None = None,
        processor_path: str | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str | dict = "auto",
        trust_remote_code: bool = True,
        max_pixels: int = 802816,
    ):
        if torch_dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        processor_path = processor_path or adapter_path or checkpoint_path
        self.model_name = adapter_path or checkpoint_path
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=trust_remote_code,
            max_pixels=max_pixels,
        )
        self.model = AutoModelImageTextToText.from_pretrained(
            checkpoint_path,
            torch_dtype=DTYPE_MAP[torch_dtype],
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.model.config.use_cache = True
        self.input_device = self.model.get_input_embeddings().weight.device
        self.chat = _Chat(self)


def main():
    parser = argparse.ArgumentParser(description="Run inference with an SFT GUI-VLM checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Base model path")
    parser.add_argument("--adapter", default=None, help="Optional SFT LoRA adapter path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    content = []
    if args.image:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.prompt})

    client = OpenAICompatibleVLMClient(args.checkpoint, adapter_path=args.adapter)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
        model=args.checkpoint,
        max_tokens=args.max_tokens,
        stream=False,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
