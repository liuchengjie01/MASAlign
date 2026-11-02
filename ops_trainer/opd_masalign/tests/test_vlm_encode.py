"""Tests for the multimodal (VLM) encode path added to opd_trainer.py.

Uses tiny fake processor/tokenizer stubs instead of downloading a real VLM,
so these run fast and offline. The real end-to-end path (a real screenshot
from data/luis/test_log.zip through HuggingFaceTB/SmolVLM-256M-Instruct) was
verified separately as a manual smoke test -- see docs/DATA_FORMAT.md.

Run with: python -m pytest opd_masalign/tests/test_vlm_encode.py -v
"""
import base64
import io

import torch
from PIL import Image

from opd_masalign.opd_trainer import (
    _batch_vision_kwargs,
    _cap_longer_side,
    _decode_image_block,
    _encode_context_and_action_vlm,
    _extract_images_and_normalize,
)


def _tiny_png_data_uri(size=(20, 40)) -> str:
    img = Image.new("RGB", size, color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def test_decode_image_block_roundtrips_a_real_image():
    uri = _tiny_png_data_uri(size=(20, 40))
    block = {"type": "image_url", "image_url": {"url": uri}}
    img = _decode_image_block(block)
    assert img.size == (20, 40)
    assert img.mode == "RGB"


def test_decode_image_block_rejects_non_data_urls():
    block = {"type": "image_url", "image_url": {"url": "https://example.com/screenshot.jpg"}}
    try:
        _decode_image_block(block)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_cap_longer_side_preserves_aspect_ratio():
    img = Image.new("RGB", (1080, 2400))
    capped = _cap_longer_side(img, 512)
    assert max(capped.size) == 512
    assert abs(capped.size[0] / capped.size[1] - 1080 / 2400) < 1e-3


def test_extract_images_and_normalize_splits_multimodal_content():
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _tiny_png_data_uri()}},
            {"type": "text", "text": "what do you see?"},
        ]},
    ]
    normalized, images = _extract_images_and_normalize(messages)
    assert len(images) == 1
    assert normalized[0] == messages[0]  # plain-string content passes through unchanged
    assert normalized[1]["content"] == [{"type": "image"}, {"type": "text", "text": "what do you see?"}]


def test_extract_images_and_normalize_handles_no_images():
    messages = [{"role": "user", "content": "just text, no image"}]
    normalized, images = _extract_images_and_normalize(messages)
    assert images == []
    assert normalized == messages


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


class _FakeProcessor:
    """Pretends the prompt always encodes to a fixed 20 tokens, regardless of
    content, so the truncation guard can be tested deterministically without
    a real model/tokenizer. Mimics Idefics3/SmolVLM's convention: bare
    pixel_values shaped (1, num_images, C, H, W) -- a real per-call batch dim.
    """

    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, text, images=None, return_tensors="pt"):
        out = {"input_ids": torch.tensor([[1] * 20])}
        if images:
            out["pixel_values"] = torch.zeros(1, len(images), 3, 4, 4)
        return out


class _FakeGlm4vProcessor(_FakeProcessor):
    """Mimics GLM-4V/Qwen2-VL's convention instead: pixel_values already
    flattened across images with NO batch dim, plus image_grid_thw.
    """

    def __call__(self, text, images=None, return_tensors="pt"):
        out = {"input_ids": torch.tensor([[1] * 20])}
        if images:
            n_patches = 6 * len(images)
            out["pixel_values"] = torch.zeros(n_patches, 8)
            out["image_grid_thw"] = torch.tensor([[1, 2, 3]] * len(images))
        return out


def test_encode_context_and_action_vlm_returns_pixel_values_for_one_image():
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _tiny_png_data_uri()}},
        {"type": "text", "text": "hi"},
    ]}]
    ids, mask, vision_kwargs = _encode_context_and_action_vlm(_FakeProcessor(), messages, "do_something", max_len=100)
    assert set(vision_kwargs.keys()) == {"pixel_values"}
    assert vision_kwargs["pixel_values"].shape == (1, 1, 3, 4, 4)
    assert sum(mask) == len("do_something")  # action span length, per _FakeTokenizer
    assert len(ids) == 20 + len("do_something")


def test_encode_context_and_action_vlm_passes_through_extra_vision_kwargs():
    # GLM-4V-style processors return image_grid_thw alongside pixel_values --
    # this must survive into vision_kwargs, not get silently dropped (the
    # actual bug that broke a real AutoGLM-Phone-9B run).
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _tiny_png_data_uri()}},
        {"type": "text", "text": "hi"},
    ]}]
    _, _, vision_kwargs = _encode_context_and_action_vlm(_FakeGlm4vProcessor(), messages, "do_something", max_len=100)
    assert set(vision_kwargs.keys()) == {"pixel_values", "image_grid_thw"}
    assert vision_kwargs["pixel_values"].shape == (6, 8)
    assert vision_kwargs["image_grid_thw"].tolist() == [[1, 2, 3]]


def test_batch_vision_kwargs_single_record_is_passthrough():
    vk = {"pixel_values": torch.zeros(1, 1, 3, 4, 4)}
    assert _batch_vision_kwargs([vk]) is vk


def test_batch_vision_kwargs_idefics3_style_strips_and_stacks():
    per_record = [
        {"pixel_values": torch.full((1, 1, 3, 2, 2), 1.0)},
        {"pixel_values": torch.full((1, 1, 3, 2, 2), 2.0)},
    ]
    out = _batch_vision_kwargs(per_record)
    assert out["pixel_values"].shape == (2, 1, 3, 2, 2)
    assert out["pixel_values"][0, 0, 0, 0, 0].item() == 1.0
    assert out["pixel_values"][1, 0, 0, 0, 0].item() == 2.0


def test_batch_vision_kwargs_glm4v_style_concatenates():
    per_record = [
        {"pixel_values": torch.ones(6, 8), "image_grid_thw": torch.tensor([[1, 2, 3]])},
        {"pixel_values": torch.ones(3, 8) * 2, "image_grid_thw": torch.tensor([[1, 1, 3]])},
    ]
    out = _batch_vision_kwargs(per_record)
    assert out["pixel_values"].shape == (9, 8)
    assert out["image_grid_thw"].shape == (2, 3)
    assert out["image_grid_thw"].tolist() == [[1, 2, 3], [1, 1, 3]]


def test_batch_vision_kwargs_rejects_mismatched_keys():
    per_record = [
        {"pixel_values": torch.zeros(1, 1, 3, 2, 2)},
        {"pixel_values": torch.zeros(6, 8), "image_grid_thw": torch.tensor([[1, 2, 3]])},
    ]
    try:
        _batch_vision_kwargs(per_record)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "same vision_kwargs keys" in str(e)


def test_encode_context_and_action_vlm_raises_instead_of_truncating_into_image_tokens():
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _tiny_png_data_uri()}},
        {"type": "text", "text": "hi"},
    ]}]
    try:
        _encode_context_and_action_vlm(_FakeProcessor(), messages, "a" * 500, max_len=50)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "truncation is not supported" in str(e)
