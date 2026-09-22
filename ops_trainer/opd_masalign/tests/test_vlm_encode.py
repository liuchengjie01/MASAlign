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
    a real model/tokenizer.
    """

    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, text, images=None, return_tensors="pt"):
        out = {"input_ids": torch.tensor([[1] * 20])}
        if images:
            out["pixel_values"] = torch.zeros(1, len(images), 3, 4, 4)
        return out


def test_encode_context_and_action_vlm_returns_pixel_values_for_one_image():
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _tiny_png_data_uri()}},
        {"type": "text", "text": "hi"},
    ]}]
    ids, mask, pixel_values = _encode_context_and_action_vlm(_FakeProcessor(), messages, "do_something", max_len=100)
    assert pixel_values is not None
    assert pixel_values.shape == (1, 3, 4, 4)
    assert sum(mask) == len("do_something")  # action span length, per _FakeTokenizer
    assert len(ids) == 20 + len("do_something")


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
