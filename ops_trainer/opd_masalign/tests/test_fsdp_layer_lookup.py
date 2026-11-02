"""Tests for opd_trainer.py::_find_decoder_layers and _find_vision_tower_blocks,
the FSDP2 sharding granularity lookups used by main()'s --distributed path.

Real incident this pins down: a real 8xA100(40GB) run against
AutoGLM-Phone-9B (GLM-4V architecture) OOM'd inside the vision tower's
forward pass because _find_decoder_layers only ever found the TEXT decoder
layers -- the vision tower fell entirely into the single outer
fully_shard(model) call and got all-gathered as one giant unit instead of
being sharded per-block. These tests build fake module trees shaped like
the three real architectures this reference actually targets (plain causal
LM, GLM-4V-style VLM, SmolVLM/Idefics3-style VLM) to confirm both lookups
find the right ModuleList without needing real weights or a GPU.

Run with: python -m pytest opd_masalign/tests/test_fsdp_layer_lookup.py -v
"""
import pytest
import torch.nn as nn

from opd_masalign.opd_trainer import _find_decoder_layers, _find_vision_tower_blocks


def _decoder_layers(n=2):
    return nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])


def _vision_blocks(n=3):
    return nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])


class _PlainCausalLM(nn.Module):
    """Shape of e.g. LlamaForCausalLM/Qwen2ForCausalLM: model.layers."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = _decoder_layers()
        self.model = inner


class _Glm4vStyleVLM(nn.Module):
    """Shape of Glm4vForConditionalGeneration: model.language_model.layers
    for text decoder blocks, model.visual.blocks for the vision tower."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        language_model = nn.Module()
        language_model.layers = _decoder_layers()
        visual = nn.Module()
        visual.blocks = _vision_blocks()
        inner.language_model = language_model
        inner.visual = visual
        self.model = inner


class _SmolVLMStyleVLM(nn.Module):
    """Shape of SmolVLMForConditionalGeneration: model.text_model.layers for
    text decoder blocks, model.vision_model.encoder.layers for the vision
    tower."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        text_model = nn.Module()
        text_model.layers = _decoder_layers()
        vision_model = nn.Module()
        encoder = nn.Module()
        encoder.layers = _vision_blocks()
        vision_model.encoder = encoder
        inner.text_model = text_model
        inner.vision_model = vision_model
        self.model = inner


class _UnknownArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.something_else = nn.Linear(4, 4)


def test_find_decoder_layers_plain_causal_lm():
    layers = _find_decoder_layers(_PlainCausalLM())
    assert len(layers) == 2


def test_find_decoder_layers_glm4v_style():
    layers = _find_decoder_layers(_Glm4vStyleVLM())
    assert len(layers) == 2


def test_find_decoder_layers_smolvlm_style():
    layers = _find_decoder_layers(_SmolVLMStyleVLM())
    assert len(layers) == 2


def test_find_decoder_layers_unknown_architecture_raises():
    with pytest.raises(NotImplementedError):
        _find_decoder_layers(_UnknownArchitecture())


def test_find_vision_tower_blocks_glm4v_style():
    blocks = _find_vision_tower_blocks(_Glm4vStyleVLM())
    assert len(blocks) == 3


def test_find_vision_tower_blocks_smolvlm_style():
    blocks = _find_vision_tower_blocks(_SmolVLMStyleVLM())
    assert len(blocks) == 3


def test_find_vision_tower_blocks_text_only_returns_none():
    """A text-only model has no vision tower at all -- this must return
    None, not raise, since that's the expected/common case, unlike
    _find_decoder_layers where every path failing means an unrecognized
    architecture."""
    assert _find_vision_tower_blocks(_PlainCausalLM()) is None
