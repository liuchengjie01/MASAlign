"""OPD training loop: consumes CriticalStepRecord JSONL (docs/DATA_FORMAT.md)
directly -- critical steps are located and outcome-labeled upstream (e.g.
against a collaborator's raw rollout logs, see
opd_masalign/extract_critical_steps.py), so no on-policy generation/vLLM and
no tree/graph/risk pipeline is needed here. Each gradient step is a plain
teacher-forced forward/backward pass:
  1. student forward on observation_context + action_taken -> pi_theta(a_u|u)
  2. EMA-teacher forward on reprompt(observation_context, f_u) + action_taken
     -> q_theta_ema(a_u|u,f_u)
  3. reverse-KL loss, every critical step weighted equally (losses.compute_opd_loss)
  4. EMA teacher update

--model_name is never hardcoded to a particular model anywhere in this file
-- whatever checkpoint is passed on the command line is what gets trained
and what old_action_logprob (see OPDTrainer.precompute_old_log_probs) gets
self-computed from.

SCOPE OF THIS FILE: single-process (CPU or 1 GPU) reference implementation,
for validating the data format + loss end-to-end on a small model before
touching the 8xA100 production path. It intentionally does NOT wire up
FSDP/DeepSpeed/accelerate yet -- multi-GPU scaling for the real 9B-scale
training job is a follow-up once this smoke test is confirmed correct
(either by adding an `accelerate` launch config here, or by porting the
losses.py math into the SDPO/verl recipe, which already has FSDP+vLLM
plumbing -- see docs/DATA_FORMAT.md and the project README for the tradeoff).

Usage (text-only smoke test, CPU or 1 GPU):
    python -m opd_masalign.opd_trainer \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --data_path data/critical_steps_smoke.jsonl \
        --output_dir /tmp/opd_smoke --max_steps 5

Usage (VLM smoke test -- observation_context.messages contains OpenAI-style
`{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}`
content blocks, e.g. mobile-agent screenshots like data/luis/test_log.zip;
data/critical_steps_autoglm.jsonl is real data with real screenshots +
real outcome_type labels, produced by opd_masalign/extract_critical_steps.py
against data/luis/ -- there is no synthetic VLM smoke fixture, since a real
extracted file is already available and just as fast to smoke-test with):
    python -m opd_masalign.opd_trainer --vlm \
        --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
        --data_path data/critical_steps_autoglm.jsonl \
        --output_dir /tmp/opd_vlm_smoke --max_steps 4 --dtype fp32 \
        --disable_image_splitting --image_max_side 512
  (--dtype fp32 matters on CPU: bf16 with no native hardware support can be
  10-50x slower there. --disable_image_splitting is required once
  --batch_size > 1, since the processor's default high-res tiling produces a
  different number of image tokens per image depending on its resolution,
  which this reference Collator cannot batch; it also avoids blowing past a
  constrained CPU's memory ulimit on full-resolution phone screenshots.)
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist  # only ever used (dist.barrier() etc) when main() has already called
                                   # dist.init_process_group() -- importing the module itself needs no backend
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .losses import compute_opd_loss, ema_update, _fsdp_version, _fsdp2_full_param_state_dict_rank0
from .optim import AdamWSR
from . import guiowl_format
from .replan import generate_teacher_replan, substitute_plan
from .reprompt_variants import (
    DEFAULT_EXECUTOR_VARIANT, DEFAULT_PLANNER_VARIANT, PLANNER_VARIANTS, VARIANTS, build_planner_reprompt_text_variant,
)
from .self_teacher import build_teacher_messages

MODEL_FORMATS = ("autoglm", "guiowl")


def _think_prefix(record: dict, model_format: str = "autoglm") -> str:
    r"""Assistant-turn text that sits between the generation prompt and the action when scoring with
    --think_prefill. AutoGLM-Phone's real generations are
        <think> \n {reasoning} \n </think> \n <answer> \n do(action=...) \n </answer>
    (<think>/<answer> are single special tokens; verified on token-level log-probs from the live server),
    so the action is scored AFTER this prefix -- scoring it straight after `<|assistant|>\n` (the
    legacy behaviour) puts the first action token at log-prob ~ -31 on the real checkpoint, i.e. a
    position the model never produces it from. The reasoning is the step's own recorded reasoning
    (action_taken.thinking, written by extract_critical_steps.py), held fixed for both student and
    teacher, so only the action tokens carry loss, exactly as before.

    model_format="guiowl" uses GUI-Owl's plain-text headers instead (guiowl_format.format_think_prefix:
    `### Thought ###\n{reasoning}\n\n### Action ###\n`).
    """
    thinking = record["action_taken"].get("thinking") or ""
    if model_format == "guiowl":
        return guiowl_format.format_think_prefix(thinking)
    return _format_think_prefix(thinking)


def _format_think_prefix(thinking: str) -> str:
    return f"<think>\n{thinking}\n</think>\n<answer>\n"


def _encode_context_and_action_text(tokenizer, messages: list[dict], action_text: str, max_len: int,
                                    assistant_prefix: str = ""):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + assistant_prefix
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    action_ids = tokenizer(action_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + action_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(action_ids)
    if len(input_ids) > max_len:
        # keep the tail (drop earliest context first) so the action span survives
        overflow = len(input_ids) - max_len
        input_ids = input_ids[overflow:]
        response_mask = response_mask[overflow:]
    return input_ids, response_mask


def _decode_image_block(block: dict) -> Image.Image:
    """Decode one OpenAI-style `{"type": "image_url", "image_url": {"url": ...}}`
    content block into a PIL Image. Only base64 data-URIs are supported --
    that's the wire format real collaborator rollout logs use (see
    data/luis/test_log.zip); remote http(s) URLs are out of scope here.
    """
    url = block["image_url"]["url"]
    if not url.startswith("data:"):
        raise NotImplementedError(
            f"only base64 data-URI images are supported, got a non-data URL: {url[:40]}..."
        )
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _cap_longer_side(img: Image.Image, max_side: int) -> Image.Image:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img


def _extract_images_and_normalize(messages: list[dict]) -> tuple[list[dict], list[Image.Image]]:
    """Split OpenAI-style multimodal `content` blocks into (processor-ready
    messages with bare `{"type": "image"}` placeholders in place of the
    inline image data, ordered list of decoded PIL images) for
    AutoProcessor.apply_chat_template. Messages whose `content` is a plain
    string (e.g. self_teacher's appended hindsight-review turn) pass through
    unchanged.
    """
    images: list[Image.Image] = []
    normalized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_blocks = []
            for block in content:
                if block.get("type") in ("image_url", "image"):
                    images.append(_decode_image_block(block))
                    new_blocks.append({"type": "image"})
                else:
                    new_blocks.append(block)
            normalized.append({**msg, "content": new_blocks})
        else:
            normalized.append(msg)
    return normalized, images


def _encode_context_and_action_vlm(
    processor, messages: list[dict], action_text: str, max_len: int, image_max_side: int | None = None,
    assistant_prefix: str = "",
):
    """Returns (input_ids, response_mask, vision_kwargs). vision_kwargs is
    whatever the processor returned besides input_ids/attention_mask, kept
    exactly as shaped for this ONE record (not stripped/reshaped) -- VLM
    families disagree on convention: Idefics3/SmolVLM returns pixel_values
    shaped (1, num_images, C, H, W) with a real per-call batch dim; GLM-4V/
    Qwen2-VL-style models instead return pixel_values already flattened
    across images as (total_patches, feat_dim) with NO batch dim, plus a
    separate image_grid_thw (num_images, 3) telling the model how to
    reassemble them. Indexing into pixel_values (e.g. `[0]`) like the old
    version of this function did is only correct for the first convention --
    for the second it silently returns one patch's feature vector instead of
    the whole image. Keeping vision_kwargs as-is defers the
    convention-specific batching decision to the Collator, where it can be
    handled per family (see Collator.__call__).
    """
    normalized_messages, images = _extract_images_and_normalize(messages)
    if image_max_side:
        images = [_cap_longer_side(img, image_max_side) for img in images]

    prompt_text = processor.apply_chat_template(normalized_messages, tokenize=False, add_generation_prompt=True)
    prompt_text += assistant_prefix
    if images:
        inputs = processor(text=prompt_text, images=images, return_tensors="pt")
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    else:
        inputs = processor(text=prompt_text, return_tensors="pt")
        vision_kwargs = {}

    prompt_ids = inputs["input_ids"][0].tolist()
    action_ids = processor.tokenizer(action_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + action_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(action_ids)

    if len(input_ids) > max_len:
        # unlike the text path, we can't just drop the earliest tokens here:
        # that could slice into the middle of the vision-token span the
        # processor inserted, desynchronizing input_ids from pixel_values.
        raise ValueError(
            f"VLM prompt+action is {len(input_ids)} tokens > --max_len={max_len}; truncation is not "
            "supported for multimodal contexts (would risk cutting the vision-token span). "
            "Increase --max_len, downsize --image_max_side, or shorten action_taken.raw_text."
        )
    return input_ids, response_mask, vision_kwargs


@torch.no_grad()
def generate_teacher_thinking(model, collator, messages: list[dict], max_new_tokens: int, temperature: float,
                              top_p: float, synced_gpus: bool = False) -> tuple[str, int, bool, bool]:
    """Let `model` (the EMA teacher) write its own reasoning for `messages` (observation_context + the
    hindsight reprompt): the prompt ends with the model's `<think>` token and generation stops at `</think>`
    (ending it with `<think>\n` instead made the model skip its reasoning on 19/19 test records).
    Returns (reasoning text, number of generated tokens, unclosed, skipped).
      unclosed: hit max_new_tokens without closing the think block (the partial reasoning is kept).
      skipped:  the model skipped reasoning and went straight to an action / `<answer>`; the reasoning
                is then EMPTY -- the student's recorded reasoning is never substituted.

    synced_gpus=True is required when `model` is FSDP2-sharded and ranks run this concurrently: every
    rank must keep calling forward until the slowest rank finishes, otherwise the parameter
    all-gathers go out of step and the job hangs.
    """
    tok = collator._tok
    if collator.is_vlm:
        ids, _, vk = _encode_context_and_action_vlm(
            collator.processor, messages, "", collator.max_len, collator.image_max_side, "<think>"
        )
    else:
        ids, _ = _encode_context_and_action_text(tok, messages, "", collator.max_len, "<think>")
        vk = {}
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    vk = {k: v.to(device) for k, v in vk.items()}

    close_id = tok.convert_tokens_to_ids("</think>")
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    out = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **vk,
        max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None, top_p=top_p if temperature > 0 else None,
        eos_token_id=[close_id] + eos, pad_token_id=tok.pad_token_id or tok.eos_token_id,
        synced_gpus=synced_gpus, use_cache=True,
    )
    new_ids = out[0, input_ids.shape[1]:].tolist()
    closed = close_id in new_ids
    if closed:
        new_ids = new_ids[: new_ids.index(close_id)]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    skipped = not closed and text.startswith(("do(", "finish("))
    if skipped or not new_ids:
        return "", len(new_ids), False, True
    return text, len(new_ids), not closed, False


class CriticalStepDataset(Dataset):
    """Reads one or more JSONL shards of CriticalStepRecord (docs/DATA_FORMAT.md)."""

    def __init__(self, path: str):
        self.records = []
        paths = [path] if os.path.isfile(path) else [
            os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".jsonl")
        ]
        for p in paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


@dataclass
class Collator:
    """Set exactly one of `tokenizer` (text-only CausalLM) or `processor`
    (VLM) depending on which model type `main()` loaded.
    """

    tokenizer: AutoTokenizer | None = None
    processor: "AutoProcessor" | None = None
    max_len: int = 4096
    use_old_logprobs: bool = True
    image_max_side: int | None = None
    disable_image_splitting: bool = False
    think_prefill: bool = False
    model_format: str = "autoglm"  # "autoglm" (<think>/<answer> tokens) or "guiowl" (### Thought ### headers)
    reprompt_variant: str | None = None  # executor-stage reprompt, see self_teacher.build_reprompt_text

    def __post_init__(self):
        if (self.tokenizer is None) == (self.processor is None):
            raise ValueError("Collator needs exactly one of tokenizer (text mode) or processor (VLM mode)")
        self.is_vlm = self.processor is not None
        self._tok = self.processor.tokenizer if self.is_vlm else self.tokenizer
        if self.is_vlm and self.disable_image_splitting and hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_image_splitting = False

    def __call__(self, batch: list[dict]) -> dict:
        student_ids, student_masks = [], []
        teacher_ids, teacher_masks = [], []
        student_vision_kwargs_per_record = []
        old_logprob_seqs = []

        for record in batch:
            ctx = record["observation_context"]
            assert ctx["format"] == "chat_messages", "Collator currently supports chat_messages contexts only"
            action_text = record["action_taken"]["raw_text"]
            teacher_messages = build_teacher_messages(record, self.reprompt_variant)
            prefix = _think_prefix(record, self.model_format) if self.think_prefill else ""

            if self.is_vlm:
                s_ids, s_mask, vk = _encode_context_and_action_vlm(
                    self.processor, ctx["messages"], action_text, self.max_len, self.image_max_side, prefix
                )
                # same observation_context -> same image for the student and
                # EMA-teacher forward passes; only the appended reprompt text
                # differs, so we only need to encode/keep one copy of vk.
                t_ids, t_mask, _ = _encode_context_and_action_vlm(
                    self.processor, teacher_messages, action_text, self.max_len, self.image_max_side, prefix
                )
                student_vision_kwargs_per_record.append(vk)
            else:
                s_ids, s_mask = _encode_context_and_action_text(self.tokenizer, ctx["messages"], action_text, self.max_len, prefix)
                t_ids, t_mask = _encode_context_and_action_text(self.tokenizer, teacher_messages, action_text, self.max_len, prefix)

            student_ids.append(s_ids)
            student_masks.append(s_mask)
            teacher_ids.append(t_ids)
            teacher_masks.append(t_mask)

            # old_action_logprob is self-computed by
            # OPDTrainer.precompute_old_log_probs before training starts (see
            # that method) and written into this record in place -- it is
            # never read from the input JSONL.
            old_lp = record.get("old_action_logprob")
            if old_lp is not None:
                old_logprob_seqs.append(old_lp["logprobs"])
            else:
                old_logprob_seqs.append(None)

        student_batch = self._pad(student_ids, student_masks)
        teacher_batch = self._pad(teacher_ids, teacher_masks)

        out = {
            "records": batch,  # lets OPDTrainer re-build the teacher side when it generates its own reasoning
            "student_input_ids": student_batch[0],
            "student_attention_mask": student_batch[1],
            "student_response_mask": student_batch[2],
            "teacher_input_ids": teacher_batch[0],
            "teacher_attention_mask": teacher_batch[1],
            "teacher_response_mask": teacher_batch[2],
        }

        if self.is_vlm:
            if any(not vk for vk in student_vision_kwargs_per_record):
                raise ValueError(
                    "VLM mode requires every record's observation_context to contain exactly one image "
                    "(text-only records mixed into the same batch are not supported by this reference Collator)"
                )
            out.update(_batch_vision_kwargs(student_vision_kwargs_per_record))

        if self.use_old_logprobs and all(x is not None for x in old_logprob_seqs):
            old_lp_tensor = torch.zeros_like(student_batch[0], dtype=torch.float32)
            for i, lp in enumerate(old_logprob_seqs):
                n = int(student_batch[2][i].sum().item())
                # place the provided per-token old log-probs at the response
                # positions; if lengths mismatch (tokenizer drift vs the
                # collaborator's tokenizer), truncate/pad defensively rather
                # than crash -- but this signals a tokenizer-consistency bug
                # (see docs/DATA_FORMAT.md's "key prerequisite" note).
                resp_positions = student_batch[2][i].nonzero(as_tuple=True)[0]
                m = min(n, len(lp))
                old_lp_tensor[i, resp_positions[:m]] = torch.tensor(lp[:m], dtype=torch.float32)
            out["old_log_probs"] = old_lp_tensor
        else:
            out["old_log_probs"] = None

        return out

    def _pad(self, ids_list, mask_list):
        max_len = max(len(x) for x in ids_list)
        pad_id = self._tok.pad_token_id or self._tok.eos_token_id
        input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        response_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            response_mask[i, : len(mask)] = torch.tensor(mask, dtype=torch.long)
        return input_ids, attention_mask, response_mask


def _batch_vision_kwargs(per_record: list[dict]) -> dict:
    """Combine each record's own vision_kwargs (see _encode_context_and_action_vlm)
    into one batch-ready dict, using whichever batching convention actually
    matches the shapes this VLM's processor produced -- VLM families are NOT
    interchangeable here:

      - Idefics3/SmolVLM-style: only `pixel_values`, per-record shape
        (1, num_images, C, H, W) (a real per-call batch dim, always 1).
        Strip that dim-0, then STACK across records into a new batch dim:
        (B, num_images, C, H, W).
      - GLM-4V/Qwen2-VL-style: `pixel_values` already flattened across
        images with NO batch dim (total_patches, feat_dim), plus a separate
        image_grid_thw (num_images, 3) telling the model how to split it
        back apart. Batching here means CONCATENATING every record's
        tensors along dim 0 -- exactly what that family's own multi-image
        processor call would itself produce.

    With exactly one record both conventions are already correctly shaped
    as-is, so this is a no-op in the (very common, given VLM memory cost)
    batch_size=1 case.
    """
    if len(per_record) == 1:
        return per_record[0]

    keys = set(per_record[0].keys())
    if any(set(vk.keys()) != keys for vk in per_record):
        raise ValueError(
            f"VLM batching requires every record to produce the same vision_kwargs keys, "
            f"got {[sorted(vk.keys()) for vk in per_record]}"
        )

    if keys == {"pixel_values"}:
        return {"pixel_values": torch.stack([vk["pixel_values"][0] for vk in per_record], dim=0)}

    return {k: torch.cat([vk[k] for vk in per_record], dim=0) for k in keys}


def _sequence_log_probs(model, input_ids, attention_mask, response_mask, vision_kwargs=None):
    """Teacher-forced log pi(token_t | token_<t) at every response position."""
    # use_cache=False: this is a single teacher-forced forward over the
    # WHOLE sequence at once, never incremental decoding, so the KV cache
    # HF builds by default (use_cache=True) is pure wasted activation
    # memory here -- real cost on a real 9B VLM with long vision-token
    # sequences (see opd_trainer.py's --gradient_checkpointing docstring).
    model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
    if vision_kwargs:
        model_kwargs.update(vision_kwargs)
    out = model(**model_kwargs)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs_all = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # response_mask is aligned to input_ids; shift to match targets (t+1)
    shifted_mask = response_mask[:, 1:]
    return token_log_probs, shifted_mask


_NON_VISION_BATCH_KEYS = {
    "student_input_ids", "student_attention_mask", "student_response_mask",
    "teacher_input_ids", "teacher_attention_mask", "teacher_response_mask",
    "old_log_probs", "delta_weights", "records",
}


class OPDTrainer:
    def __init__(
        self, model, tokenizer, ema_alpha: float = 0.02, is_clip: float | None = 2.0, lr: float = 1e-5,
        ema_model=None, max_grad_norm: float | None = 1.0,
    ):
        self.model = model
        # copy.deepcopy(model) is the right way to build the EMA teacher ONLY
        # for a plain/unsharded model. An FSDP2-sharded model can't be
        # deepcopy'd cleanly (each rank only holds a shard, and deepcopy
        # doesn't know how to replicate DTensor/FSDPModule wrapping) -- for
        # --distributed runs, main() instead loads and shards a SEPARATE
        # instance of the same checkpoint and passes it in here.
        self.ema_model = ema_model if ema_model is not None else copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.ema_alpha = ema_alpha
        self.is_clip = is_clip
        self.max_grad_norm = max_grad_norm
        self.teacher_gen = None  # set by enable_teacher_generation()
        # bf16 weights + round-to-nearest AdamW drops most small updates (see optim.py); use the
        # stochastic-rounding variant whenever the parameters are bf16
        params = list(model.parameters())
        self.optimizer = (
            AdamWSR(params, lr=lr) if params and params[0].dtype == torch.bfloat16
            else torch.optim.AdamW(params, lr=lr)
        )

    @torch.no_grad()
    def precompute_old_log_probs(self, dataset: "CriticalStepDataset", collator: "Collator") -> None:
        """Self-compute every record's old_action_logprob from THIS model
        (the exact checkpoint/tokenizer training is about to start from) and
        write it into dataset.records in place, mutating the in-memory
        records -- the on-disk JSONL is untouched.

        Why self-compute instead of reading a collaborator-supplied
        old_action_logprob (docs/DATA_FORMAT.md): IS-clip's ratio
        exp(student_logp - old_logp) only means anything when "old" is under
        the SAME tokenizer as "student" -- otherwise token positions don't
        even line up. The one case that's actually guaranteed to satisfy
        that is "the checkpoint this training run itself starts from",
        which this function has direct access to regardless of what
        --model_name was passed (nothing here is hardcoded to a specific
        model). This must run once, BEFORE the first train_step, so the
        cached value reflects the pre-training policy across all
        --epochs -- recomputing it fresh every step (the old fallback) would
        make the IS ratio trivially ~1 forever and defeat the point.
        """
        was_training = self.model.training
        self.model.eval()
        for record in dataset.records:
            ctx = record["observation_context"]
            action_text = record["action_taken"]["raw_text"]
            prefix = _think_prefix(record, collator.model_format) if collator.think_prefill else ""

            if collator.is_vlm:
                ids, mask, vision_kwargs = _encode_context_and_action_vlm(
                    collator.processor, ctx["messages"], action_text, collator.max_len, collator.image_max_side, prefix
                )
                # single record -- already correctly shaped as-is, whichever
                # VLM family's convention this is (see _batch_vision_kwargs).
            else:
                ids, mask = _encode_context_and_action_text(
                    collator.tokenizer, ctx["messages"], action_text, collator.max_len, prefix
                )
                vision_kwargs = None

            device = next(self.model.parameters()).device
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            response_mask = torch.tensor([mask], dtype=torch.long, device=device)
            if vision_kwargs:
                vision_kwargs = {k: v.to(device) for k, v in vision_kwargs.items()}
            log_probs, shifted_mask = _sequence_log_probs(
                self.model, input_ids, attention_mask, response_mask, vision_kwargs=vision_kwargs
            )
            resp_log_probs = log_probs[0][shifted_mask[0].bool()].tolist()
            # token_ids here is informational only (never read by Collator,
            # which only consumes .logprobs) -- the action span is the tail
            # of `ids` by construction of _encode_context_and_action_*.
            action_token_ids = ids[-len(resp_log_probs):] if resp_log_probs else []
            record["old_action_logprob"] = {"token_ids": action_token_ids, "logprobs": resp_log_probs}
        if was_training:
            self.model.train()

    def enable_teacher_generation(self, collator: "Collator", max_new_tokens: int, temperature: float,
                                  top_p: float, synced_gpus: bool, replan_variant: str | None = None,
                                  replan_max_new_tokens: int = 400) -> None:
        """Used with --think_prefill: the EMA teacher writes its own reasoning (under
        observation_context + the hindsight reprompt) before the original action is scored. Handing it
        the student's recorded reasoning instead would pre-decide the conclusion the reprompt is meant
        to influence. The student side keeps its recorded reasoning -- that is what it really produced.

        replan_variant (GUI-Owl, --teacher_replan): instead of appending an executor-stage reprompt, the
        teacher first rewrites the PLANNER's plan under record["planner_context"] + the planner-stage
        reprompt (reprompt_variants.PLANNER_VARIANTS[replan_variant]), that plan replaces the executor
        prompt's "### Overall Plan ###"/"### Current Subgoal ###" (replan.substitute_plan), and the
        teacher then writes its thinking under that replanned executor context with no further reprompt.
        On GUI-Owl-7B this lowered the teacher's log-prob of the attacked action, while every
        executor-stage reprompt raised it (see opd_masalign/README.md)."""
        self.teacher_gen = dict(collator=collator, max_new_tokens=max_new_tokens, temperature=temperature,
                                top_p=top_p, synced_gpus=synced_gpus, replan_variant=replan_variant,
                                replan_max_new_tokens=replan_max_new_tokens)

    def _teacher_context(self, record: dict, cfg: dict) -> tuple[list[dict], bool, int]:
        """Teacher-side messages for one record -> (messages, replanned, plan_tokens). Every record makes
        exactly one replan generate() call when replanning is on (main() drops records without a
        planner_context up front), so FSDP ranks with synced_gpus stay in step even when a plan comes
        out empty; that record then falls back to the executor-stage reprompt."""
        collator = cfg["collator"]
        if not cfg.get("replan_variant"):
            return build_teacher_messages(record, collator.reprompt_variant), False, 0
        replan_msgs = list(record["planner_context"]["messages"]) + [
            {"role": "user", "content": build_planner_reprompt_text_variant(record, cfg["replan_variant"])}
        ]
        new_plan, _, n_plan = generate_teacher_replan(
            self.ema_model, collator, replan_msgs, cfg["replan_max_new_tokens"], cfg["temperature"],
            cfg["top_p"], cfg["synced_gpus"],
        )
        orig = record["observation_context"]["messages"]
        if new_plan:
            messages = substitute_plan(orig, new_plan)
            if messages != orig:
                return messages, True, n_plan
        return build_teacher_messages(record, collator.reprompt_variant), False, n_plan

    def _rebuild_teacher_batch(self, batch: dict, device) -> tuple[dict, dict]:
        cfg = self.teacher_gen
        collator = cfg["collator"]
        self.ema_model.eval()
        guiowl = collator.model_format == "guiowl"
        gen_thinking = guiowl_format.generate_teacher_thinking if guiowl else generate_teacher_thinking
        fmt_prefix = guiowl_format.format_think_prefix if guiowl else _format_think_prefix
        t_ids, t_masks, n_tokens, n_truncated, n_skipped = [], [], [], 0, 0
        n_replanned, plan_tokens = 0, []
        for record in batch["records"]:
            messages, replanned, n_plan = self._teacher_context(record, cfg)
            n_replanned += int(replanned)
            plan_tokens.append(n_plan)
            thinking, n, truncated, skipped = gen_thinking(
                self.ema_model, collator, messages, cfg["max_new_tokens"], cfg["temperature"],
                cfg["top_p"], cfg["synced_gpus"],
            )
            prefix = fmt_prefix(thinking)
            action_text = record["action_taken"]["raw_text"]
            if collator.is_vlm:
                ids, mask, _ = _encode_context_and_action_vlm(
                    collator.processor, messages, action_text, collator.max_len, collator.image_max_side, prefix
                )
            else:
                ids, mask = _encode_context_and_action_text(collator._tok, messages, action_text, collator.max_len, prefix)
            t_ids.append(ids)
            t_masks.append(mask)
            n_tokens.append(n)
            n_truncated += int(truncated)
            n_skipped += int(skipped)
        input_ids, attention_mask, response_mask = collator._pad(t_ids, t_masks)
        batch = dict(batch)
        batch["teacher_input_ids"] = input_ids.to(device)
        batch["teacher_attention_mask"] = attention_mask.to(device)
        batch["teacher_response_mask"] = response_mask.to(device)
        metrics = {
            "opd/teacher_think_tokens": sum(n_tokens) / len(n_tokens),
            "opd/teacher_think_unclosed": float(n_truncated),
            "opd/teacher_think_skipped": float(n_skipped),
        }
        if cfg.get("replan_variant"):
            metrics["opd/teacher_replanned"] = float(n_replanned)
            metrics["opd/teacher_plan_tokens"] = sum(plan_tokens) / len(plan_tokens)
        return batch, metrics

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        device = next(self.model.parameters()).device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        gen_metrics = {}
        if self.teacher_gen is not None:
            batch, gen_metrics = self._rebuild_teacher_batch(batch, device)
        # anything in the batch besides the known text/label keys is a
        # vision kwarg the Collator batched (pixel_values, and for some VLM
        # families also image_grid_thw etc. -- see _batch_vision_kwargs).
        vision_kwargs = {k: v for k, v in batch.items() if k not in _NON_VISION_BATCH_KEYS}
        student_log_probs, student_mask = _sequence_log_probs(
            self.model, batch["student_input_ids"], batch["student_attention_mask"], batch["student_response_mask"],
            vision_kwargs=vision_kwargs,
        )
        with torch.no_grad():
            teacher_log_probs, teacher_mask = _sequence_log_probs(
                self.ema_model, batch["teacher_input_ids"], batch["teacher_attention_mask"], batch["teacher_response_mask"],
                vision_kwargs=vision_kwargs,
            )

        old_log_probs_shifted = batch["old_log_probs"][:, 1:] if batch.get("old_log_probs") is not None else None
        student_resp, teacher_resp, align_mask, old_resp = _align_student_teacher(
            student_log_probs, student_mask, teacher_log_probs, teacher_mask, old_log_probs_shifted
        )

        loss, metrics = compute_opd_loss(
            student_log_probs=student_resp,
            teacher_log_probs=teacher_resp,
            response_mask=align_mask,
            old_log_probs=old_resp if old_resp is not None else student_resp.detach(),
            is_clip=self.is_clip,
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            # returns the pre-clip global norm; works with FSDP2 DTensor gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            metrics["opd/grad_norm"] = float(grad_norm)
        self.optimizer.step()
        ema_update(self.model, self.ema_model, self.ema_alpha)

        metrics["opd/loss"] = loss.item()
        metrics.update(gen_metrics)
        return metrics


def _align_student_teacher(
    student_log_probs: torch.Tensor,
    student_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_mask: torch.Tensor,
    old_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Extract each row's own valid action-token log-probs from student and
    teacher -- which live at different absolute offsets within their
    respective sequences, since the teacher's reprompt context is longer --
    and pack them into aligned (B, T) tensors for compute_opd_loss.

    Per row i, student and teacher get the SAME a_u tokens appended (same
    text, same tokenizer), so their own valid lengths normally already
    match; taking the last min(student_len_i, teacher_len_i) is a safety
    net for the rare case they don't (e.g. one side got its action span
    clipped by an unusually small --max_len), not the common case.

    Different ROWS (different critical steps) usually have different action
    lengths, though -- e.g. "done" (1 token) vs a long structured action
    string (dozens of tokens) in the same batch. T is the MAX such per-row
    length in this batch, and shorter rows are padded (not the other way
    around): a previous version instead cropped every row down to the
    batch's SHORTEST row via a global min, which silently dropped real
    tokens -- and their gradient contribution -- from every longer row's
    action span. Padding costs nothing extra here since compute_opd_loss's
    masked_mean already ignores align_mask==0 positions correctly.

    Returns (student_resp, teacher_resp, align_mask, old_resp), all shape
    (B, T) except old_resp, which is None when old_log_probs is None.
    """
    B = student_log_probs.shape[0]
    student_lens = student_mask.sum(-1)
    teacher_lens = teacher_mask.sum(-1)
    per_row_k = torch.minimum(student_lens, teacher_lens)
    T = int(per_row_k.max().item()) if B > 0 else 0

    student_resp = torch.zeros(B, T, dtype=student_log_probs.dtype, device=student_log_probs.device)
    teacher_resp = torch.zeros(B, T, dtype=teacher_log_probs.dtype, device=teacher_log_probs.device)
    align_mask = torch.zeros(B, T, dtype=torch.long, device=student_log_probs.device)
    old_resp = torch.zeros(B, T, dtype=old_log_probs.dtype, device=old_log_probs.device) if old_log_probs is not None else None

    for i in range(B):
        k_i = int(per_row_k[i].item())
        if k_i == 0:
            continue
        s_idx = student_mask[i].nonzero(as_tuple=True)[0][-k_i:]
        t_idx = teacher_mask[i].nonzero(as_tuple=True)[0][-k_i:]
        student_resp[i, :k_i] = student_log_probs[i, s_idx]
        teacher_resp[i, :k_i] = teacher_log_probs[i, t_idx]
        align_mask[i, :k_i] = 1
        if old_log_probs is not None:
            old_resp[i, :k_i] = old_log_probs[i, s_idx]

    return student_resp, teacher_resp, align_mask, old_resp


def _find_decoder_layers(model: torch.nn.Module):
    """Best-effort lookup of the list of transformer decoder blocks to wrap
    individually with fully_shard (matches production FSDP2 usage: shard
    per-block, not just the whole model -- see
    manual_multi_rank_train_step_check.py). Different model families nest
    this list at different attribute paths; only the two paths this
    reference has actually been validated against are supported here.
    Fails loudly rather than silently wrapping nothing (or the wrong thing)
    for an architecture nobody's confirmed FSDP2 sharding against yet.
    """
    # plain causal LM / GLM-4V-style VLM / SmolVLM-Idefics3-style VLM
    for path in ("model.layers", "model.language_model.layers", "model.text_model.layers"):
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise NotImplementedError(
        f"--distributed (FSDP2) doesn't know how to find {type(model).__name__}'s decoder layers to "
        f"shard -- only 'model.layers' (most causal LMs), 'model.language_model.layers' (GLM-4V-style "
        f"VLMs), and 'model.text_model.layers' (SmolVLM/Idefics3-style VLMs) are supported. Confirm this "
        f"model's actual transformer-block attribute path and add it here."
    )


def _find_vision_tower_blocks(model: torch.nn.Module):
    """Best-effort lookup of a VLM's SEPARATE vision-encoder transformer
    blocks (the ViT that turns image patches into embeddings), distinct
    from _find_decoder_layers' text decoder blocks above.

    Real incident this fixes: a real 8xA100(40GB) run against AutoGLM-Phone-9B
    (GLM-4V architecture) OOM'd inside the vision tower's forward pass even
    though the text decoder layers were sharded per-block via
    _find_decoder_layers. Root cause: _find_decoder_layers only shards
    model.language_model.layers -- the vision tower (model.visual, a whole
    separate stack of Glm4vVisionBlock modules) was never individually
    wrapped, so it fell entirely into the single outer fully_shard(model)
    call in main() and got treated as ONE giant FSDP unit. FSDP2 unshards
    (all-gathers) a unit's full parameters at once before using them, so
    that one call was materializing the ENTIRE vision tower's parameters on
    every GPU simultaneously, instead of one block's worth at a time --
    exactly the memory blow-up FSDP2 sharding is supposed to avoid.

    Returns None (not an error) when no separate vision tower is found --
    unlike _find_decoder_layers, a missing vision tower is the EXPECTED case
    for a text-only model, not an unrecognized architecture.
    """
    for path in ("model.visual.blocks", "model.vision_model.encoder.layers"):  # GLM-4V / SmolVLM-Idefics3
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _save_checkpoint(output_dir, trainer, tokenizer, processor, args, distributed, is_main_process) -> None:
    """Save trainer.model (+tokenizer/processor) to output_dir. Shared by the
    periodic --save_every checkpoints and the final end-of-run save so both
    go through the exact same FSDP2-gather-aware path (see main()'s original
    single-use version of this code, before --save_every existed).
    """
    if distributed:
        dist.barrier()  # every rank must reach here before any rank starts the collective state-dict gather below
        if _fsdp_version(trainer.model) == 2:
            full_state = _fsdp2_full_param_state_dict_rank0(trainer.model)
            if is_main_process:
                trainer.model.save_pretrained(output_dir, state_dict=full_state)
        elif is_main_process:
            trainer.model.save_pretrained(output_dir)
    elif is_main_process:
        trainer.model.save_pretrained(output_dir)

    if is_main_process:
        if args.vlm:
            processor.save_pretrained(output_dir)
        else:
            tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="full passes over --data_path. old_action_logprob is self-computed from the "
        "pre-training checkpoint (see OPDTrainer.precompute_old_log_probs) so IS-clip still has "
        "a staleness signal across epochs; SDPO's reference PPO config still uses ppo_epochs=1 "
        "for the same underlying reason (policy drift), so keep this small regardless.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--ema_alpha", type=float, default=0.02,
        help="teacher <- alpha * student + (1 - alpha) * teacher each step; 0 keeps the teacher fixed at the initial model",
    )
    # with --think_prefill the EMA teacher writes its OWN reasoning under observation_context + hindsight
    # reprompt, and the original action is scored after that reasoning (the student keeps its recorded one)
    parser.add_argument("--teacher_think_max_tokens", type=int, default=1024)
    parser.add_argument("--teacher_think_temperature", type=float, default=0.6)
    parser.add_argument("--teacher_think_top_p", type=float, default=0.9)
    parser.add_argument(
        "--model_format", choices=MODEL_FORMATS, default="autoglm",
        help="response grammar of the model being trained: autoglm = <think>..</think><answer> special tokens; "
        "guiowl = GUI-Owl/Mobile-Agent-V3 executor headers (### Thought ### / ### Action ###), records from "
        "extract_critical_steps.py --format guiowl",
    )
    parser.add_argument(
        "--reprompt_variant", choices=["v0", *VARIANTS], default=DEFAULT_EXECUTOR_VARIANT,
        help="executor-stage hindsight reprompt appended to observation_context for the teacher "
        "(reprompt_variants.VARIANTS; v0 = the original popup-only self_teacher.py template). With "
        "--teacher_replan it is only the fallback for records whose replan comes out empty.",
    )
    parser.add_argument(
        "--teacher_replan", action="store_true",
        help="GUI-Owl only, needs --think_prefill: the teacher rewrites the planner's plan under "
        "record.planner_context + the --planner_variant reprompt, substitutes it into the executor prompt and "
        "writes its thinking there with no executor reprompt (see OPDTrainer.enable_teacher_generation). "
        "Records without planner_context are dropped.",
    )
    parser.add_argument("--planner_variant", choices=list(PLANNER_VARIANTS), default=DEFAULT_PLANNER_VARIANT)
    parser.add_argument("--teacher_replan_max_tokens", type=int, default=400)
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="clip the global gradient norm to this value before each optimizer step; 0 disables",
    )
    parser.add_argument("--is_clip", type=float, default=2.0)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument(
        "--train_outcomes", default="attack_success",
        help="comma-separated outcome_type values to train on (default: attack_success only). The hindsight "
        "reprompt says the action was an attack-induced mistake, which is wrong for benign_success steps; "
        "pass an empty string to train on every record.",
    )
    parser.add_argument("--no_is_clip", action="store_true", help="disable IS-clip term and skip the old_action_logprob precompute pass entirely")
    parser.add_argument(
        "--vlm", action="store_true",
        help="load --model_name as a vision-language model (AutoProcessor + AutoModelForImageTextToText) instead of "
        "a text-only causal LM, for observation_context.messages containing OpenAI-style image_url content blocks",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp32"], default="bf16",
        help="bf16 needs native hardware support -- on a CPU without it (no AVX512-BF16/AMX), forward/backward "
        "can be 10-50x slower than fp32; use fp32 for CPU smoke tests",
    )
    parser.add_argument(
        "--disable_image_splitting", action="store_true",
        help="VLM only: turn off the processor's high-res tiling. Recommended for phone-screenshot-scale images "
        "(e.g. 1080x2400) -- tiling can exceed a constrained machine's memory ulimit, and is required for "
        "batch_size > 1 since tile count otherwise varies per image and can't be batched by this reference Collator",
    )
    parser.add_argument(
        "--image_max_side", type=int, default=None,
        help="VLM only: downscale images so the longer side is at most this many pixels before encoding "
        "(aspect ratio preserved)",
    )
    parser.add_argument(
        "--save_every", type=int, default=None,
        help="also save an intermediate checkpoint every N steps, to {output_dir}/checkpoint-{step}/ "
        "(the final checkpoint is still saved to --output_dir itself when training finishes, regardless of "
        "this flag). Off by default: previously a crash/OOM/Slurm-timeout mid-run lost the entire run, since "
        "only that final checkpoint ever got written -- see opd_masalign/README.md's known limitations. Each "
        "checkpoint is a full copy of the model, so watch disk usage for a real multi-billion-parameter model.",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="trade compute for activation memory on the student's backward pass (recompute activations "
        "during backward instead of keeping all of them resident). FSDP2 (--distributed via torchrun) only "
        "shards PARAMETERS/gradients/optimizer state across GPUs -- it does NOT shard activation memory, "
        "which still scales with batch_size x sequence_length x hidden_size x num_layers same as single-GPU. "
        "For a real multi-billion-parameter VLM, a long vision-token sequence (many images and/or high "
        "--image_max_side) can OOM on activations alone even across 8 GPUs with plenty of aggregate memory "
        "for the model itself -- this is the first thing to try if that happens.",
    )
    parser.add_argument(
        "--think_prefill", action="store_true",
        help="score the action after the model's real assistant prefix `<think>\\n{reasoning}\\n</think>\\n<answer>\\n` "
        "(reasoning = action_taken.thinking from extract_critical_steps.py) instead of straight after "
        "`<|assistant|>\\n`. Use for AutoGLM-Phone-style models whose chat template has <think>/<answer> special "
        "tokens; leave off for models without them (e.g. the SmolVLM/Qwen smoke-test checkpoints). With "
        "--model_format guiowl the prefix is `### Thought ###\\n{reasoning}\\n\\n### Action ###\\n` instead.",
    )
    args = parser.parse_args()
    if args.teacher_replan and not (args.model_format == "guiowl" and args.think_prefill and args.vlm):
        parser.error("--teacher_replan needs --model_format guiowl --think_prefill --vlm")

    # Multi-GPU (FSDP2) launch: `torchrun --nproc_per_node=N -m opd_masalign.opd_trainer ...`
    # sets WORLD_SIZE/LOCAL_RANK/RANK. world_size==1 (the default when not
    # launched via torchrun) is the original single-process path, unchanged.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        local_rank, rank = 0, 0
        # explicit index (cuda:0), not bare "cuda" -- torch.device("cuda").index is
        # None until a CUDA op actually runs, and device_map={"": None} crashes deep
        # inside transformers' caching_allocator_warmup (torch.device(None) TypeError)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main_process = rank == 0

    def _log(msg: str) -> None:
        if is_main_process:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    _log(f"using device: {device} (world_size={world_size})")
    # device_map streams weights straight into GPU memory as each shard is
    # read off disk, instead of materializing the full model in CPU RAM
    # first and only THEN copying it to GPU via a separate .to(device) --
    # for a small smoke-test model the difference is invisible, but for a
    # real ~9B checkpoint the CPU-first path means loading twice the memory
    # (transiently) and a slow, silent multi-GB PCIe copy with no progress
    # output, which looks exactly like a hang. {"": device.index} means
    # "this rank's own GPU" (index 0 in the non-distributed case, or
    # local_rank under torchrun -- respects CUDA_VISIBLE_DEVICES either way).
    device_map = {"": device.index} if device.type == "cuda" else None

    def load_model_and_tokenizer():
        if args.vlm:
            processor = AutoProcessor.from_pretrained(args.model_name)
            if not hasattr(processor, "tokenizer") or not hasattr(processor, "image_processor"):
                # A text-only checkpoint has no preprocessor_config.json for images, so
                # AutoProcessor silently falls back to returning a bare tokenizer instead
                # of raising -- the failure only shows up later as a confusing
                # "PreTrainedTokenizerFast has no attribute 'tokenizer'" deep inside
                # this function. Catch it here with a message that actually says what's wrong.
                raise SystemExit(
                    f"--vlm was set but AutoProcessor.from_pretrained({args.model_name!r}) returned a bare "
                    f"{type(processor).__name__}, not a real multimodal processor -- this checkpoint has no "
                    f"image preprocessor config, so it's a text-only model. --model_name must be a "
                    f"vision-language checkpoint for --vlm (e.g. HuggingFaceTB/SmolVLM-256M-Instruct, "
                    f"Qwen/Qwen2-VL-2B-Instruct -- note there is no 0.5B Qwen VL variant). "
                    f"Drop --vlm if you want text-only training instead."
                )
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            m = AutoModelForImageTextToText.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device_map)
            return m, processor.tokenizer, processor
        else:
            tok = AutoTokenizer.from_pretrained(args.model_name)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            m = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device_map)
            return m, tok, None

    _log(f"loading student from {args.model_name} ...")
    t0 = time.time()
    model, tokenizer, processor = load_model_and_tokenizer()
    _log(f"student loaded in {time.time() - t0:.1f}s")
    if args.gradient_checkpointing:
        # EMA teacher deliberately excluded: its forward always runs under
        # torch.no_grad() (see OPDTrainer.train_step), so it never builds a
        # backward graph and has no activations to checkpoint against.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # incompatible with gradient checkpointing during training

    ema_model = None
    if distributed:
        from torch.distributed._composable.fsdp import fully_shard
        for layer in _find_decoder_layers(model):
            fully_shard(layer)
        for blk in (_find_vision_tower_blocks(model) or []):
            fully_shard(blk)
        fully_shard(model)

        # EMA teacher must be a SEPARATE instance of the same checkpoint,
        # sharded the same way -- copy.deepcopy (OPDTrainer's default when
        # ema_model=None) doesn't work on an FSDP2-sharded module, see
        # OPDTrainer.__init__'s docstring and
        # manual_multi_rank_train_step_check.py.
        t0 = time.time()
        ema_model, _, _ = load_model_and_tokenizer()
        _log(f"EMA teacher loaded in {time.time() - t0:.1f}s")
        for layer in _find_decoder_layers(ema_model):
            fully_shard(layer)
        for blk in (_find_vision_tower_blocks(ema_model) or []):
            fully_shard(blk)
        fully_shard(ema_model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    dataset = CriticalStepDataset(args.data_path)
    if args.train_outcomes:
        keep = set(args.train_outcomes.split(","))
        before = len(dataset.records)
        dataset.records = [r for r in dataset.records if r.get("outcome_type") in keep]
        _log(f"--train_outcomes {sorted(keep)}: kept {len(dataset.records)}/{before} records")
        if not dataset.records:
            raise SystemExit("no records left after --train_outcomes filtering")
    if args.teacher_replan:
        before = len(dataset.records)
        dataset.records = [r for r in dataset.records if (r.get("planner_context") or {}).get("messages")]
        _log(f"--teacher_replan: kept {len(dataset.records)}/{before} records that carry planner_context")
        if not dataset.records:
            raise SystemExit("--teacher_replan but no record has planner_context; re-extract with "
                             "extract_critical_steps.py --format guiowl")

    if args.epochs > 1 and args.no_is_clip:
        print(
            f"WARNING: --epochs={args.epochs} > 1 with --no_is_clip: IS-clip is fully disabled, so "
            f"there is NO safety valve for policy drift from the pre-training policy across epochs."
        )

    collator = Collator(
        tokenizer=None if args.vlm else tokenizer,
        processor=processor if args.vlm else None,
        max_len=args.max_len,
        use_old_logprobs=not args.no_is_clip,
        image_max_side=args.image_max_side,
        disable_image_splitting=args.disable_image_splitting,
        think_prefill=args.think_prefill,
        model_format=args.model_format,
        reprompt_variant=args.reprompt_variant,
    )
    _log(f"teacher: model_format={args.model_format} executor reprompt={args.reprompt_variant} "
         f"replan={'planner ' + args.planner_variant if args.teacher_replan else 'off'}")
    if args.think_prefill:
        n_missing = sum(1 for r in dataset.records if r["action_taken"].get("thinking") is None)
        if n_missing:
            _log(
                f"WARNING: --think_prefill but {n_missing}/{len(dataset.records)} records have no "
                f"action_taken.thinking; they get an EMPTY reasoning block. Re-extract with "
                f"extract_critical_steps.py to populate it."
            )

    trainer = OPDTrainer(
        model, processor if args.vlm else tokenizer, ema_alpha=args.ema_alpha,
        is_clip=None if args.no_is_clip else args.is_clip, lr=args.lr, ema_model=ema_model,
        max_grad_norm=args.max_grad_norm,
    )

    if args.think_prefill:
        trainer.enable_teacher_generation(
            collator, args.teacher_think_max_tokens, args.teacher_think_temperature,
            args.teacher_think_top_p, synced_gpus=distributed,
            replan_variant=args.planner_variant if args.teacher_replan else None,
            replan_max_new_tokens=args.teacher_replan_max_tokens,
        )

    if not args.no_is_clip:
        _log(f"precomputing old_action_logprob from {args.model_name} for {len(dataset.records)} records...")
        # Runs redundantly on every rank (each computes the same
        # deterministic eval-mode forward over the FULL dataset.records,
        # not a rank-specific shard) -- correct, since every rank starts
        # from identical weights, but not distributed-optimized. Splitting
        # this across ranks and gathering results would be the efficient
        # version; skipped here as a real, documented scope limitation.
        t0 = time.time()
        trainer.precompute_old_log_probs(dataset, collator)
        _log(f"old_action_logprob precompute done in {time.time() - t0:.1f}s")

    sampler = None
    if distributed:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, collate_fn=collator)

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed:
        dist.barrier()  # other ranks wait for output_dir to exist before anything reads/writes under it
    # metrics only ever went to stdout before -- once that scrollback is gone
    # (e.g. a Slurm job's stdout capture expires) there was no record left of
    # the loss/distillation-signal/gradient-adjacent metrics a run produced.
    # metrics.jsonl gives every run (smoke test or real) a persistent,
    # inspectable record in --output_dir itself. Only rank 0 writes it --
    # every rank computes the same metrics values (FSDP2 collectives keep
    # them in sync), so N ranks writing the same file would just be N-way
    # duplicate/interleaved noise.
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    step = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)  # re-shuffle differently each epoch, same as DataLoader(shuffle=True) does
        for batch in loader:
            t0 = time.time()
            metrics = trainer.train_step(batch)
            step_time = time.time() - t0
            if is_main_process:
                print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch} step {step} ({step_time:.1f}s): {metrics}")
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"epoch": epoch, "step": step, "step_time_s": step_time, **metrics}) + "\n")
            step += 1
            if args.save_every and step % args.save_every == 0:
                # every rank must call this (not just is_main_process) -- the FSDP2 gather
                # inside _save_checkpoint is a collective op, see that function's docstring.
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                _log(f"step {step}: saving intermediate checkpoint to {ckpt_dir} ...")
                t_ckpt = time.time()
                _save_checkpoint(ckpt_dir, trainer, tokenizer, processor, args, distributed, is_main_process)
                _log(f"intermediate checkpoint saved in {time.time() - t_ckpt:.1f}s")
            if step >= args.max_steps:
                stop = True
                break

    _log("training loop done, saving checkpoint...")
    t_save = time.time()
    _save_checkpoint(args.output_dir, trainer, tokenizer, processor, args, distributed, is_main_process)
    if is_main_process:
        _log(
            f"training done, {step} steps over {min(epoch + 1, args.epochs)} epoch(s), "
            f"checkpoint at {args.output_dir} (checkpoint save took {time.time() - t_save:.1f}s total)"
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
