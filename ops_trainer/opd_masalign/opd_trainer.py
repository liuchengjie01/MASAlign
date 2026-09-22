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
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from .losses import compute_opd_loss, ema_update
from .self_teacher import build_teacher_messages


def _encode_context_and_action_text(tokenizer, messages: list[dict], action_text: str, max_len: int):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    processor, messages: list[dict], action_text: str, max_len: int, image_max_side: int | None = None
):
    normalized_messages, images = _extract_images_and_normalize(messages)
    if image_max_side:
        images = [_cap_longer_side(img, image_max_side) for img in images]

    prompt_text = processor.apply_chat_template(normalized_messages, tokenize=False, add_generation_prompt=True)
    if images:
        inputs = processor(text=prompt_text, images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # (num_images, C, H, W)
    else:
        inputs = processor(text=prompt_text, return_tensors="pt")
        pixel_values = None

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
    return input_ids, response_mask, pixel_values


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
        student_pixel_values = []
        old_logprob_seqs = []

        for record in batch:
            ctx = record["observation_context"]
            assert ctx["format"] == "chat_messages", "Collator currently supports chat_messages contexts only"
            action_text = record["action_taken"]["raw_text"]
            teacher_messages = build_teacher_messages(record)

            if self.is_vlm:
                s_ids, s_mask, pv = _encode_context_and_action_vlm(
                    self.processor, ctx["messages"], action_text, self.max_len, self.image_max_side
                )
                # same observation_context -> same image for the student and
                # EMA-teacher forward passes; only the appended reprompt text
                # differs, so we only need to encode/keep one pixel_values.
                t_ids, t_mask, _ = _encode_context_and_action_vlm(
                    self.processor, teacher_messages, action_text, self.max_len, self.image_max_side
                )
                student_pixel_values.append(pv)
            else:
                s_ids, s_mask = _encode_context_and_action_text(self.tokenizer, ctx["messages"], action_text, self.max_len)
                t_ids, t_mask = _encode_context_and_action_text(self.tokenizer, teacher_messages, action_text, self.max_len)

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
            "student_input_ids": student_batch[0],
            "student_attention_mask": student_batch[1],
            "student_response_mask": student_batch[2],
            "teacher_input_ids": teacher_batch[0],
            "teacher_attention_mask": teacher_batch[1],
            "teacher_response_mask": teacher_batch[2],
        }

        if self.is_vlm:
            if any(pv is None for pv in student_pixel_values):
                raise ValueError(
                    "VLM mode requires every record's observation_context to contain exactly one image "
                    "(text-only records mixed into the same batch are not supported by this reference Collator)"
                )
            # requires every image to produce the same (num_images, C, H, W)
            # shape -- true whenever --disable_image_splitting is set (fixed
            # canonical resize), which is why it's required for batch_size > 1.
            out["pixel_values"] = torch.stack(student_pixel_values, dim=0)

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


def _sequence_log_probs(model, input_ids, attention_mask, response_mask, pixel_values=None):
    """Teacher-forced log pi(token_t | token_<t) at every response position."""
    model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if pixel_values is not None:
        model_kwargs["pixel_values"] = pixel_values
    out = model(**model_kwargs)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs_all = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # response_mask is aligned to input_ids; shift to match targets (t+1)
    shifted_mask = response_mask[:, 1:]
    return token_log_probs, shifted_mask


class OPDTrainer:
    def __init__(self, model, tokenizer, ema_alpha: float = 0.02, is_clip: float | None = 2.0, lr: float = 1e-5):
        self.model = model
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.ema_alpha = ema_alpha
        self.is_clip = is_clip
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

            if collator.is_vlm:
                ids, mask, pixel_values = _encode_context_and_action_vlm(
                    collator.processor, ctx["messages"], action_text, collator.max_len, collator.image_max_side
                )
                pixel_values = pixel_values.unsqueeze(0) if pixel_values is not None else None
            else:
                ids, mask = _encode_context_and_action_text(collator.tokenizer, ctx["messages"], action_text, collator.max_len)
                pixel_values = None

            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            response_mask = torch.tensor([mask], dtype=torch.long)
            log_probs, shifted_mask = _sequence_log_probs(
                self.model, input_ids, attention_mask, response_mask, pixel_values=pixel_values
            )
            resp_log_probs = log_probs[0][shifted_mask[0].bool()].tolist()
            # token_ids here is informational only (never read by Collator,
            # which only consumes .logprobs) -- the action span is the tail
            # of `ids` by construction of _encode_context_and_action_*.
            action_token_ids = ids[-len(resp_log_probs):] if resp_log_probs else []
            record["old_action_logprob"] = {"token_ids": action_token_ids, "logprobs": resp_log_probs}
        if was_training:
            self.model.train()

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        pixel_values = batch.get("pixel_values")
        student_log_probs, student_mask = _sequence_log_probs(
            self.model, batch["student_input_ids"], batch["student_attention_mask"], batch["student_response_mask"],
            pixel_values=pixel_values,
        )
        with torch.no_grad():
            teacher_log_probs, teacher_mask = _sequence_log_probs(
                self.ema_model, batch["teacher_input_ids"], batch["teacher_attention_mask"], batch["teacher_response_mask"],
                pixel_values=pixel_values,
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
        self.optimizer.step()
        ema_update(self.model, self.ema_model, self.ema_alpha)

        metrics["opd/loss"] = loss.item()
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
    parser.add_argument("--ema_alpha", type=float, default=0.02)
    parser.add_argument("--is_clip", type=float, default=2.0)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--no_is_clip", action="store_true", help="disable IS-clip term and skip the old_action_logprob precompute pass entirely")
    parser.add_argument(
        "--vlm", action="store_true",
        help="load --model_name as a vision-language model (AutoProcessor + AutoModelForVision2Seq) instead of "
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
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.vlm:
        processor = AutoProcessor.from_pretrained(args.model_name)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model = AutoModelForVision2Seq.from_pretrained(args.model_name, torch_dtype=dtype)
        tokenizer = processor.tokenizer
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)

    dataset = CriticalStepDataset(args.data_path)

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
    )

    trainer = OPDTrainer(
        model, processor if args.vlm else tokenizer, ema_alpha=args.ema_alpha,
        is_clip=None if args.no_is_clip else args.is_clip, lr=args.lr,
    )

    if not args.no_is_clip:
        print(f"precomputing old_action_logprob from {args.model_name} for {len(dataset.records)} records...")
        trainer.precompute_old_log_probs(dataset, collator)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    os.makedirs(args.output_dir, exist_ok=True)
    # metrics only ever went to stdout before -- once that scrollback is gone
    # (e.g. a Slurm job's stdout capture expires) there was no record left of
    # the loss/distillation-signal/gradient-adjacent metrics a run produced.
    # metrics.jsonl gives every run (smoke test or real) a persistent,
    # inspectable record in --output_dir itself.
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    step = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for batch in loader:
            metrics = trainer.train_step(batch)
            print(f"epoch {epoch} step {step}: {metrics}")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": step, **metrics}) + "\n")
            step += 1
            if step >= args.max_steps:
                stop = True
                break

    trainer.model.save_pretrained(args.output_dir)
    if args.vlm:
        processor.save_pretrained(args.output_dir)
    else:
        tokenizer.save_pretrained(args.output_dir)
    print(f"training done, {step} steps over {min(epoch + 1, args.epochs)} epoch(s), checkpoint at {args.output_dir}")


if __name__ == "__main__":
    main()
