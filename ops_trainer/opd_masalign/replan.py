"""Reflect-and-replan: have the EMA teacher rewrite the GUI-Owl *planner's* "### Plan ###"
section under the same hindsight reprompt, instead of only reprompting the executor (see
guiowl_format.py / reprompt_variants.py). Motivation: inspecting real rollouts
(data/luis/gui-owl-7b.zip) shows the planner sub-call runs on the SAME screenshot, one call
before the executor, and already commits to the attacked action there (e.g. "the next logical
step is to confirm this action by clicking 'Fill'") -- that text is then baked verbatim into the
executor's "### Overall Plan ###" / "### Current Subgoal ###" sections. Reprompting only the
executor means arguing against a plan the model itself just wrote moments ago; reprompting the
planner lets the correction happen upstream, before it's cast as an "already decided" plan.

Kept separate from guiowl_format.py (different sub-call, different output grammar -- the planner
emits "### Thought ### / [### Historical Operations ###] / ### Plan ###" with no further
sections, confirmed against real planning_response values, so generation just runs to EOS, no
stop_strings needed).
"""
from __future__ import annotations

import copy
import re

import torch

THOUGHT_HEADER = "### Thought ###"
PLAN_HEADER = "### Plan ###"


def extract_plan_text(planning_response: str) -> str:
    """Pull the numbered plan out of a planner turn's output; '' if malformed."""
    if PLAN_HEADER not in planning_response:
        return ""
    return planning_response.split(PLAN_HEADER, 1)[1].strip()


@torch.no_grad()
def generate_teacher_replan(model, collator, messages: list[dict], max_new_tokens: int, temperature: float,
                            top_p: float, synced_gpus: bool = False) -> tuple[str, str, int]:
    """Teacher writes its own ### Thought ### / ### Plan ### given `messages` (planning_messages +
    hindsight reprompt). Returns (new_plan_text, raw_generated_text, num_tokens).
    """
    from opd_masalign.opd_trainer import _encode_context_and_action_vlm

    tok = collator._tok
    assert collator.is_vlm, "replan.generate_teacher_replan only implements the VLM path"
    ids, _, vk = _encode_context_and_action_vlm(
        collator.processor, messages, "", collator.max_len, collator.image_max_side, THOUGHT_HEADER + "\n"
    )
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    vk = {k: v.to(device) for k, v in vk.items()}

    out = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **vk,
        max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None, top_p=top_p if temperature > 0 else None,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        synced_gpus=synced_gpus, use_cache=True,
    )
    new_ids = out[0, input_ids.shape[1]:].tolist()
    text = THOUGHT_HEADER + "\n" + tok.decode(new_ids, skip_special_tokens=True).strip()
    return extract_plan_text(text), text, len(new_ids)


def substitute_plan(action_messages: list[dict], new_plan_text: str) -> list[dict]:
    """Returns a deep copy of `action_messages` with the "### Overall Plan ###" and
    "### Current Subgoal ###" section bodies both replaced by `new_plan_text` (same text in
    both -- the real harness derives Current Subgoal from Overall Plan by some chunking rule
    that isn't reproduced here; using the same full text in both is good enough to test whether
    replanning upstream changes the executor's behavior, without replicating that chunking).
    Everything else (screenshot, guidelines, atomic-action spec, latest action history) is kept
    byte-identical.
    """
    messages = copy.deepcopy(action_messages)
    for m in messages:
        content = m.get("content")
        blocks = content if isinstance(content, list) else None
        if blocks is None:
            continue
        for b in blocks:
            if b.get("type") != "text" or "### Overall Plan ###" not in b.get("text", ""):
                continue
            text = b["text"]
            text = re.sub(
                r"(### Overall Plan ###\n).*?(\n\n### Current Subgoal ###\n).*?(\n\n###)",
                lambda m: m.group(1) + new_plan_text + m.group(2) + new_plan_text + m.group(3),
                text, count=1, flags=re.S,
            )
            b["text"] = text
    return messages
