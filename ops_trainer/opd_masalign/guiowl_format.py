"""GUI-Owl/Mobile-Agent-V3 executor response format: no special think/answer tokens, just
plain-text section headers (`### Thought ### / ### Action ### / ### Description ###`). This
mirrors opd_trainer.py's AutoGLM-specific `_think_prefix`/`_format_think_prefix`/
`generate_teacher_thinking`, but using text markers (stop_strings) instead of a stop token id,
since this model has no `<think>`/`</think>` tokens to key off of. Kept separate from
opd_trainer.py (rather than parametrizing it) so the already-validated AutoGLM training path is
never touched by this.
"""
from __future__ import annotations

import torch

THOUGHT_HEADER = "### Thought ###"
ACTION_HEADER = "### Action ###"


def format_think_prefix(thinking: str) -> str:
    """Assistant-turn text between the generation prompt and the action, matching the real
    action_response format recorded in data/luis/gui-owl-7b.zip (see extract_critical_steps.py
    --format guiowl): `### Thought ###\\n{reasoning}\\n\\n### Action ###\\n{action}`.
    """
    return f"{THOUGHT_HEADER}\n{thinking}\n\n{ACTION_HEADER}\n"


@torch.no_grad()
def generate_teacher_thinking(model, collator, messages: list[dict], max_new_tokens: int, temperature: float,
                              top_p: float, synced_gpus: bool = False) -> tuple[str, int, bool, bool]:
    """Let `model` write its own reasoning for `messages` (observation_context + hindsight
    reprompt): the prompt ends with `### Thought ###\\n` and generation stops at the literal
    string `### Action ###` (HF's StopStringCriteria; this model has no special stop token for
    it, unlike AutoGLM's `</think>` -- see opd_trainer.py::generate_teacher_thinking).
    Returns (reasoning text, number of generated tokens, unclosed, skipped).
      unclosed: hit max_new_tokens without ever producing "### Action ###" (partial text kept).
      skipped:  degenerate case, reasoning came out empty.
    """
    from opd_masalign.opd_trainer import _encode_context_and_action_vlm

    tok = collator._tok
    assert collator.is_vlm, "guiowl_format.generate_teacher_thinking only implements the VLM path"
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
        stop_strings=[ACTION_HEADER], tokenizer=tok,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        synced_gpus=synced_gpus, use_cache=True,
    )
    new_ids = out[0, input_ids.shape[1]:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    closed = ACTION_HEADER in text
    if closed:
        text = text[: text.index(ACTION_HEADER)]
    text = text.strip()
    if not text:
        return "", len(new_ids), False, True
    return text, len(new_ids), not closed, False
