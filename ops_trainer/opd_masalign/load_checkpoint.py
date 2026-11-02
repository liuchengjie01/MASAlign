"""加载 opd_trainer.py 存出来的checkpoint，跑一遍推理做sanity check。

checkpoint目录（无论是`--output_dir`本身，还是`--save_every`产出的
`checkpoint-{step}/`）都是标准的`save_pretrained`输出，本可以直接用
`AutoModelForCausalLM.from_pretrained`/`AutoModelForImageTextToText.from_pretrained`
加载去做别的事——这个脚本只是把"加载 + 挑几条held-out数据验证输出是否合理"这
一步先跑通，省得每次都要重新拼这几行。是纯文本还是VLM checkpoint通过目录里
有没有`preprocessor_config.json`或`processor_config.json`（新版transformers用后者）自动判断，不需要额外传`--vlm`。

用法：
    # 只加载，确认checkpoint能读起来（不跑推理）
    python -m opd_masalign.load_checkpoint --checkpoint_dir /tmp/opd_smoke

    # 加载后，对data_path里的记录做生成，和ground-truth action_taken对比
    python -m opd_masalign.load_checkpoint --checkpoint_dir /tmp/opd_smoke \
        --data_path data/critical_steps_smoke.jsonl --num_records 3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .opd_trainer import CriticalStepDataset, _extract_images_and_normalize
from .self_teacher import build_teacher_messages


def _is_vlm_checkpoint(checkpoint_dir: str) -> bool:
    # Older transformers writes the image processor to preprocessor_config.json; newer versions fold it
    # into processor_config.json instead (seen on a real AutoGLM-Phone-9B checkpoint saved by
    # opd_trainer.py -- only processor_config.json was written, so checking for the old name alone
    # misclassified it as text-only and failed with "Unrecognized configuration class Glm4vConfig").
    return any(
        os.path.exists(os.path.join(checkpoint_dir, f)) for f in ("preprocessor_config.json", "processor_config.json")
    )


def load_model(checkpoint_dir: str, dtype: torch.dtype, device: torch.device, disable_image_splitting: bool = False):
    # device_map streams weights straight into GPU memory as each shard is read off disk, instead of
    # materializing the full model in CPU RAM first and only THEN .to(device)-copying it -- for a real
    # multi-billion-parameter checkpoint (e.g. AutoGLM-Phone-9B) the naive path transiently needs BOTH
    # a full CPU-RAM copy and a full GPU copy at once, which can OOM the HOST (not the GPU) well before
    # generate() ever runs -- see opd_trainer.py's load_model_and_tokenizer for the same reasoning.
    device_map = {"": device.index} if device.type == "cuda" else None
    is_vlm = _is_vlm_checkpoint(checkpoint_dir)
    if is_vlm:
        processor = AutoProcessor.from_pretrained(checkpoint_dir)
        if disable_image_splitting and hasattr(processor, "image_processor"):
            # matches opd_trainer.py's Collator: without this, the processor's default high-res tiling
            # can split ONE phone screenshot into many tiles, multiplying vision-token count (and thus
            # KV-cache memory for that record) well beyond what --image_max_side alone controls for --
            # and tile count varies per image resolution, so this is also a source of the per-record
            # size variance described in generate_for_record's docstring.
            processor.image_processor.do_image_splitting = False
        model = AutoModelForImageTextToText.from_pretrained(checkpoint_dir, torch_dtype=dtype, device_map=device_map)
        return model.eval(), processor.tokenizer, processor, is_vlm
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=dtype, device_map=device_map)
        return model.eval(), tokenizer, None, is_vlm


@torch.no_grad()
def generate_for_record(model, tokenizer, processor, is_vlm: bool, record: dict, max_new_tokens: int,
                         image_max_side: int | None, use_teacher_reprompt: bool, device: torch.device,
                         think_prefill: bool = False) -> tuple[str, int]:
    """Encode record's observation_context (or, with use_teacher_reprompt, the
    same EMA-teacher hindsight reprompt opd_trainer.py trains the student
    against) with add_generation_prompt=True and greedily generate the
    checkpoint's own predicted next action -- no ground-truth action_taken
    tokens are fed in, unlike training's teacher-forced forward.

    Returns (prediction, prompt_len) -- prompt_len is surfaced to the caller
    so a run that OOMs partway through a data_path can be correlated against
    per-record prompt/image size (records are NOT uniform size: deeper
    conversation history or more images = a longer prompt = more activation/
    KV-cache memory for THAT record's generate() call alone, independent of
    anything from earlier records).
    """
    ctx = record["observation_context"]
    messages = build_teacher_messages(record) if use_teacher_reprompt else ctx["messages"]

    if is_vlm:
        normalized_messages, images = _extract_images_and_normalize(messages)
        if image_max_side:
            from .opd_trainer import _cap_longer_side
            images = [_cap_longer_side(img, image_max_side) for img in images]
        prompt_text = processor.apply_chat_template(normalized_messages, tokenize=False, add_generation_prompt=True)
        if think_prefill:
            prompt_text += "<think>"
        inputs = processor(text=prompt_text, images=images or None, return_tensors="pt").to(device)
    else:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if think_prefill:
            prompt_text += "<think>"
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

    prompt_len = inputs["input_ids"].shape[1]
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)

    # each record has a DIFFERENT prompt shape (context depth, image count/resolution for VLM) --
    # PyTorch's CUDA caching allocator only reuses a freed block if a later allocation fits it, so a
    # loop over varying shapes can ratchet up RESERVED (not just allocated) GPU memory call over call
    # even though every individual call's peak usage is unremarkable. del + empty_cache() before
    # returning keeps this loop's footprint at "current record's own peak", not "worst record so far
    # plus every shape-mismatch fragment before it".
    del inputs, out_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, prompt_len


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", required=True, help="opd_trainer.py --output_dir 或 checkpoint-{step}/ 目录")
    p.add_argument("--data_path", default=None, help="给了才跑推理；不给就只加载做sanity check")
    p.add_argument("--num_records", type=int, default=3, help="从data_path开头取几条跑推理对比")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--image_max_side", type=int, default=None, help="VLM checkpoint且图片较大时用，和训练时保持一致")
    p.add_argument(
        "--disable_image_splitting", action="store_true",
        help="VLM checkpoint用，关掉processor默认的高分辨率tiling——同一张手机截图不tiling能显著减少视觉token数/"
        "KV cache占用，见opd_trainer.py README里同名参数的说明；真实高分辨率截图不加这个很容易是OOM的主因之一",
    )
    p.add_argument(
        "--use_teacher_reprompt", action="store_true",
        help="用EMA-teacher的事后复盘reprompt（self_teacher.build_teacher_messages）而不是原始observation_context "
        "去生成——想单独看teacher-forcing那条路径训得怎么样时用这个，默认关闭（用原始context，即部署时真实看到的输入）",
    )
    p.add_argument(
        "--think_prefill", action="store_true",
        help="生成前在assistant回合预填 `<think>`，让模型保持训练时的 <think>..</think><answer>..</answer> 结构"
        "（和 opd_trainer.py --think_prefill 、client.ModelConfig.think_prefill 一致）；AutoGLM-Phone 这类带"
        "<think>/<answer>特殊token的checkpoint用，其它模型不要加",
    )
    p.add_argument(
        "--dtype", choices=["bf16", "fp32"], default=None,
        help="不传时按device自动选：GPU上用bf16（和opd_trainer.py训练时的默认一致，权重+激活值/KV cache都是fp32的一半，"
        "真实大模型上这个差别足以决定会不会OOM），CPU上用fp32（bf16在没有原生加速的CPU上通常更慢）",
    )
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    t0 = time.time()
    model, tokenizer, processor, is_vlm = load_model(
        args.checkpoint_dir, dtype, device, disable_image_splitting=args.disable_image_splitting
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"loaded {'VLM' if is_vlm else 'text-only'} checkpoint from {args.checkpoint_dir} "
        f"({n_params / 1e6:.1f}M params, {dtype}, {device}) in {time.time() - t0:.1f}s"
    )

    if args.data_path is None:
        return

    dataset = CriticalStepDataset(args.data_path)
    n = min(args.num_records, len(dataset))
    print(f"running generation on the first {n}/{len(dataset)} record(s) from {args.data_path}...")
    for i in range(n):
        record = dataset.records[i]
        ground_truth = record["action_taken"]["raw_text"]
        t0 = time.time()
        prediction, prompt_len = generate_for_record(
            model, tokenizer, processor, is_vlm, record, args.max_new_tokens,
            args.image_max_side, args.use_teacher_reprompt, device, args.think_prefill,
        )
        mem = ""
        if device.type == "cuda":
            # allocated = tensors actually live right now; reserved = what the caching allocator has
            # claimed from CUDA and may not give back. allocated staying flat while reserved climbs
            # record over record is the fragmentation signature (see generate_for_record's docstring);
            # both climbing together instead means the records themselves are getting bigger.
            mem = (
                f", gpu_allocated={torch.cuda.memory_allocated(device) / 1e9:.2f}GB"
                f", gpu_reserved={torch.cuda.memory_reserved(device) / 1e9:.2f}GB"
                f", gpu_max_allocated={torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB"
            )
        print(f"\n--- record {i}, prompt_len={prompt_len} tokens ({time.time() - t0:.1f}s{mem}) ---")
        print(f"ground truth : {ground_truth!r}")
        print(f"prediction   : {prediction!r}")


if __name__ == "__main__":
    main()
