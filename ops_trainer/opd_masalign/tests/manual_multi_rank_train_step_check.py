"""Manual multi-PROCESS (world_size > 1) check of the FULL OPD training
step -- forward, compute_opd_loss, backward, optimizer.step(), ema_update --
under REAL FSDP2 sharding. Complements manual_multi_rank_ema_check.py, which
only exercises ema_update() in isolation on toy nn.Linear layers; this one
runs an actual small HF causal LM through the exact same functions
opd_trainer.py's single-process OPDTrainer.train_step uses
(Collator, _sequence_log_probs, compute_opd_loss, ema_update), just with
FSDP2-wrapped student/teacher instead of opd_trainer.py's single-process
model.

NOT auto-run by pytest -- needs a real multi-process launch. Uses the same
real (text-only) smoke data and a small model to keep this focused on
proving the DISTRIBUTED plumbing works, not on validating on real data
again (that was already done single-process against the real extracted
data/critical_steps_autoglm.jsonl -- see slurm/refactor_smoke_test.slurm).

    cd /sfs/ceph/standard/AikyamLab/yuyang/MASAlign
    source opd_masalign/.venv_gpu/bin/activate   # needs a CUDA-enabled torch -- opd_masalign/.venv is CPU-only
    torchrun --standalone --nproc_per_node=<N_GPUS> \
        -m opd_masalign.tests.manual_multi_rank_train_step_check

Requires real CUDA GPUs (FSDP2's lazy init unconditionally creates
torch.cuda.Stream objects -- see opd_masalign/losses.py::ema_update's
docstring for the CPU-only failure this hit). On CPU, only
manual_multi_rank_ema_check.py (no forward pass, pure state_dict gather/
scatter) is exercisable.

Per-step metrics are logged (rank 0 only, matching opd_trainer.py's own
metrics.jsonl format) to:
    .claude_scratch/opd_fsdp2_8rank_train_step_check/metrics.jsonl
"""
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATA_PATH = "data/critical_steps_smoke.jsonl"
OUT_DIR = ".claude_scratch/opd_fsdp2_8rank_train_step_check"
N_STEPS = 4


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    torch.manual_seed(0)
    if rank == 0:
        print(f"use_cuda={use_cuda} device={device} world_size={world_size}", flush=True)

    from torch.distributed._composable.fsdp import fully_shard

    from opd_masalign.opd_trainer import Collator, CriticalStepDataset, _align_student_teacher, _sequence_log_probs
    from opd_masalign.losses import compute_opd_loss, ema_update

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # separate instances (not copy.deepcopy of an already-sharded module,
    # which FSDP2/DTensor doesn't support cleanly) -- both loaded from the
    # SAME checkpoint, matching OPDTrainer's "ema_model starts as an exact
    # copy of the student" invariant.
    # attn_implementation="eager": avoids SDPA's automatic backend selection,
    # which on some torch/transformers versions picks a triton-compiled
    # attention kernel -- this GPU node's system Python has no Python.h
    # (triton needs it to JIT its CUDA driver bridge), so anything that
    # invokes triton fails at first use. Eager attention needs no compiled
    # kernel and this check only cares about the FSDP2 plumbing, not attention speed.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    ema_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    for m in (model, ema_model):
        for layer in m.model.layers:  # per-transformer-block wrapping (how production FSDP2 usage actually shards)
            fully_shard(layer)
        fully_shard(m)

    first_param = next(model.parameters())
    local_numel, global_numel = first_param.to_local().numel(), first_param.numel()
    print(f"[rank {rank}/{world_size}] sharding sanity: first param local={local_numel} global={global_numel}", flush=True)
    assert local_numel < global_numel, "not actually sharded -- test is degenerate"

    dataset = CriticalStepDataset(DATA_PATH)
    collator = Collator(tokenizer=tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        metrics_path = os.path.join(OUT_DIR, "metrics.jsonl")
        open(metrics_path, "w").close()
    dist.barrier()
    metrics_path = os.path.join(OUT_DIR, "metrics.jsonl")

    step = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        model.train()
        student_log_probs, student_mask = _sequence_log_probs(
            model, batch["student_input_ids"], batch["student_attention_mask"], batch["student_response_mask"]
        )
        with torch.no_grad():
            teacher_log_probs, teacher_mask = _sequence_log_probs(
                ema_model, batch["teacher_input_ids"], batch["teacher_attention_mask"], batch["teacher_response_mask"]
            )
        student_resp, teacher_resp, align_mask, _ = _align_student_teacher(
            student_log_probs, student_mask, teacher_log_probs, teacher_mask
        )

        loss, metrics = compute_opd_loss(
            student_log_probs=student_resp,
            teacher_log_probs=teacher_resp,
            response_mask=align_mask,
            old_log_probs=student_resp.detach(),  # this check is about the distributed plumbing, not IS-clip
            is_clip=2.0,
        )

        optimizer.zero_grad()
        loss.backward()

        grad = first_param.grad
        grad_norm_local = grad.to_local().norm().item() if grad is not None else None

        optimizer.step()
        ema_update(model, ema_model, alpha=0.1)

        metrics["opd/loss"] = loss.item()
        metrics["opd/first_param_grad_norm_local_shard"] = grad_norm_local
        if rank == 0:
            print(f"step {step}: {metrics}", flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")
        step += 1
        if step >= N_STEPS:
            break

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"ALL_DONE, {step} real train steps (forward+loss+backward+optimizer.step()+ema_update) "
              f"under real {world_size}-way FSDP2 sharding. metrics at {metrics_path}")


if __name__ == "__main__":
    main()
