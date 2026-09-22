"""Manual multi-PROCESS (world_size > 1) check for losses.py::ema_update's
FSDP2 path. NOT auto-run by pytest -- pytest runs single-process, so it
can't exercise real parameter sharding; test_losses.py's
test_ema_update_matches_plain_path_under_fsdp2 only proves the code path
doesn't crash at world_size=1, which is a degenerate case (nothing is
actually split across ranks there).

Run with 8 simulated "ranks" as 8 CPU processes + the gloo backend (the
real 8xA100 training job would use --nproc_per_node=8 with the SAME command
and losses.py needs no changes -- only the backend init in a real trainer
would be "nccl" instead of gloo, which is orthogonal to this file):

    cd /sfs/ceph/standard/AikyamLab/yuyang/MASAlign
    source opd_masalign/.venv/bin/activate
    torchrun --standalone --nproc_per_node=8 -m opd_masalign.tests.manual_multi_rank_ema_check

Checks two things against a REAL 8-way-sharded FSDP2 module:
  1. losses.py::ema_update's gather/lerp/scatter implementation produces
     the exact expected EMA values under real sharding (not just at
     world_size=1).
  2. Whether a NAIVE per-parameter lerp directly on the sharded DTensor
     parameters -- no gather at all -- ALSO produces the correct result.
     If it does, ema_update's FSDP2 path is paying for a full gather +
     broadcast on every call that isn't actually necessary, and should be
     simplified to the naive loop (much cheaper: zero communication). If it
     doesn't, the gather-based version is load-bearing and must stay.
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(128, 128, bias=False)
        self.b = nn.Linear(128, 128, bias=False)


def main():
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    torch.manual_seed(0)

    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from opd_masalign.losses import ema_update

    def make_pair(student_val, teacher_val):
        student, teacher = Toy().to(device), Toy().to(device)
        with torch.no_grad():
            for p in student.parameters():
                p.fill_(student_val)
            for p in teacher.parameters():
                p.fill_(teacher_val)
        for m in (student, teacher):
            fully_shard(m.a)
            fully_shard(m.b)
            fully_shard(m)
        return student, teacher

    def full_state(model):
        # NOTE: no cpu_offload here (unlike losses.py's internal helper) --
        # per torch's own StateDictOptions docstring, cpu_offload=True +
        # full_state_dict=True is what makes this rank0-only (an OOM-safety
        # default); this is a tiny toy model purely for verification, so we
        # deliberately skip that and let every rank read the real value
        # directly to assert against.
        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))

    # ---- sanity: confirm this is REAL world_size-way sharding, not degenerate ----
    probe, _ = make_pair(1.0, 0.0)
    local_numel = probe.a.weight.to_local().numel()
    global_numel = probe.a.weight.numel()
    print(f"[rank {rank}/{world_size}] a.weight global numel={global_numel} local numel={local_numel}", flush=True)
    assert local_numel < global_numel, (
        f"rank {rank}: local shard ({local_numel}) is not smaller than the global tensor "
        f"({global_numel}) -- sharding isn't actually happening, this test is degenerate"
    )

    # ---- check 1: the real ema_update() ----
    student, teacher = make_pair(1.0, 0.0)
    ema_update(student, teacher, alpha=0.1)
    got1 = full_state(teacher)["a.weight"][0, 0].item()
    if rank == 0:
        print(f"[rank 0] ema_update() call 1: teacher.a.weight[0,0] = {got1} (expect 0.1)")
    assert abs(got1 - 0.1) < 1e-6, f"rank {rank}: expected 0.1, got {got1}"

    ema_update(student, teacher, alpha=0.1)
    got2 = full_state(teacher)["a.weight"][0, 0].item()
    if rank == 0:
        print(f"[rank 0] ema_update() call 2: teacher.a.weight[0,0] = {got2} (expect 0.19)")
    assert abs(got2 - 0.19) < 1e-6, f"rank {rank}: expected 0.19, got {got2}"

    # ---- check 2: does a NAIVE no-gather loop also work under real sharding? ----
    student2, teacher2 = make_pair(1.0, 0.0)
    alpha = 0.1
    for p_s, p_t in zip(student2.parameters(), teacher2.parameters()):
        p_t.data.mul_(1 - alpha).add_(p_s.data, alpha=alpha)
    got_naive = full_state(teacher2)["a.weight"][0, 0].item()
    naive_correct = abs(got_naive - 0.1) < 1e-6
    if rank == 0:
        verdict = "MATCHES (gather is unnecessary overhead for FSDP2)" if naive_correct else "DIFFERS (gather is load-bearing)"
        print(f"[rank 0] naive no-gather loop: teacher.a.weight[0,0] = {got_naive} (expect 0.1) -> {verdict}")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
