"""OPD loss (Eq. 9-11) and EMA teacher update (Section 7).

The reverse-KL trick in `compute_opd_loss` is adapted from SDPO's
`verl/trainer/ppo/core_algos.py::compute_self_distillation_loss` (the
non-full-logit, alpha=1 branch), which implements exactly the score-function
gradient identity proved in Proposition 1:

    A^OPD(u,a) = log( pi_theta(a|u) / q_theta(a|u,f_u) )     (stopgrad)
    grad_theta L_u = E_a[ A^OPD(u,a) * grad_theta log pi_theta(a|u) ]

realized per-token as `(student_logp - teacher_logp).detach() * student_logp`,
whose autograd gradient w.r.t. theta is exactly A^OPD * grad_theta log_prob.

This module is framework-agnostic (plain torch, no verl/TRL dependency) so
it can be unit-tested on CPU and reused from either the local TRL-based
smoke-test trainer or a future verl recipe integration.
"""
from __future__ import annotations

from typing import Optional

import torch

from .optim import lerp_into_


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1.0)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1.0)


def compute_opd_loss(
    student_log_probs: torch.Tensor,  # (B, T) log pi_theta(a_t | u_t), teacher-forced on a_u
    teacher_log_probs: torch.Tensor,  # (B, T) log q_theta_ema(a_t | u, f_u), teacher-forced on the SAME a_u tokens
    response_mask: torch.Tensor,  # (B, T) 1 for valid a_u tokens, 0 for padding/context
    old_log_probs: Optional[torch.Tensor] = None,  # (B, T), for IS-clip; None disables that term
    is_clip: Optional[float] = 2.0,
    loss_agg_mode: str = "seq-mean-token-mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """No per-record weighting: every critical step in the batch counts
    equally (this package doesn't compute a Delta(u)-style divergence signal
    -- critical steps are located and labeled upstream, handed off already
    filtered, so there's nothing here to weight them by).
    """
    metrics: dict[str, float] = {}

    log_ratio = student_log_probs - teacher_log_probs
    per_token_loss = log_ratio.detach() * student_log_probs

    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required when is_clip is set")
        negative_approx_kl = (student_log_probs - old_log_probs).detach().clamp(min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio
        metrics["opd/mean_is_ratio"] = masked_mean(ratio, response_mask).item()

    if loss_agg_mode == "seq-mean-token-mean":
        per_seq_loss = masked_mean(per_token_loss, response_mask, dim=-1)  # (B,)
        loss = per_seq_loss.mean()  # every sequence weighted equally, regardless of length
    elif loss_agg_mode == "token-mean":
        loss = masked_mean(per_token_loss, response_mask)  # every valid token weighted equally
    else:
        raise ValueError(f"unknown loss_agg_mode: {loss_agg_mode}")

    metrics["opd/mean_log_ratio"] = masked_mean(log_ratio, response_mask).item()
    return loss, metrics


def _fsdp_version(model: torch.nn.Module) -> int:
    """0 = plain/DDP module (every rank holds full parameter tensors),
    1 = FSDP1 (FullyShardedDataParallel), 2 = FSDP2 (FSDPModule / fully_shard).

    Mirrors SDPO's verl/utils/fsdp_utils.py::fsdp_version so this matches how
    the SDPO/verl recipe this loss is meant to be ported into (see this
    file's module docstring) actually wraps models -- without importing verl
    itself (this module stays torch-only, see module docstring). The
    try/except ladder for FSDP2's marker class mirrors verl's own: newer
    torch exposes it at torch.distributed.fsdp, older torch (e.g. 2.4, what
    this repo's CPU dev venv pins) only has it under the "composable" prefix.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        return 1
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        try:
            from torch.distributed._composable.fsdp import FSDPModule
        except ImportError:
            FSDPModule = None
    if FSDPModule is not None and isinstance(model, FSDPModule):
        return 2
    return 0


def _fsdp2_full_param_state_dict_rank0(model: torch.nn.Module) -> dict:
    """Gather model's FULL (unsharded) parameters, restricted to
    named_parameters() (buffers -- e.g. rotary embedding caches -- are left
    alone, matching the plain/DDP path below which also only touches
    .parameters(), never buffers).

    Per torch's own StateDictOptions docstring: "if cpu_offload [...] and
    full_state_dict [...] then only rank0 will get the state_dict and all
    other ranks will get empty state_dict" -- this is intentional (avoids
    every rank separately materializing the full unsharded model), NOT a
    bug to route around. Callers MUST treat the return value as
    rank0-only-meaningful and gate any use of its contents on
    dist.get_rank() == 0.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    full_state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    param_names = {name for name, _ in model.named_parameters()}
    return {k: v for k, v in full_state.items() if k in param_names}


def _fsdp2_load_param_state_dict(model: torch.nn.Module, param_state: dict) -> None:
    """Scatter a parameter-only state dict -- populated on rank 0, `{}` on
    every other rank -- back into an already-materialized (not meta-device)
    FSDP2 model: broadcasts rank 0's values out and re-shards on load.
    `strict=False` since param_state deliberately omits buffers (e.g. rotary
    embedding caches), which must be left untouched, not zeroed/errored on.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    set_model_state_dict(
        model, param_state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False),
    )


@torch.no_grad()
def ema_update(student: torch.nn.Module, teacher: torch.nn.Module, alpha: float) -> None:
    """theta_ema <- alpha * theta + (1 - alpha) * theta_ema.

    Plain/DDP path (student and teacher both unwrapped, or DDP-wrapped):
    every rank already holds the full tensor for every parameter, so a
    direct per-parameter lerp is correct and needs no communication -- this
    is the original, well-tested behavior (see test_losses.py) and stays
    the fast path.

    FSDP2 path: student and teacher are wrapped by the same code over the same
    mesh, so parameter i holds the same slice of the same tensor on every
    rank. Both modules are resharded first (the teacher only runs forward, so
    the root module's own parameters would otherwise still be all-gathered),
    then the local shards are lerped in place on the GPU. No communication,
    no full-state-dict gather. Shape mismatches raise instead of silently
    updating the wrong slice.

    Verified on 4 x A100 with the real AutoGLM-Phone-9B (5 training steps).

    FSDP1 (the older `FullyShardedDataParallel` wrapper) is deliberately NOT
    handled here: its rank0-broadcast load semantics differ from FSDP2's and
    guessing them without a way to test against a real process group risked
    shipping a subtly-wrong sync. Route FSDP1 users through
    verl/utils/fsdp_utils.py::get_fsdp_full_state_dict directly (they already
    have verl as a dependency at that point) or migrate to FSDP2.
    """
    s_ver = _fsdp_version(student)
    t_ver = _fsdp_version(teacher)

    if s_ver == 0 and t_ver == 0:
        for p_student, p_teacher in zip(student.parameters(), teacher.parameters()):
            lerp_into_(p_teacher.data, p_student.data, alpha)
        return

    if s_ver != t_ver:
        raise ValueError(
            f"ema_update: student and teacher must use identical wrapping "
            f"(student fsdp_version={s_ver}, teacher fsdp_version={t_ver})."
        )
    if s_ver == 1:
        raise NotImplementedError(
            "ema_update: FSDP1 (FullyShardedDataParallel) sharded sync is not implemented -- "
            "see this function's docstring. Use FSDP2 (fully_shard) or verl's own "
            "verl/utils/fsdp_utils.py::get_fsdp_full_state_dict directly."
        )

    # student and teacher are sharded by the same code over the same mesh, so parameter i holds the same
    # slice of the same tensor on every rank: lerp the local shards in place on the GPU, no communication.
    # (Gathering the full state dict to rank 0 and lerping on CPU also worked, but is far slower and,
    # with stochastic rounding, needs a random number per parameter on the host.)
    # The root module's own parameters (e.g. the vision position embedding) stay all-gathered after a
    # forward that has no backward (the teacher's), so to_local() would return the full tensor. Reshard first.
    for m in (student, teacher):
        if hasattr(m, "reshard"):
            m.reshard()
    student_params = dict(student.named_parameters())
    for name, p_teacher in teacher.named_parameters():
        p_student = student_params.get(name)
        if p_student is None:
            raise ValueError(f"ema_update: teacher parameter {name!r} has no counterpart in the student")
        t_local = p_teacher.to_local() if hasattr(p_teacher, "to_local") else p_teacher.data
        s_local = p_student.to_local() if hasattr(p_student, "to_local") else p_student.data
        if t_local.shape != s_local.shape:
            raise ValueError(
                f"ema_update: {name!r} is sharded differently in student {tuple(s_local.shape)} and "
                f"teacher {tuple(t_local.shape)}"
            )
        with torch.no_grad():
            lerp_into_(t_local, s_local, alpha)
