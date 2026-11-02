"""bf16-safe AdamW and bf16-safe EMA lerp, via stochastic rounding.

The student/EMA weights are stored in bf16 (the checkpoint dtype). bf16 has ~8 bits of mantissa, so a
per-step update smaller than half the spacing between neighbouring bf16 values is rounded away by
ordinary round-to-nearest: with AdamW at lr 1e-5 on weights of magnitude ~0.01 (spacing ~7.6e-5) about
three quarters of the entries never move (measured on a 5-step run of the real 9B). Stochastic rounding
rounds up or down with probability proportional to the remainder, so the EXPECTED update is exact, with
no extra memory (no fp32 master copy) and no change to how weights are loaded or sharded.
"""
from __future__ import annotations

import torch


def stochastic_round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16, rounding up/down at random so that E[result] == x."""
    assert x.dtype == torch.float32
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=x.device)
    bits = (bits + noise) & -65536  # keep the top 16 bits (sign, exponent, 7 mantissa bits)
    return bits.view(torch.float32).to(torch.bfloat16)


def _store(dst: torch.Tensor, value_fp32: torch.Tensor) -> None:
    dst.copy_(stochastic_round_to_bf16(value_fp32) if dst.dtype == torch.bfloat16 else value_fp32)


def _local(t: torch.Tensor) -> torch.Tensor:
    """The local shard of an FSDP2 DTensor, or the tensor itself."""
    return t.to_local() if hasattr(t, "to_local") else t


def lerp_into_(teacher: torch.Tensor, student: torch.Tensor, alpha: float) -> None:
    """teacher <- (1 - alpha) * teacher + alpha * student, computed in fp32 and stored with stochastic
    rounding when the tensors are bf16."""
    t32 = teacher.float().mul_(1 - alpha).add_(student.float(), alpha=alpha)
    _store(teacher, t32)


class AdamWSR(torch.optim.Optimizer):
    """AdamW whose update is computed in fp32 and written back to bf16 parameters (and bf16 moment
    estimates) with stochastic rounding. State dtype follows the parameter dtype, so memory is the same
    as torch.optim.AdamW on bf16 parameters. Works on FSDP2 DTensor parameters (local shards)."""

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                w, g = _local(p), _local(p.grad).float()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(w)
                    st["v"] = torch.zeros_like(w)
                st["step"] += 1
                m = st["m"].float().mul_(b1).add_(g, alpha=1 - b1)
                v = st["v"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                _store(st["m"], m)
                _store(st["v"], v)
                bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
                update = (m / bc1) / ((v / bc2).sqrt_().add_(eps))
                w32 = w.float().mul_(1 - lr * wd).add_(update, alpha=-lr)
                _store(w, w32)
