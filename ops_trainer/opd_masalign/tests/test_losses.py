"""torch-dependent tests for the OPD loss and EMA update.

Run with: python -m pytest opd_masalign/tests/test_losses.py -v
"""
import os

import pytest
import torch
import torch.nn as nn

from opd_masalign.losses import compute_opd_loss, ema_update


def test_loss_is_zero_when_student_equals_teacher():
    B, T = 4, 6
    log_probs = torch.randn(B, T) - 1.0
    mask = torch.ones(B, T)

    loss, metrics = compute_opd_loss(
        student_log_probs=log_probs,
        teacher_log_probs=log_probs.clone(),
        response_mask=mask,
        old_log_probs=log_probs.clone(),
        is_clip=2.0,
    )
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_loss_gradient_pushes_student_toward_teacher():
    torch.manual_seed(0)
    B, T, V = 2, 5, 20
    mask = torch.ones(B, T)

    logits = torch.zeros(B, T, V, requires_grad=True)
    actions = torch.randint(0, V, (B, T))
    log_probs_all = torch.log_softmax(logits, dim=-1)
    student_log_probs = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    # teacher strongly prefers the taken action (i.e. teacher_log_probs are
    # less negative than the student's uniform-init log_probs) -> reverse-KL
    # gradient should push student_log_probs UP (student more confident).
    teacher_log_probs = student_log_probs.detach() + 2.0

    loss, _ = compute_opd_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=mask,
        old_log_probs=student_log_probs.detach(),
        is_clip=2.0,
    )
    loss.backward()

    # log_ratio = student - teacher = -2.0 (negative) -> per_token_loss =
    # -2.0 * student_log_probs -> d(loss)/d(student_log_probs) = -2.0 < 0,
    # so gradient descent increases student_log_probs, i.e. logits.grad at
    # the taken action should be negative (since increasing that logit
    # increases student_log_probs and decreases loss).
    taken_grad = logits.grad.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    assert (taken_grad < 0).all()


def test_seq_mean_and_token_mean_agree_when_response_lengths_are_equal():
    """With no per-record weighting, seq-mean-token-mean and token-mean only
    differ when sequences in the batch have different valid-response
    lengths (see compute_opd_loss's docstring) -- confirm they agree when
    lengths are equal, and diverge when they aren't.
    """
    # per-token loss = (student-teacher).detach() * student -> row 0 is
    # 2*(-1)=-2 everywhere, row 1 is 2*(-2)=-4 everywhere: genuinely
    # different per-row values, so seq-mean (equal per SEQUENCE) and
    # token-mean (equal per TOKEN) can actually diverge once row lengths differ.
    student = torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-3.0, -3.0, -3.0], [-4.0, -4.0, -4.0]])
    old = student.detach().clone()
    equal_mask = torch.ones(2, 3)

    loss_seq, _ = compute_opd_loss(student, teacher, equal_mask, old, is_clip=2.0, loss_agg_mode="seq-mean-token-mean")
    loss_tok, _ = compute_opd_loss(student, teacher, equal_mask, old, is_clip=2.0, loss_agg_mode="token-mean")
    assert torch.allclose(loss_seq, loss_tok, atol=1e-6)

    ragged_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])  # row 1 has only 1 valid token
    loss_seq_r, _ = compute_opd_loss(student, teacher, ragged_mask, old, is_clip=2.0, loss_agg_mode="seq-mean-token-mean")
    loss_tok_r, _ = compute_opd_loss(student, teacher, ragged_mask, old, is_clip=2.0, loss_agg_mode="token-mean")
    assert not torch.allclose(loss_seq_r, loss_tok_r, atol=1e-6)


def test_ema_update_moves_teacher_toward_student():
    student = torch.nn.Linear(4, 4, bias=False)
    teacher = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        student.weight.fill_(1.0)
        teacher.weight.fill_(0.0)

    ema_update(student, teacher, alpha=0.1)
    assert torch.allclose(teacher.weight, torch.full((4, 4), 0.1))

    ema_update(student, teacher, alpha=0.1)
    assert torch.allclose(teacher.weight, torch.full((4, 4), 0.19), atol=1e-6)


def test_ema_update_matches_plain_path_under_fsdp2():
    """Only a world_size=1 check (this CI/dev box has no multi-GPU hardware),
    but it exercises the REAL gather/lerp/scatter code path in
    losses.py::ema_update for FSDP2-wrapped modules -- not a hand-rolled
    reimplementation -- and asserts it reproduces the exact same numbers as
    the plain-path test above. Multi-rank sharding correctness is NOT
    covered by this test; see ema_update's docstring.
    """
    try:
        import torch.distributed as dist
        from torch.distributed._composable.fsdp import fully_shard
    except ImportError:
        pytest.skip("FSDP2 (fully_shard) not available in this torch build")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        student = nn.Linear(4, 4, bias=False)
        teacher = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            student.weight.fill_(1.0)
            teacher.weight.fill_(0.0)
        fully_shard(student)
        fully_shard(teacher)

        ema_update(student, teacher, alpha=0.1)
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        full = get_model_state_dict(teacher, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        assert torch.allclose(full["weight"], torch.full((4, 4), 0.1), atol=1e-6)

        ema_update(student, teacher, alpha=0.1)
        full = get_model_state_dict(teacher, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        assert torch.allclose(full["weight"], torch.full((4, 4), 0.19), atol=1e-6)
    finally:
        dist.destroy_process_group()
