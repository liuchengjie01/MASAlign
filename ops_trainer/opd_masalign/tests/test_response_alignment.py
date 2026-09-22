"""Tests for opd_trainer.py::_align_student_teacher, which extracts each
row's own valid action-token log-probs from student/teacher (which live at
different absolute offsets -- see that function's docstring) into aligned
(B, T) tensors for compute_opd_loss.

The function replaced an earlier _gather_last_k-based version that CROPPED
every row in a batch down to the batch's SHORTEST row's length, silently
discarding real tokens -- and their gradient contribution -- from every
longer row. These tests exist specifically to pin down that the fix (pad
shorter rows instead of cropping longer ones) actually holds, plus the
narrower per-row alignment behavior it's built on.

Run with: python -m pytest opd_masalign/tests/test_response_alignment.py -v
"""
import torch

from opd_masalign.opd_trainer import _align_student_teacher


def test_equal_length_rows_all_positions_kept():
    student_log_probs = torch.tensor([
        [0.0, 0.0, 10.0, 20.0],
        [0.0, 0.0, 30.0, 40.0],
    ])
    student_mask = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])
    teacher_log_probs = torch.tensor([
        [0.0, 0.0, 0.0, 11.0, 21.0],
        [0.0, 0.0, 0.0, 31.0, 41.0],
    ])
    teacher_mask = torch.tensor([
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ])

    student_resp, teacher_resp, align_mask, old_resp = _align_student_teacher(
        student_log_probs, student_mask, teacher_log_probs, teacher_mask
    )

    assert align_mask.tolist() == [[1, 1], [1, 1]]
    assert student_resp.tolist() == [[10.0, 20.0], [30.0, 40.0]]
    assert teacher_resp.tolist() == [[11.0, 21.0], [31.0, 41.0]]
    assert old_resp is None


def test_pads_shorter_rows_instead_of_cropping_longer_ones():
    # row 0: only 1 valid response token; row 1: 3 valid response tokens
    student_log_probs = torch.tensor([
        [0.0, 0.0, 0.0, 9.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    student_mask = torch.tensor([
        [0, 0, 0, 1],
        [0, 1, 1, 1],
    ])
    teacher_log_probs = torch.tensor([
        [0.0, 0.0, 0.0, 90.0],
        [0.0, 10.0, 20.0, 30.0],
    ])
    teacher_mask = torch.tensor([
        [0, 0, 0, 1],
        [0, 1, 1, 1],
    ])

    student_resp, teacher_resp, align_mask, _ = _align_student_teacher(
        student_log_probs, student_mask, teacher_log_probs, teacher_mask
    )

    # T must be the batch's LONGEST per-row length (3), not the shortest (1)
    assert student_resp.shape == (2, 3)
    # row 1's full 3 real tokens must all survive -- the old crop-to-batch-min
    # behavior would have kept only its LAST 1 token, discarding [1.0, 2.0]
    assert align_mask[1].tolist() == [1, 1, 1]
    assert student_resp[1].tolist() == [1.0, 2.0, 3.0]
    assert teacher_resp[1].tolist() == [10.0, 20.0, 30.0]
    # row 0 only has 1 real token; the rest is padding, correctly masked out
    assert align_mask[0].tolist() == [1, 0, 0]
    assert student_resp[0, 0].item() == 9.0


def test_row_with_student_teacher_length_mismatch_takes_the_min_and_right_aligns():
    # this row's student side has 4 valid tokens, teacher side only 3 -- the
    # per-row min must be taken and RIGHT-aligned (drop the EARLIEST student
    # token, not an arbitrary one), matching "same a_u tokens, tail-anchored"
    student_log_probs = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    student_mask = torch.tensor([[1, 1, 1, 1]])
    teacher_log_probs = torch.tensor([[0.0, 100.0, 200.0, 300.0]])
    teacher_mask = torch.tensor([[0, 1, 1, 1]])

    student_resp, teacher_resp, align_mask, _ = _align_student_teacher(
        student_log_probs, student_mask, teacher_log_probs, teacher_mask
    )

    assert align_mask.tolist() == [[1, 1, 1]]
    assert student_resp.tolist() == [[20.0, 30.0, 40.0]]  # first token (10.0) dropped
    assert teacher_resp.tolist() == [[100.0, 200.0, 300.0]]


def test_old_log_probs_aligned_to_the_same_indices_as_student():
    student_log_probs = torch.tensor([[10.0, 20.0, 30.0]])
    student_mask = torch.tensor([[1, 1, 1]])
    teacher_log_probs = torch.tensor([[100.0, 200.0, 300.0]])
    teacher_mask = torch.tensor([[1, 1, 1]])
    old_log_probs = torch.tensor([[1.0, 2.0, 3.0]])  # same positions as student

    _, _, _, old_resp = _align_student_teacher(
        student_log_probs, student_mask, teacher_log_probs, teacher_mask, old_log_probs
    )
    assert old_resp.tolist() == [[1.0, 2.0, 3.0]]


def test_zero_length_row_does_not_crash():
    student_log_probs = torch.tensor([
        [0.0, 0.0],
        [5.0, 6.0],
    ])
    student_mask = torch.tensor([
        [0, 0],
        [1, 1],
    ])
    teacher_log_probs = torch.zeros(2, 2)
    teacher_mask = torch.tensor([
        [0, 0],
        [1, 1],
    ])

    student_resp, teacher_resp, align_mask, _ = _align_student_teacher(
        student_log_probs, student_mask, teacher_log_probs, teacher_mask
    )
    assert align_mask[0].sum().item() == 0
    assert align_mask[1].tolist() == [1, 1]


def test_padding_preserves_gradient_at_every_real_position():
    student_log_probs = torch.tensor([[0.0, 1.0, 2.0, 3.0]], requires_grad=True)
    student_mask = torch.tensor([[0, 1, 1, 1]])
    teacher_log_probs = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
    teacher_mask = torch.tensor([[0, 1, 1, 1]])

    student_resp, _, _, _ = _align_student_teacher(
        student_log_probs, student_mask, teacher_log_probs, teacher_mask
    )
    student_resp.sum().backward()

    # every real (mask==1) source position must receive a nonzero gradient --
    # a crop-to-batch-min bug would zero out earlier positions' gradient
    # whenever another row in the same batch was shorter
    assert student_log_probs.grad[0].tolist() == [0.0, 1.0, 1.0, 1.0]
