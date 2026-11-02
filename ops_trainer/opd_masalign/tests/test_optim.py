import torch

from opd_masalign.optim import AdamWSR, lerp_into_, stochastic_round_to_bf16


def test_stochastic_rounding_is_unbiased():
    torch.manual_seed(0)
    x = torch.full((200000,), 0.01 + 1e-5, dtype=torch.float32)  # 1e-5 is far below the bf16 spacing at 0.01
    mean = stochastic_round_to_bf16(x).float().mean().item()
    assert abs(mean - x[0].item()) < 5e-7
    assert torch.round(x.to(torch.bfloat16).float() - x[0]).abs().max() == 0  # nearest rounding would be stuck


def test_adamw_sr_moves_bf16_weights_that_round_to_nearest_would_not():
    torch.manual_seed(0)
    n = 50000
    w0 = torch.full((n,), 0.01, dtype=torch.bfloat16)
    p = torch.nn.Parameter(w0.clone())
    ref = torch.nn.Parameter(w0.clone())
    opt, opt_ref = AdamWSR([p], lr=1e-5, weight_decay=0.0), torch.optim.AdamW([ref], lr=1e-5, weight_decay=0.0)
    for _ in range(100):
        p.grad = torch.ones(n, dtype=torch.bfloat16)
        ref.grad = torch.ones(n, dtype=torch.bfloat16)
        opt.step()
        opt_ref.step()
    moved = (w0.float() - p.detach().float()).mean().item()  # ideal: ~ lr * steps = 1e-3
    assert 0.7e-3 < moved < 1.3e-3
    assert (w0.float() - ref.detach().float()).mean().item() < 0.5 * moved  # plain bf16 AdamW loses most of it


def test_lerp_into_matches_fp32_in_expectation():
    torch.manual_seed(0)
    t = torch.full((100000,), 0.02, dtype=torch.bfloat16)
    s = torch.full((100000,), 0.02 + 1e-3, dtype=torch.bfloat16)
    expected = 0.98 * t.float()[0] + 0.02 * s.float()[0]
    lerp_into_(t, s, 0.02)
    assert abs(t.float().mean().item() - expected.item()) < 2e-6
