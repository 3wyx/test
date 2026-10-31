"""Correctness of all Triton kernels against plain PyTorch and bitsandbytes, forward and backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float().clone(), up.grad.float().clone()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 129, 1536)])
@pytest.mark.parametrize("reconstruct_x", [False, True])
def test_rmsnorm_matches_pytorch(dtype, shape, reconstruct_x):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = (
        (torch.ones(shape[-1], device="cuda", dtype=dtype) + 0.1 * torch.randn(shape[-1], device="cuda", dtype=dtype))
        .detach()
        .requires_grad_(False)
    )
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref = (x_f * torch.rsqrt(var + eps)) * w.float()
    ref.backward(grad.float())
    ref_dx = x.grad.float().clone()
    x.grad = None

    out = rmsnorm(x, w, eps, reconstruct_x=reconstruct_x)
    out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rope_matches_pytorch(dtype):
    from solution.rope import rope_qwen2

    torch.manual_seed(0)
    bsz, n_qh, n_kh, seq_len, head_dim = 4, 12, 2, 128, 128
    q = (
        torch.randn((bsz, seq_len, n_qh, head_dim), device="cuda", dtype=dtype)
        .transpose(1, 2)
        .detach()
        .requires_grad_(True)
    )
    k = (
        torch.randn((bsz, seq_len, n_kh, head_dim), device="cuda", dtype=dtype)
        .transpose(1, 2)
        .detach()
        .requires_grad_(True)
    )
    angles = torch.randn((bsz, seq_len, head_dim), device="cuda", dtype=torch.float32)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    dq = torch.randn_like(q)
    dk = torch.randn_like(k)

    def rotate_half(t):
        t1 = t[..., : t.shape[-1] // 2]
        t2 = t[..., t.shape[-1] // 2 :]
        return torch.cat((-t2, t1), dim=-1)

    c_u = cos.unsqueeze(1).float()
    s_u = sin.unsqueeze(1).float()
    ref_q = q.float() * c_u + rotate_half(q.float()) * s_u
    ref_k = k.float() * c_u + rotate_half(k.float()) * s_u
    (ref_q * dq.float()).sum().add_((ref_k * dk.float()).sum()).backward()
    ref_gq, ref_gk = q.grad.float().clone(), k.grad.float().clone()
    q.grad = k.grad = None

    out_q, out_k = rope_qwen2(q, k, cos, sin)
    ((out_q.float() * dq.float()).sum() + (out_k.float() * dk.float()).sum()).backward()

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out_q.float(), ref_q, rtol=tol, atol=tol)
    torch.testing.assert_close(out_k.float(), ref_k, rtol=tol, atol=tol)
    torch.testing.assert_close(q.grad.float(), ref_gq, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k.grad.float(), ref_gk, rtol=1e-2, atol=1e-2)


def test_chunked_head_loss_matches_pytorch_with_grad_scaler():
    from solution.cross_entropy import _run_triton_chunked_head_loss

    torch.manual_seed(0)
    bsz, seq_len, hidden_dim, vocab_size = 2, 64, 128, 4096
    h1 = torch.randn((bsz, seq_len, hidden_dim), device="cuda", dtype=torch.float16, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)
    head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda", dtype=torch.float16)
    head.weight.requires_grad_(False)

    labels = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda", dtype=torch.long)
    labels[:, :20] = -100
    labels[1, 45:] = -100
    attn_mask = torch.ones((bsz, seq_len), device="cuda", dtype=torch.long)
    attn_mask[1, 45:] = 0

    scale = 65536.0
    shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    logits_ref = head(h1).float()
    loss_ref = torch.nn.functional.cross_entropy(
        logits_ref.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100, reduction="mean"
    )
    (loss_ref * scale).backward()

    loss_out, step = _run_triton_chunked_head_loss(h2, head, labels, attention_mask=attn_mask)
    (loss_out * scale).backward()

    assert torch.isfinite(h2.grad).all()
    torch.testing.assert_close(loss_out.float(), loss_ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(h2.grad.float(), h1.grad.float(), rtol=5e-3, atol=5e-3)

    ref_probs = torch.log_softmax(logits_ref.view(-1, vocab_size), dim=-1)
    ref_entropy = -(ref_probs.exp() * ref_probs).sum(-1).view(bsz, seq_len)
    valid = attn_mask.bool()
    torch.testing.assert_close(step.token_entropy[valid], ref_entropy[valid], rtol=5e-3, atol=5e-3)
    ref_correct = ((logits_ref.view(-1, vocab_size).argmax(-1) == shift_labels.view(-1)) & (shift_labels.view(-1) != -100)).sum()
    assert int(step.correct.item()) == int(ref_correct.item())


def test_nf4_dequant_and_matmul_matches_bitsandbytes():
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    from solution.nf4 import TritonMatMul4Bit, triton_dequantize_nf4

    torch.manual_seed(0)
    w = torch.randn((1536, 1536), device="cuda", dtype=torch.float16)
    quant_w, quant_state = bnb_F.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
    ref = bnb_F.dequantize_4bit(quant_w, quant_state)
    out = triton_dequantize_nf4(quant_w, quant_state)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    x1 = torch.randn((2, 16, 1536), device="cuda", dtype=torch.float16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    dy = torch.randn((2, 16, 1536), device="cuda", dtype=torch.float16)

    y_ref = bnb.matmul_4bit(x1, quant_w.t(), quant_state=quant_state)
    y_ref.backward(dy)

    y_out = TritonMatMul4Bit.apply(x2, quant_w.t(), None, None, quant_state)
    y_out.backward(dy)

    torch.testing.assert_close(y_out, y_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x2.grad, x1.grad, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("recompute", [False, True])
def test_fused_norm_mlp_matches_pytorch(recompute):
    import bitsandbytes.functional as bnb_F
    from solution.swiglu import _FusedNormMLP

    torch.manual_seed(0)
    hidden, inter, rows = 256, 512, 96

    def quant(shape):
        w = 0.05 * torch.randn(shape, device="cuda", dtype=torch.float16)
        qw, qs = bnb_F.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
        return qw, qs, bnb_F.dequantize_4bit(qw, qs)

    gate_q, gate_s, gate_w = quant((inter, hidden))
    up_q, up_s, up_w = quant((inter, hidden))
    down_q, down_s, down_w = quant((hidden, inter))
    norm_w = (1 + 0.1 * torch.randn(hidden, device="cuda", dtype=torch.float16)).detach()

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)

    x_ref = x.detach().clone().requires_grad_(True)
    xf = x_ref.float()
    x_norm = ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)) * norm_w.float()).half()
    act = (torch.nn.functional.silu((x_norm @ gate_w.t()).float()) * (x_norm @ up_w.t()).float()).half()
    ref = act @ down_w.t()
    ref.backward(grad)

    out = _FusedNormMLP.apply(
        x, norm_w, 1e-6, gate_q, gate_s, None, up_q, up_s, None, down_q, down_s, None, recompute
    )
    out.backward(grad)

    assert (out.float() - ref.float()).abs().max() <= 0.03 * ref.float().abs().max()
    assert (x.grad.float() - x_ref.grad.float()).abs().max() <= 0.03 * x_ref.grad.float().abs().max()