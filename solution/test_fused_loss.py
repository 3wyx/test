"""Our fused loss kernels against plain PyTorch: loss, gradient, entropy, accuracy, per-token loss.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(logits, labels, num_items):
    batch, length, vocab = logits.shape
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
    flat = logits.float().reshape(-1, vocab)
    token_ce = torch.nn.functional.cross_entropy(flat, shifted, ignore_index=-100,
                                                 reduction="none")
    denom = num_items if num_items is not None else (shifted != -100).sum()
    loss = token_ce.sum() / denom
    logp = torch.log_softmax(flat, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    correct = (flat.argmax(-1) == shifted) & (shifted != -100)
    return loss, token_ce.view(batch, length), entropy.view(batch, length), correct.view(batch, length)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 64, 151936), (3, 17, 1001)])
@pytest.mark.parametrize("num_items", [None, 37])
def test_fused_loss_matches_pytorch(dtype, shape, num_items):
    from solution import fused_loss

    torch.manual_seed(0)
    batch, length, vocab = shape
    logits = (3 * torch.randn(shape, device="cuda", dtype=torch.float32)).to(dtype)
    logits.requires_grad_(True)
    labels = torch.randint(0, vocab, (batch, length), device="cuda")
    labels[torch.rand(batch, length, device="cuda") < 0.4] = -100  # prompt / padding positions
    labels[0, :] = -100  # a row without any supervised token
    items = None if num_items is None else torch.tensor(num_items, device="cuda")

    loss_fn = fused_loss.make_loss(lambda *a, **k: None)
    loss = loss_fn(logits, labels, vocab, items)
    loss.backward()
    side = fused_loss.lookup(logits)
    assert side is not None

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss, ref_tok, ref_ent, ref_correct = reference(ref_logits, labels, items)
    ref_loss.backward()

    tol = 2e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(loss, ref_loss, rtol=tol, atol=tol)
    torch.testing.assert_close(side["tokloss"], ref_tok, rtol=tol, atol=tol)
    torch.testing.assert_close(side["entropy"], ref_ent, rtol=tol, atol=tol)
    # exact ties between two logits can legitimately differ; allow a handful
    assert ((side["correct"] != 0) != ref_correct).sum().item() <= 1
    torch.testing.assert_close(logits.grad.float(), ref_logits.grad.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("shape", [(1536, 8960), (257, 1000)])
def test_nf4_dequant_matches_bitsandbytes(shape):
    import bitsandbytes.functional as BF
    from bitsandbytes.backends.cuda import ops

    from solution import nf4

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16)
    packed, state = BF.quantize_4bit(weight, blocksize=64, quant_type="nf4",
                                     compress_statistics=True)
    absmax = state.absmax
    if state.nested:
        absmax = BF.dequantize_blockwise(state.absmax, state.state2) + state.offset
    absmax = absmax.float().contiguous()

    expected = torch.empty(shape, dtype=torch.float16, device="cuda")
    ops._dequantize_4bit_impl(packed, absmax, 64, "nf4", torch.float16, out=expected)
    ours = torch.empty(shape, dtype=torch.float16, device="cuda")
    nf4.dequantize_nf4(packed.contiguous(), absmax, ours)
    torch.testing.assert_close(ours, expected, rtol=0, atol=0)
