"""Fused causal-LM loss over the 151,936-token vocabulary (our own Triton kernels).

Base Soup touches the full logits (4 x 512 x 151,936) several times per step: transformers upcasts
them to fp32 and runs cross-entropy, TRL computes the entropy chunk by chunk and copies the
shifted logits to take an argmax, and Soup computes a loss for every row again for rewind.jsonl.

Here ONE forward kernel reads each logits row once and produces everything those places need:
log-sum-exp, the token loss, the entropy and "argmax == label". ONE backward kernel writes the
gradient of the logits straight from the saved log-sum-exp, without a softmax / one-hot copy.
The small per-token results are kept in `_SIDE`, so the patched TRL metric code and the patched
Soup rewind code reuse them instead of reading the logits again (see `patches.py`).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK = 4096
NUM_WARPS = 8

# The per-token results of the latest forward pass (a few KB, never the logits themselves).
_SIDE = {}


@triton.jit
def _fused_ce_fwd_kernel(logits_ptr, target_ptr, lse_ptr, tokloss_ptr, entropy_ptr, correct_ptr,
                         V, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride
    target = tl.load(target_ptr + row)

    m = float("-inf")  # running max
    s = 0.0  # running sum of exp(x - m)
    t = 0.0  # running sum of exp(x - m) * x
    best = float("-inf")  # largest logit so far
    best_idx = 0  # its index
    target_logit = 0.0

    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, 0)
        m_new = tl.maximum(m, block_max)
        alpha = tl.exp(m - m_new)
        e = tl.exp(x - m_new)  # 0 on masked lanes
        x_safe = tl.where(mask, x, 0.0)
        s = s * alpha + tl.sum(e, 0)
        t = t * alpha + tl.sum(e * x_safe, 0)
        m = m_new

        block_arg = tl.argmax(x, 0) + start
        best_idx = tl.where(block_max > best, block_arg, best_idx)
        best = tl.maximum(best, block_max)

        target_logit += tl.sum(tl.where(offs == target, x_safe, 0.0), 0)

    lse = m + tl.log(s)
    valid = target != -100
    tl.store(lse_ptr + row, lse)
    tl.store(entropy_ptr + row, lse - t / s)
    tl.store(tokloss_ptr + row, tl.where(valid, lse - target_logit, 0.0))
    tl.store(correct_ptr + row, ((best_idx == target) & valid).to(tl.int32))


@triton.jit
def _fused_ce_bwd_kernel(logits_ptr, dlogits_ptr, target_ptr, lse_ptr, scale_ptr, V, stride,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offset = row.to(tl.int64) * stride
    target = tl.load(target_ptr + row)
    valid = target != -100
    lse = tl.load(lse_ptr + row)
    scale = tl.load(scale_ptr)

    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(logits_ptr + offset + offs, mask=mask & valid, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        g = tl.where(offs == target, p - 1.0, p) * scale
        g = tl.where(valid, g, 0.0)
        tl.store(dlogits_ptr + offset + offs, g.to(dlogits_ptr.dtype.element_ty), mask=mask)


class _FusedCE(torch.autograd.Function):
    """sum over valid tokens of cross-entropy / denom, for flat logits (N, V) and targets (N,)."""

    @staticmethod
    def forward(ctx, logits, targets, denom, batch, length):
        n, vocab = logits.shape
        device = logits.device
        lse = torch.empty(n, dtype=torch.float32, device=device)
        tokloss = torch.empty(n, dtype=torch.float32, device=device)
        entropy = torch.empty(n, dtype=torch.float32, device=device)
        correct = torch.empty(n, dtype=torch.int32, device=device)
        _fused_ce_fwd_kernel[(n,)](logits, targets, lse, tokloss, entropy, correct, vocab,
                                   logits.stride(0), BLOCK=BLOCK, num_warps=NUM_WARPS)
        _SIDE.clear()
        _SIDE.update(ptr=logits.data_ptr(), numel=logits.numel(), vocab=vocab, batch=batch,
                     length=length, tokloss=tokloss.view(batch, length),
                     entropy=entropy.view(batch, length), correct=correct.view(batch, length))
        ctx.save_for_backward(logits, targets, lse, denom)
        return tokloss.sum() / denom

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets, lse, denom = ctx.saved_tensors
        n, vocab = logits.shape
        dlogits = logits
        scale = (grad_out.to(torch.float32) / denom).reshape(1).contiguous()
        _fused_ce_bwd_kernel[(n,)](logits, dlogits, targets, lse, scale, vocab, logits.stride(0),
                                   BLOCK=BLOCK, num_warps=NUM_WARPS)
        return dlogits, None, None, None, None


def lookup(logits):
    """The per-token results of the latest forward pass, if `logits` has its shape, else None.

    The match is by shape, not by memory address: with fp16 training, accelerate hands the
    trainer an fp32 COPY of the logits, so `outputs.logits` is a different tensor from the one
    the loss saw. `_SIDE` is cleared before every forward pass (see patches.py), so what is in
    it always belongs to the step that is running.
    """
    if not _SIDE or not torch.is_tensor(logits) or logits.dim() != 3:
        return None
    if (logits.shape[0] == _SIDE["batch"] and logits.shape[1] == _SIDE["length"]
            and logits.shape[2] == _SIDE["vocab"]):
        return _SIDE
    return None


def make_loss(original):
    """Build the replacement of transformers' ForCausalLMLoss (same signature)."""

    def fused_for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None,
                                 ignore_index=-100, shift_labels=None, **kwargs):
        if (not torch.is_tensor(logits) or not logits.is_cuda or logits.dim() != 3
                or ignore_index != -100 or logits.shape[-1] != vocab_size):
            return original(logits, labels, vocab_size, num_items_in_batch, ignore_index,
                            shift_labels, **kwargs)
        batch, length, _ = logits.shape
        if shift_labels is None:
            shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
        targets = shift_labels.reshape(-1).to(device=logits.device, dtype=torch.int64)
        flat = logits.reshape(-1, vocab_size)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        if num_items_in_batch is None:
            denom = (targets != ignore_index).sum().to(torch.float32).clamp(min=1.0)
        else:
            denom = torch.as_tensor(num_items_in_batch, device=logits.device).to(torch.float32)
        return _FusedCE.apply(flat, targets.contiguous(), denom, batch, length)

    return fused_for_causal_lm_loss
