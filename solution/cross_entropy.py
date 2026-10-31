"""Chunked LM-head + online-softmax cross-entropy, entropy and accuracy in Triton; the full logits are never materialized."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CHUNK_SIZE = 256
BLOCK_SIZE = 4096
NUM_WARPS = 8


@dataclass(frozen=True)
class _SoupChunkedStepResult:
    token_entropy: Any
    token_loss: Any
    correct: Any
    labelled: Any


@triton.jit
def qwen2_ce_fwd_kernel(
    logits_ptr, target_ptr, lse_ptr, tokloss_ptr, entropy_ptr, correct_ptr, V, stride, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride
    target = tl.load(target_ptr + row)

    m = float("-inf")
    s = 0.0
    t = 0.0
    best = float("-inf")
    best_idx = 0
    target_logit = 0.0

    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)

        block_max = tl.max(x, 0)
        m_new = tl.maximum(m, block_max)
        alpha = tl.exp(m - m_new)
        e = tl.exp(x - m_new)
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
    tl.store(correct_ptr + row, ((best_idx == target) & valid).to(tl.int8))


@triton.jit
def qwen2_ce_bwd_kernel(logits_ptr, dlogits_ptr, target_ptr, lse_ptr, scale_ptr, V, stride, BLOCK: tl.constexpr):
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


class _TritonChunkedLMHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, target, attention_mask, denom, ignore_index, with_grad):
        bsz, seq_len, hidden_dim = hidden.shape
        vocab_size = weight.shape[0]
        total_tokens = bsz * seq_len
        device = hidden.device

        compute_dtype = hidden.dtype
        head_w = weight if weight.dtype == compute_dtype else weight.to(compute_dtype)

        flat_hidden = hidden.detach().contiguous().view(total_tokens, hidden_dim)
        flat_target = target.contiguous().view(total_tokens)

        is_labelled = flat_target != ignore_index
        if attention_mask is not None and attention_mask.numel() == total_tokens:
            flat_mask = attention_mask.contiguous().view(total_tokens) != 0
            is_prompt = flat_mask & (~is_labelled)
        else:
            is_prompt = ~is_labelled

        labelled_idx = torch.nonzero(is_labelled, as_tuple=False).squeeze(-1)
        prompt_idx = torch.nonzero(is_prompt, as_tuple=False).squeeze(-1)

        flat_entropy = torch.zeros(total_tokens, dtype=torch.float32, device=device)
        flat_token_loss = torch.zeros(total_tokens, dtype=torch.float32, device=device)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)

        grad_flat_hidden = torch.zeros_like(flat_hidden) if with_grad else None

        n_labelled = labelled_idx.numel()
        n_prompt = prompt_idx.numel()
        max_m = min(CHUNK_SIZE, max(n_labelled, n_prompt))

        if max_m > 0:
            logits_buf = torch.empty((max_m, vocab_size), dtype=compute_dtype, device=device)
            loss_buf = torch.empty(max_m, dtype=torch.float32, device=device)
            lse_buf = torch.empty(max_m, dtype=torch.float32, device=device)
            ent_buf = torch.empty(max_m, dtype=torch.float32, device=device)
            correct_buf = torch.empty(max_m, dtype=torch.int8, device=device)
            grad_h_buf = torch.empty((max_m, hidden_dim), dtype=compute_dtype, device=device) if with_grad else None
            unit = torch.ones(1, dtype=torch.float32, device=device) if with_grad else None

        for start in range(0, n_labelled, CHUNK_SIZE):
            idx = labelled_idx[start : start + CHUNK_SIZE]
            m = idx.numel()
            h_chunk = flat_hidden.index_select(0, idx)
            t_chunk = flat_target.index_select(0, idx)

            logits_chunk = logits_buf[:m]
            torch.mm(h_chunk, head_w.t(), out=logits_chunk)
            if bias is not None:
                logits_chunk.add_(bias)

            qwen2_ce_fwd_kernel[(m,)](
                logits_chunk,
                t_chunk,
                lse_buf,
                loss_buf,
                ent_buf,
                correct_buf,
                vocab_size,
                logits_chunk.stride(0),
                BLOCK=BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )

            flat_entropy.index_copy_(0, idx, ent_buf[:m])
            flat_token_loss.index_copy_(0, idx, loss_buf[:m])
            total_loss = total_loss + loss_buf[:m].sum()
            correct = correct + correct_buf[:m].to(torch.long).sum()

            if with_grad:
                qwen2_ce_bwd_kernel[(m,)](
                    logits_chunk,
                    logits_chunk,
                    t_chunk,
                    lse_buf,
                    unit,
                    vocab_size,
                    logits_chunk.stride(0),
                    BLOCK=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )
                grad_h = grad_h_buf[:m]
                torch.mm(logits_chunk, head_w, out=grad_h)
                grad_flat_hidden.index_copy_(0, idx, grad_h.to(grad_flat_hidden.dtype))

        for start in range(0, n_prompt, CHUNK_SIZE):
            idx = prompt_idx[start : start + CHUNK_SIZE]
            m = idx.numel()
            h_chunk = flat_hidden.index_select(0, idx)
            t_chunk = flat_target.index_select(0, idx)

            logits_chunk = logits_buf[:m]
            torch.mm(h_chunk, head_w.t(), out=logits_chunk)
            if bias is not None:
                logits_chunk.add_(bias)

            qwen2_ce_fwd_kernel[(m,)](
                logits_chunk,
                t_chunk,
                lse_buf,
                loss_buf,
                ent_buf,
                correct_buf,
                vocab_size,
                logits_chunk.stride(0),
                BLOCK=BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
            flat_entropy.index_copy_(0, idx, ent_buf[:m])

        if max_m > 0:
            del logits_buf, loss_buf, lse_buf, ent_buf, correct_buf, grad_h_buf, unit

        ctx.with_grad = with_grad
        if with_grad:
            ctx.save_for_backward(grad_flat_hidden.view(bsz, seq_len, hidden_dim), denom)
        token_entropy = flat_entropy.view(bsz, seq_len)
        token_loss = flat_token_loss.view(bsz, seq_len)
        ctx.mark_non_differentiable(token_entropy, token_loss, correct)
        return total_loss / denom, token_entropy, token_loss, correct

    @staticmethod
    def backward(ctx, grad_loss, _grad_ent, _grad_tok_loss, _grad_corr):
        if not ctx.with_grad:
            return (None,) * 8
        grad_hidden, denom = ctx.saved_tensors
        if grad_loss is not None:
            scale = (grad_loss.float() / denom.float()).to(grad_hidden.dtype)
            grad_hidden = grad_hidden * scale
        return (grad_hidden,) + (None,) * 7


_CHUNK_STATE = {"enabled": False, "attention_mask": None, "last_step": None, "token_loss": None}


def _run_triton_chunked_head_loss(
    hidden,
    lm_head,
    labels,
    attention_mask=None,
    num_items_in_batch=None,
    shift_labels=None,
    ignore_index=-100,
):
    target = (
        shift_labels.contiguous()
        if shift_labels is not None
        else F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    )
    target = target.to(hidden.device)
    labelled = (target != ignore_index).sum()
    if num_items_in_batch is None:
        denom = labelled.clamp(min=1).to(torch.float32)
    elif torch.is_tensor(num_items_in_batch):
        denom = num_items_in_batch.to(device=hidden.device, dtype=torch.float32).clamp(min=1.0)
    else:
        denom = torch.tensor(float(max(num_items_in_batch, 1)), dtype=torch.float32, device=hidden.device)

    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    loss, token_entropy, token_loss, correct = _TritonChunkedLMHeadLoss.apply(
        hidden,
        lm_head.weight,
        lm_head.bias,
        target,
        attention_mask,
        denom,
        ignore_index,
        with_grad,
    )
    step = _SoupChunkedStepResult(
        token_entropy=token_entropy,
        token_loss=token_loss,
        correct=correct,
        labelled=labelled,
    )
    return loss, step


def _log_chunked_trl_metrics(trainer, mode, inputs, step):
    metrics = getattr(trainer, "_metrics", None)
    gather = getattr(getattr(trainer, "accelerator", None), "gather_for_metrics", None)
    if metrics is None or gather is None:
        return
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        tokens = gather(attention_mask.sum()).sum().item()
    else:
        entropy = step.token_entropy.mean()
        tokens = None
    metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train" and tokens is not None and hasattr(trainer, "_total_train_tokens"):
        trainer._total_train_tokens += tokens
    if hasattr(trainer, "_total_train_tokens"):
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)


def patch_chunked_loss():
    from transformers.models.qwen2 import modeling_qwen2

    orig_causal_fwd = modeling_qwen2.Qwen2ForCausalLM.forward
    if getattr(orig_causal_fwd, "_is_fused", False):
        return

    def fast_qwen2_causal_lm_forward(self, *args, **kwargs):
        if (
            not _CHUNK_STATE["enabled"]
            or "labels" not in kwargs
            or kwargs["labels"] is None
            or not isinstance(kwargs["labels"], torch.Tensor)
            or not kwargs["labels"].is_cuda
            or getattr(self.lm_head.weight, "requires_grad", False)
        ):
            return orig_causal_fwd(self, *args, **kwargs)

        labels = kwargs.pop("labels")
        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        shift_labels = kwargs.get("shift_labels", None)
        attn_mask = kwargs.get("attention_mask", _CHUNK_STATE["attention_mask"])

        held = []
        orig_head_fwd = self.lm_head.forward

        def _hold_head(hidden_states, *h_args, **h_kwargs):
            placeholder = hidden_states.new_empty((*hidden_states.shape[:-1], 0))
            held.append(hidden_states)
            return placeholder

        self.lm_head.forward = _hold_head
        try:
            outputs = orig_causal_fwd(self, *args, **kwargs)
        finally:
            self.lm_head.forward = orig_head_fwd

        if not held:
            kwargs["labels"] = labels
            return orig_causal_fwd(self, *args, **kwargs)

        loss, step = _run_triton_chunked_head_loss(
            held[-1],
            self.lm_head,
            labels,
            attention_mask=attn_mask,
            num_items_in_batch=num_items_in_batch,
            shift_labels=shift_labels,
        )
        _CHUNK_STATE["last_step"] = step
        rest = {k: v for k, v in outputs.items() if k not in ("loss", "logits")}
        return type(outputs)(loss=loss, **rest)

    fast_qwen2_causal_lm_forward._is_fused = True
    modeling_qwen2.Qwen2ForCausalLM.forward = fast_qwen2_causal_lm_forward

    from transformers import Trainer
    from trl.trainer.sft_trainer import SFTTrainer

    orig_sft_compute_loss = SFTTrainer.compute_loss

    def fast_sft_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (
            not isinstance(inputs, dict)
            or "labels" not in inputs
            or not isinstance(inputs["labels"], torch.Tensor)
            or not inputs["labels"].is_cuda
        ):
            return orig_sft_compute_loss(
                self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        mode = "train" if getattr(model, "training", False) else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False

        _CHUNK_STATE["enabled"] = True
        _CHUNK_STATE["attention_mask"] = inputs.get("attention_mask")
        _CHUNK_STATE["last_step"] = None
        _CHUNK_STATE["token_loss"] = None
        try:
            loss, outputs = Trainer.compute_loss(
                self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        finally:
            _CHUNK_STATE["enabled"] = False
            _CHUNK_STATE["attention_mask"] = None

        step = _CHUNK_STATE["last_step"]
        _CHUNK_STATE["last_step"] = None
        setattr(self, "_soup_chunked_step", step)
        if step is not None:
            _CHUNK_STATE["token_loss"] = step.token_loss
            _log_chunked_trl_metrics(self, mode, inputs, step)
        return (loss, outputs) if return_outputs else loss

    fast_sft_compute_loss._is_fused = True
    SFTTrainer.compute_loss = fast_sft_compute_loss

    try:
        from soup_cli.trainer import rewind_hf

        original_row_losses = rewind_hf.row_losses

        def row_losses(logits, labels):
            token_loss = _CHUNK_STATE["token_loss"]
            if token_loss is not None and tuple(token_loss.shape) == tuple(labels.shape):
                return rewind_hf.row_losses_from_token_losses(token_loss, labels)
            return original_row_losses(logits, labels)

        row_losses._is_fused = True
        rewind_hf.row_losses = row_losses
    except Exception:
        pass