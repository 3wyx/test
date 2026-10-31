"""Fused Q+K rotary embedding kernels and a compacted RoPE + attention node that recomputes attention in backward."""

import torch
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint


@triton.jit
def qwen2_rope_fwd_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_cos_b,
    stride_cos_s,
    stride_qob,
    stride_qoh,
    stride_qos,
    stride_kob,
    stride_koh,
    stride_kos,
    n_qh,
    seq_len,
    HALF_DIM: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_idx = pid_bs // seq_len
    s_idx = pid_bs % seq_len

    offs = tl.arange(0, HALF_DIM)
    cos_base = cos_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_base = sin_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s

    c1 = tl.load(cos_base + offs).to(tl.float32)
    c2 = tl.load(cos_base + HALF_DIM + offs).to(tl.float32)
    s1 = tl.load(sin_base + offs).to(tl.float32)
    s2 = tl.load(sin_base + HALF_DIM + offs).to(tl.float32)

    if pid_h < n_qh:
        in_base = q_ptr + b_idx * stride_qb + pid_h * stride_qh + s_idx * stride_qs
        out_base = q_out_ptr + b_idx * stride_qob + pid_h * stride_qoh + s_idx * stride_qos
    else:
        kh_idx = pid_h - n_qh
        in_base = k_ptr + b_idx * stride_kb + kh_idx * stride_kh + s_idx * stride_ks
        out_base = k_out_ptr + b_idx * stride_kob + kh_idx * stride_koh + s_idx * stride_kos

    x1 = tl.load(in_base + offs).to(tl.float32)
    x2 = tl.load(in_base + HALF_DIM + offs).to(tl.float32)

    y1 = x1 * c1 - x2 * s1
    y2 = x2 * c2 + x1 * s2

    tl.store(out_base + offs, y1.to(q_out_ptr.dtype.element_ty))
    tl.store(out_base + HALF_DIM + offs, y2.to(q_out_ptr.dtype.element_ty))


@triton.jit
def qwen2_rope_bwd_kernel(
    dq_out_ptr,
    dk_out_ptr,
    cos_ptr,
    sin_ptr,
    dq_ptr,
    dk_ptr,
    stride_dqob,
    stride_dqoh,
    stride_dqos,
    stride_dkob,
    stride_dkoh,
    stride_dkos,
    stride_cos_b,
    stride_cos_s,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dkb,
    stride_dkh,
    stride_dks,
    n_qh,
    seq_len,
    HALF_DIM: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_idx = pid_bs // seq_len
    s_idx = pid_bs % seq_len

    offs = tl.arange(0, HALF_DIM)
    cos_base = cos_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_base = sin_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s

    c1 = tl.load(cos_base + offs).to(tl.float32)
    c2 = tl.load(cos_base + HALF_DIM + offs).to(tl.float32)
    s1 = tl.load(sin_base + offs).to(tl.float32)
    s2 = tl.load(sin_base + HALF_DIM + offs).to(tl.float32)

    if pid_h < n_qh:
        dy_base = dq_out_ptr + b_idx * stride_dqob + pid_h * stride_dqoh + s_idx * stride_dqos
        dx_base = dq_ptr + b_idx * stride_dqb + pid_h * stride_dqh + s_idx * stride_dqs
    else:
        kh_idx = pid_h - n_qh
        dy_base = dk_out_ptr + b_idx * stride_dkob + kh_idx * stride_dkoh + s_idx * stride_dkos
        dx_base = dk_ptr + b_idx * stride_dkb + kh_idx * stride_dkh + s_idx * stride_dks

    dy1 = tl.load(dy_base + offs).to(tl.float32)
    dy2 = tl.load(dy_base + HALF_DIM + offs).to(tl.float32)

    dx1 = dy1 * c1 + dy2 * s2
    dx2 = dy2 * c2 - dy1 * s1

    tl.store(dx_base + offs, dx1.to(dq_ptr.dtype.element_ty))
    tl.store(dx_base + HALF_DIM + offs, dx2.to(dq_ptr.dtype.element_ty))


class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        if cos.ndim == 4:
            cos = cos.squeeze(1)
            sin = sin.squeeze(1)
        cos = cos.contiguous()
        sin = sin.contiguous()

        bsz, n_qh, seq_len, head_dim = q.shape
        _, n_kh, _, _ = k.shape
        half_dim = head_dim // 2

        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()

        q_out = torch.empty((bsz, n_qh, seq_len, head_dim), dtype=q.dtype, device=q.device)
        k_out = torch.empty((bsz, n_kh, seq_len, head_dim), dtype=k.dtype, device=k.device)

        stride_cos_b = cos.stride(0) if cos.size(0) > 1 else 0
        stride_cos_s = cos.stride(1)

        qwen2_rope_fwd_kernel[(bsz * seq_len, n_qh + n_kh)](
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            stride_cos_b,
            stride_cos_s,
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            n_qh,
            seq_len,
            HALF_DIM=half_dim,
            num_warps=2,
        )

        ctx.save_for_backward(cos, sin)
        ctx.stride_cos_b = stride_cos_b
        ctx.stride_cos_s = stride_cos_s
        ctx.half_dim = half_dim
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out, dk_out):
        cos, sin = ctx.saved_tensors
        bsz, n_qh, seq_len, head_dim = dq_out.shape
        _, n_kh, _, _ = dk_out.shape

        dq = dq_out.contiguous()
        dk = dk_out.contiguous()

        qwen2_rope_bwd_kernel[(bsz * seq_len, n_qh + n_kh)](
            dq,
            dk,
            cos,
            sin,
            dq,
            dk,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            ctx.stride_cos_b,
            ctx.stride_cos_s,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            n_qh,
            seq_len,
            HALF_DIM=ctx.half_dim,
            num_warps=2,
        )
        return dq, dk, None, None


class _CompactRoPESDPA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_valid,
        k_valid,
        v_valid,
        cos_valid,
        sin_valid,
        active_idx_smax,
        attn_mask,
        bsz,
        s_max,
        n_qh,
        n_kh,
        head_dim,
        attn_module,
        orig_sdpa,
    ):
        n_active = q_valid.shape[0]
        half_dim = head_dim // 2

        q_3d = q_valid.contiguous().view(n_active, n_qh, head_dim)
        k_3d = k_valid.contiguous().view(n_active, n_kh, head_dim)
        v_3d = v_valid.contiguous().view(n_active, n_kh, head_dim)

        q_rope = torch.empty_like(q_3d)
        k_rope = torch.empty_like(k_3d)

        cos_c = cos_valid.contiguous().view(1, n_active, head_dim)
        sin_c = sin_valid.contiguous().view(1, n_active, head_dim)

        qwen2_rope_fwd_kernel[(n_active, n_qh + n_kh)](
            q_3d,
            k_3d,
            cos_c,
            sin_c,
            q_rope,
            k_rope,
            0,
            q_3d.stride(1),
            q_3d.stride(0),
            0,
            k_3d.stride(1),
            k_3d.stride(0),
            0,
            cos_c.stride(1),
            0,
            q_rope.stride(1),
            q_rope.stride(0),
            0,
            k_rope.stride(1),
            k_rope.stride(0),
            n_qh,
            n_active,
            HALF_DIM=half_dim,
            num_warps=2,
        )

        total_smax = bsz * s_max
        q_pad = torch.zeros((total_smax, n_qh, head_dim), dtype=q_valid.dtype, device=q_valid.device)
        k_pad = torch.zeros((total_smax, n_kh, head_dim), dtype=k_valid.dtype, device=k_valid.device)
        v_pad = torch.zeros((total_smax, n_kh, head_dim), dtype=v_valid.dtype, device=v_valid.device)
        q_pad.index_copy_(0, active_idx_smax, q_rope)
        k_pad.index_copy_(0, active_idx_smax, k_rope)
        v_pad.index_copy_(0, active_idx_smax, v_3d)

        q_4d = q_pad.view(bsz, s_max, n_qh, head_dim).transpose(1, 2)
        k_4d = k_pad.view(bsz, s_max, n_kh, head_dim).transpose(1, 2)
        v_4d = v_pad.view(bsz, s_max, n_kh, head_dim).transpose(1, 2)

        with torch.no_grad():
            attn_out, _ = orig_sdpa(
                attn_module,
                q_4d,
                k_4d,
                v_4d,
                attn_mask,
                dropout=0.0,
                scaling=attn_module.scaling,
                sliding_window=getattr(attn_module, "sliding_window", None),
            )

        out_2d = attn_out.reshape(total_smax, n_qh * head_dim)
        out_valid = out_2d.index_select(0, active_idx_smax)

        ctx.save_for_backward(q_rope, k_rope, v_3d, cos_c, sin_c, active_idx_smax, attn_mask)
        ctx.bsz = bsz
        ctx.s_max = s_max
        ctx.n_qh = n_qh
        ctx.n_kh = n_kh
        ctx.head_dim = head_dim
        ctx.half_dim = half_dim
        ctx.attn_module = attn_module
        ctx.orig_sdpa = orig_sdpa
        return out_valid

    @staticmethod
    def backward(ctx, grad_out_valid):
        device = grad_out_valid.device
        q_rope, k_rope, v_3d, cos_c, sin_c, active_idx_smax, attn_mask = ctx.saved_tensors

        bsz = ctx.bsz
        s_max = ctx.s_max
        n_qh = ctx.n_qh
        n_kh = ctx.n_kh
        head_dim = ctx.head_dim
        total_smax = bsz * s_max
        n_active = k_rope.shape[0]

        q_pad = torch.zeros((total_smax, n_qh, head_dim), dtype=q_rope.dtype, device=device)
        k_pad = torch.zeros((total_smax, n_kh, head_dim), dtype=k_rope.dtype, device=device)
        v_pad = torch.zeros((total_smax, n_kh, head_dim), dtype=v_3d.dtype, device=device)
        q_pad.index_copy_(0, active_idx_smax, q_rope)
        k_pad.index_copy_(0, active_idx_smax, k_rope)
        v_pad.index_copy_(0, active_idx_smax, v_3d)

        q_4d = q_pad.view(bsz, s_max, n_qh, head_dim).transpose(1, 2).detach().requires_grad_(True)
        k_4d = k_pad.view(bsz, s_max, n_kh, head_dim).transpose(1, 2).detach().requires_grad_(True)
        v_4d = v_pad.view(bsz, s_max, n_kh, head_dim).transpose(1, 2).detach().requires_grad_(True)

        grad_pad = torch.zeros((total_smax, n_qh * head_dim), dtype=grad_out_valid.dtype, device=device)
        grad_pad.index_copy_(0, active_idx_smax, grad_out_valid.contiguous())
        grad_4d = grad_pad.view(bsz, s_max, n_qh, head_dim)

        with torch.enable_grad():
            attn_out, _ = ctx.orig_sdpa(
                ctx.attn_module,
                q_4d,
                k_4d,
                v_4d,
                attn_mask,
                dropout=0.0,
                scaling=ctx.attn_module.scaling,
                sliding_window=getattr(ctx.attn_module, "sliding_window", None),
            )
            torch.autograd.backward(attn_out, grad_4d)

        dq_rope_pad = q_4d.grad.transpose(1, 2).reshape(total_smax, n_qh, head_dim)
        dk_rope_pad = k_4d.grad.transpose(1, 2).reshape(total_smax, n_kh, head_dim)
        dv_pad = v_4d.grad.transpose(1, 2).reshape(total_smax, n_kh * head_dim)

        dq_rope = dq_rope_pad.index_select(0, active_idx_smax)
        dk_rope = dk_rope_pad.index_select(0, active_idx_smax)
        dv_valid = dv_pad.index_select(0, active_idx_smax)
        del q_4d, k_4d, v_4d, q_pad, k_pad, v_pad, dq_rope_pad, dk_rope_pad, dv_pad, grad_pad, grad_4d, attn_out

        qwen2_rope_bwd_kernel[(n_active, n_qh + n_kh)](
            dq_rope,
            dk_rope,
            cos_c,
            sin_c,
            dq_rope,
            dk_rope,
            0,
            dq_rope.stride(1),
            dq_rope.stride(0),
            0,
            dk_rope.stride(1),
            dk_rope.stride(0),
            0,
            cos_c.stride(1),
            0,
            dq_rope.stride(1),
            dq_rope.stride(0),
            0,
            dk_rope.stride(1),
            dk_rope.stride(0),
            n_qh,
            n_active,
            HALF_DIM=ctx.half_dim,
            num_warps=2,
        )
        return (
            dq_rope.view(n_active, n_qh * head_dim),
            dk_rope.view(n_active, n_kh * head_dim),
            dv_valid,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rope_qwen2(q, k, cos, sin):
    return TritonRoPE.apply(q, k, cos, sin)


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original_rope = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original_rope, "_is_fused", False):
        return

    def fast_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if (
            not isinstance(q, torch.Tensor)
            or not q.is_cuda
            or unsqueeze_dim != 1
            or q.ndim != 4
            or (q.shape[-1] % 2 != 0)
        ):
            return original_rope(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
        return rope_qwen2(q, k, cos, sin)

    fast_apply_rotary_pos_emb._is_fused = True
    modeling_qwen2.apply_rotary_pos_emb = fast_apply_rotary_pos_emb

    try:
        attn_funcs = getattr(modeling_qwen2, "ALL_ATTENTION_FUNCTIONS", None)
        if attn_funcs is not None and "sdpa" in attn_funcs:
            orig_sdpa = attn_funcs["sdpa"]
            if getattr(orig_sdpa, "_orig_sdpa", None) is not None:
                return

            def checkpointed_sdpa(module, query, key, value, attention_mask, *args, **kwargs):
                if (
                    torch.is_grad_enabled()
                    and isinstance(query, torch.Tensor)
                    and query.is_cuda
                    and query.requires_grad
                    and not kwargs.get("output_attentions", False)
                ):

                    def _run_sdpa(q, k, v, mask):
                        out, _ = orig_sdpa(module, q, k, v, mask, *args, **kwargs)
                        return out

                    attn_out = checkpoint(_run_sdpa, query, key, value, attention_mask, use_reentrant=False)
                    return attn_out, None
                return orig_sdpa(module, query, key, value, attention_mask, *args, **kwargs)

            checkpointed_sdpa._orig_sdpa = orig_sdpa
            attn_funcs["sdpa"] = checkpointed_sdpa
    except Exception:
        pass