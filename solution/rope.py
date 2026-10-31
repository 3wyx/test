import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, sxb, sxh, sxt, scb, sct, T, H,
                     HALF: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    i = tl.arange(0, HALF)
    xbase = x_ptr + b * sxb + h * sxh + t * sxt
    cbase = b * scb + t * sct
    x1 = tl.load(xbase + i).to(tl.float32)
    x2 = tl.load(xbase + HALF + i).to(tl.float32)
    c1 = tl.load(cos_ptr + cbase + i).to(tl.float32)
    c2 = tl.load(cos_ptr + cbase + HALF + i).to(tl.float32)
    s1 = tl.load(sin_ptr + cbase + i).to(tl.float32)
    s2 = tl.load(sin_ptr + cbase + HALF + i).to(tl.float32)
    ybase = y_ptr + ((b * H + h) * T + t) * (2 * HALF)
    out_t = y_ptr.dtype.element_ty
    tl.store(ybase + i, (x1 * c1 - x2 * s1).to(out_t))
    tl.store(ybase + HALF + i, (x2 * c2 + x1 * s2).to(out_t))


@triton.jit
def _rope_bwd_kernel(g_ptr, cos_ptr, sin_ptr, dx_ptr, scb, sct, T, H, HALF: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    i = tl.arange(0, HALF)
    offset = ((b * H + h) * T + t) * (2 * HALF)
    cbase = b * scb + t * sct
    g1 = tl.load(g_ptr + offset + i).to(tl.float32)
    g2 = tl.load(g_ptr + offset + HALF + i).to(tl.float32)
    c1 = tl.load(cos_ptr + cbase + i).to(tl.float32)
    c2 = tl.load(cos_ptr + cbase + HALF + i).to(tl.float32)
    s1 = tl.load(sin_ptr + cbase + i).to(tl.float32)
    s2 = tl.load(sin_ptr + cbase + HALF + i).to(tl.float32)
    out_t = dx_ptr.dtype.element_ty
    tl.store(dx_ptr + offset + i, (g1 * c1 + g2 * s2).to(out_t))
    tl.store(dx_ptr + offset + HALF + i, (g2 * c2 - g1 * s1).to(out_t))


class _Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, out_dtype):
        b, h, t, d = x.shape
        y = torch.empty((b, h, t, d), dtype=out_dtype, device=x.device)
        scb = cos.stride(0) if cos.shape[0] > 1 else 0
        _rope_fwd_kernel[(t, h, b)](x, cos, sin, y, x.stride(0), x.stride(1), x.stride(2),
                                    scb, cos.stride(1), t, h, HALF=d // 2)
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        g = grad_out.contiguous()
        b, h, t, d = g.shape
        dx = torch.empty((b, h, t, d), dtype=ctx.x_dtype, device=g.device)
        scb = cos.stride(0) if cos.shape[0] > 1 else 0
        _rope_bwd_kernel[(t, h, b)](g, cos, sin, dx, scb, cos.stride(1), t, h, HALF=d // 2)
        return dx, None, None, None


_OK = (torch.float16, torch.bfloat16, torch.float32)


def _supported(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim != 1 or q.dim() != 4 or k.dim() != 4 or cos.dim() != 3:
        return False
    if not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    if q.dtype not in _OK or k.dtype not in _OK or cos.dtype not in _OK or sin.dtype != cos.dtype:
        return False
    b, _, t, d = q.shape
    half = d // 2
    if d % 2 or half & (half - 1):
        return False
    if k.shape[0] != b or k.shape[2] != t or k.shape[3] != d:
        return False
    if cos.shape != sin.shape or cos.shape[1] != t or cos.shape[2] != d:
        return False
    if cos.shape[0] not in (1, b):
        return False
    return q.stride(-1) == 1 and k.stride(-1) == 1


def _out_dtype(x, cos):
    out = torch.promote_types(x.dtype, cos.dtype)
    if out == torch.float32 and torch.is_autocast_enabled():
        out = torch.get_autocast_dtype("cuda")
    return out


def make_rope(original):
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if not _supported(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim)
        cos = cos.contiguous()
        sin = sin.contiguous()
        q_embed = _Rope.apply(q, cos, sin, _out_dtype(q, cos))
        k_embed = _Rope.apply(k, cos, sin, _out_dtype(k, cos))
        return q_embed, k_embed

    apply_rotary_pos_emb._is_fused = True
    return apply_rotary_pos_emb