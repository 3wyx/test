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