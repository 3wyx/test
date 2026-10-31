"""NF4 dequantization (own Triton kernel, nested absmax) and a frozen 4-bit matmul that reuses one scratch buffer."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def qwen2_nf4_dequant_kernel(
    w_ptr,
    absmax_ptr,
    code_ptr,
    s2_absmax_ptr,
    s2_code_ptr,
    offset_ptr,
    out_ptr,
    n_bytes,
    n_blocks,
    BLOCKSIZE_HALF: tl.constexpr,
    S2_BLOCKSIZE: tl.constexpr,
    IS_NESTED: tl.constexpr,
    BLOCKS_PER_PID: tl.constexpr,
):
    pid = tl.program_id(0)
    blk_offs = pid * BLOCKS_PER_PID + tl.arange(0, BLOCKS_PER_PID)
    blk_mask = blk_offs < n_blocks

    if IS_NESTED:
        q_abs = tl.load(absmax_ptr + blk_offs, mask=blk_mask, other=0).to(tl.int32)
        s2_val = tl.load(s2_code_ptr + q_abs, mask=blk_mask, other=0.0).to(tl.float32)
        s2_abs = tl.load(s2_absmax_ptr + (blk_offs // S2_BLOCKSIZE), mask=blk_mask, other=0.0).to(tl.float32)
        offset_val = tl.load(offset_ptr).to(tl.float32)
        scale = tl.fma(s2_val, s2_abs, offset_val)
    else:
        scale = tl.load(absmax_ptr + blk_offs, mask=blk_mask, other=0.0).to(tl.float32)

    byte_offs = blk_offs[:, None] * BLOCKSIZE_HALF + tl.arange(0, BLOCKSIZE_HALF)[None, :]
    byte_mask = byte_offs < n_bytes

    packed = tl.load(w_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)
    hi_nib = (packed >> 4) & 0x0F
    lo_nib = packed & 0x0F

    val_hi = tl.load(code_ptr + hi_nib, mask=byte_mask, other=0.0).to(tl.float32)
    val_lo = tl.load(code_ptr + lo_nib, mask=byte_mask, other=0.0).to(tl.float32)

    scale_2d = scale[:, None]
    out_pair = tl.join(val_hi * scale_2d, val_lo * scale_2d)

    out_elems_per_pid: tl.constexpr = BLOCKS_PER_PID * BLOCKSIZE_HALF * 2
    out_offs = pid * out_elems_per_pid + tl.arange(0, out_elems_per_pid)
    out_mask = out_offs < (n_bytes * 2)
    tl.store(
        out_ptr + out_offs,
        tl.reshape(out_pair, (out_elems_per_pid,)).to(out_ptr.dtype.element_ty),
        mask=out_mask,
    )


_DUMMY_TENSOR = None
_DEQ_SCRATCH = {}
_SCRATCH_MIN = 8960 * 1536


def _get_dummy(device):
    global _DUMMY_TENSOR
    if _DUMMY_TENSOR is None or _DUMMY_TENSOR.device != device:
        _DUMMY_TENSOR = torch.zeros(1, dtype=torch.float32, device=device)
    return _DUMMY_TENSOR


def _get_deq_scratch(shape, dtype, device):
    key = (device, dtype)
    numel = shape.numel()
    buf = _DEQ_SCRATCH.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(max(numel, _SCRATCH_MIN), dtype=dtype, device=device)
        _DEQ_SCRATCH[key] = buf
    return buf[:numel].view(shape)


def triton_dequantize_nf4(weight, quant_state, out=None, dtype=None):
    out_shape = quant_state.shape
    out_dtype = dtype if dtype is not None else quant_state.dtype
    n_elements = out_shape.numel()
    n_bytes = n_elements // 2

    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=weight.device)

    w_flat = weight.contiguous().view(-1)
    out_flat = out.view(-1)

    blocksize_half = int(quant_state.blocksize) // 2
    n_blocks = triton.cdiv(n_bytes, blocksize_half)
    is_nested = bool(getattr(quant_state, "nested", False))

    if is_nested:
        state2 = quant_state.state2
        s2_absmax = state2.absmax
        s2_code = state2.code
        s2_blocksize = int(state2.blocksize)
        offset = quant_state.offset
    else:
        dummy = _get_dummy(weight.device)
        s2_absmax = dummy
        s2_code = dummy
        s2_blocksize = 256
        offset = dummy

    blocks_per_pid = 32
    grid = (triton.cdiv(n_blocks, blocks_per_pid),)
    qwen2_nf4_dequant_kernel[grid](
        w_flat,
        quant_state.absmax,
        quant_state.code,
        s2_absmax,
        s2_code,
        offset,
        out_flat,
        n_bytes,
        n_blocks,
        BLOCKSIZE_HALF=blocksize_half,
        S2_BLOCKSIZE=s2_blocksize,
        IS_NESTED=is_nested,
        BLOCKS_PER_PID=blocks_per_pid,
        num_warps=4,
    )
    return out


def _dequantize_nf4_scratch(weight, quant_state, dtype):
    scratch = _get_deq_scratch(quant_state.shape, dtype, weight.device)
    return triton_dequantize_nf4(weight, quant_state, out=scratch, dtype=dtype)


class TritonMatMul4Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        ctx.quant_state = quant_state
        ctx.has_bias = bias is not None
        ctx.save_for_backward(B)

        w_deq = _dequantize_nf4_scratch(B, quant_state, dtype=A.dtype)
        bias_c = bias.to(A.dtype) if (bias is not None and bias.dtype != A.dtype) else bias
        output = F.linear(A, w_deq, bias_c)
        if out is not None:
            out.copy_(output)
            return out
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (B,) = ctx.saved_tensors
        grad_A = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            w_deq = _dequantize_nf4_scratch(B, ctx.quant_state, dtype=grad_output.dtype)
            grad_A = torch.matmul(grad_output, w_deq)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_A, None, None, grad_bias, None


def _nf4_supported(quant_state):
    return (
        quant_state is not None
        and getattr(quant_state, "quant_type", None) == "nf4"
        and getattr(quant_state, "blocksize", 0) in (32, 64, 128)
        and quant_state.shape.numel() % 2 == 0
    )


def patch_nf4_dequant():
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F

    original_dequant = bnb_F.dequantize_4bit
    if getattr(original_dequant, "_is_fused", False):
        return

    def fast_dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=64, quant_type="fp4"):
        if (
            _nf4_supported(quant_state)
            and isinstance(A, torch.Tensor)
            and A.is_cuda
            and A.dtype == torch.uint8
        ):
            return triton_dequantize_nf4(A, quant_state, out=out)
        return original_dequant(
            A, quant_state=quant_state, absmax=absmax, out=out, blocksize=blocksize, quant_type=quant_type
        )

    fast_dequantize_4bit._is_fused = True
    bnb_F.dequantize_4bit = fast_dequantize_4bit

    original_matmul_4bit = bnb.matmul_4bit

    def fast_matmul_4bit(A, B, out=None, bias=None, quant_state=None):
        if (
            _nf4_supported(quant_state)
            and isinstance(A, torch.Tensor)
            and A.is_cuda
            and isinstance(B, torch.Tensor)
            and B.is_cuda
            and B.dtype == torch.uint8
        ):
            return TritonMatMul4Bit.apply(A, B, out, bias, quant_state)
        return original_matmul_4bit(A, B, out=out, bias=bias, quant_state=quant_state)

    bnb.matmul_4bit = fast_matmul_4bit
    try:
        import bitsandbytes.autograd._functions as bnb_autograd

        bnb_autograd.matmul_4bit = fast_matmul_4bit
        if hasattr(bnb_autograd, "dequantize_4bit"):
            bnb_autograd.dequantize_4bit = fast_dequantize_4bit
    except Exception:
        pass
    try:
        import bitsandbytes.nn.modules as bnb_modules

        if hasattr(bnb_modules, "bnb"):
            bnb_modules.bnb.matmul_4bit = fast_matmul_4bit
    except Exception:
        pass