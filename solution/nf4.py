import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NF4_CODE = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141506195, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039558105, 0.7229568362236999, 1.0)
SUPPORTED_BLOCKSIZES = (32, 64, 128)
BLOCK = 1024
NUM_WARPS = 4

_codes = {}
_scratch = {}


@triton.jit
def _nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, out_ptr, n_out, n_bytes,
                        HALF_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n_bytes
    packed = tl.load(packed_ptr + offs, mask=in_range, other=0).to(tl.int32)
    scales = tl.load(absmax_ptr + offs // HALF_BLOCK, mask=in_range, other=0.0)
    hi = tl.load(code_ptr + (packed >> 4)) * scales
    lo = tl.load(code_ptr + (packed & 15)) * scales
    pair = tl.arange(0, 2)[None, :]
    vals = tl.where(pair == 0, hi[:, None], lo[:, None])
    out_offs = offs[:, None] * 2 + pair
    tl.store(out_ptr + out_offs, vals.to(out_ptr.dtype.element_ty), mask=out_offs < n_out)


@triton.jit
def _nested_scales_kernel(q_ptr, code_ptr, absmax2_ptr, offset_ptr, out_ptr, n_blocks,
                          S2_BLOCKSIZE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n_blocks
    q = tl.load(q_ptr + offs, mask=in_range, other=0).to(tl.int32)
    code = tl.load(code_ptr + q, mask=in_range, other=0.0)
    absmax2 = tl.load(absmax2_ptr + offs // S2_BLOCKSIZE, mask=in_range, other=0.0)
    offset = tl.load(offset_ptr)
    tl.store(out_ptr + offs, code * absmax2 + offset, mask=in_range)


def _code_table(device):
    table = _codes.get(device)
    if table is None:
        table = _codes[device] = torch.tensor(NF4_CODE, dtype=torch.float32, device=device)
    return table


def dequantize_nf4(packed, absmax, out, blocksize=64):
    packed = packed.contiguous()
    absmax = absmax.contiguous()
    n_out = out.numel()
    n_bytes = (n_out + 1) // 2
    _nf4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK),)](
        packed, absmax, _code_table(out.device), out, n_out, n_bytes,
        HALF_BLOCK=blocksize // 2, BLOCK=BLOCK, num_warps=NUM_WARPS)


def _block_scales(quant_state, device):
    absmax = quant_state.absmax
    if not getattr(quant_state, "nested", False):
        return absmax.to(device=device, dtype=torch.float32).contiguous()
    state2 = quant_state.state2
    n_blocks = absmax.numel()
    scales = torch.empty(n_blocks, dtype=torch.float32, device=device)
    offset = torch.as_tensor(quant_state.offset, dtype=torch.float32, device=device).reshape(1)
    _nested_scales_kernel[(triton.cdiv(n_blocks, 1024),)](
        absmax.contiguous(), state2.code.to(torch.float32).contiguous(),
        state2.absmax.to(torch.float32).contiguous(), offset, scales, n_blocks,
        S2_BLOCKSIZE=int(state2.blocksize), BLOCK=1024)
    return scales


def triton_dequantize_nf4(weight, quant_state, out=None, dtype=None):
    shape = quant_state.shape
    out_dtype = dtype if dtype is not None else quant_state.dtype
    if out is None:
        out = torch.empty(shape, dtype=out_dtype, device=weight.device)
    scales = _block_scales(quant_state, weight.device)
    dequantize_nf4(weight, scales, out, int(quant_state.blocksize))
    return out


def _dequantize_nf4_scratch(weight, quant_state, dtype):
    key = (weight.device, dtype, tuple(quant_state.shape))
    buf = _scratch.get(key)
    if buf is None:
        buf = _scratch[key] = torch.empty(tuple(quant_state.shape), dtype=dtype, device=weight.device)
    return triton_dequantize_nf4(weight, quant_state, out=buf, dtype=dtype)


def _nf4_supported(quant_state):
    return (
        quant_state is not None
        and getattr(quant_state, "quant_type", None) == "nf4"
        and int(getattr(quant_state, "blocksize", 0)) in SUPPORTED_BLOCKSIZES
        and len(quant_state.shape) == 2
    )


def _is_transposed_orientation(A, quant_state):
    k = A.shape[-1]
    return k == quant_state.shape[0] and k != quant_state.shape[1]


class TritonMatMul4Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        ctx.quant_state = quant_state
        ctx.has_bias = bias is not None
        ctx.transposed = _is_transposed_orientation(A, quant_state)
        ctx.save_for_backward(B)
        weight = _dequantize_nf4_scratch(B, quant_state, A.dtype)
        if ctx.transposed:
            weight = weight.t()
        bias_c = bias.to(A.dtype) if (bias is not None and bias.dtype != A.dtype) else bias
        output = F.linear(A, weight, bias_c)
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
            weight = _dequantize_nf4_scratch(B, ctx.quant_state, grad_output.dtype)
            if ctx.transposed:
                weight = weight.t()
            grad_A = torch.matmul(grad_output, weight)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_A, None, None, grad_bias, None


def make_dequant(original):
    def dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out):
        if (quant_type == "nf4" and blocksize in SUPPORTED_BLOCKSIZES and torch.is_tensor(A)
                and A.is_cuda and out.is_cuda and out.is_contiguous()
                and absmax.dtype == torch.float32
                and out.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            dequantize_nf4(A, absmax, out, blocksize)
            return None
        return original(A, absmax, blocksize, quant_type, dtype, out)

    dequantize_4bit_impl._is_fused = True
    return dequantize_4bit_impl


def patch_nf4_dequant():
    from bitsandbytes.backends.cuda import ops

    original = ops._dequantize_4bit_impl
    if getattr(original, "_is_fused", False):
        return
    ops._dequantize_4bit_impl = make_dequant(original)
