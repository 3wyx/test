"""NF4 dequantization (own Triton kernel, nested absmax) and a frozen 4-bit matmul that reuses one scratch buffer."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _nf4_dequant_block_kernel(
    w_ptr,
    absmax_ptr,
    code_ptr,
    out_ptr,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    k_start = pid_k * BLOCK_K

    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)

    packed_offs = offs_n[:, None] * (K // 2) + (offs_k // 2)[:, None]
    packed = tl.load(w_ptr + packed_offs, mask=(offs_n[:, None] < N) & (offs_k[:, None] < K), other=0)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    scale_idx = (offs_k // BLOCKSIZE)
    scale = tl.load(absmax_ptr + scale_idx, mask=(scale_idx < (K // BLOCKSIZE)), other=0.0).to(tl.float32)

    low_val = tl.load(code_ptr + low, mask=(low < 16), other=0.0).to(tl.float32)
    high_val = tl.load(code_ptr + high, mask=(high < 16), other=0.0).to(tl.float32)

    vals = tl.where((offs_k % 2) == 0, high_val, low_val)
    deq = vals * scale

    out_offs = offs_n[:, None] * K + offs_k[None, :]
    tl.store(out_ptr + out_offs, deq, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))


@triton.jit
def _nf4_matmul_fused_kernel(
    a_ptr,
    w_ptr,
    absmax_ptr,
    code_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCKSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_block = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        packed_k = offs_k // 2
        packed = tl.load(
            w_ptr + offs_n[:, None] * (K // 2) + packed_k[None, :],
            mask=(offs_n[:, None] < N) & (packed_k[None, :] < (K // 2)),
            other=0,
        )

        low = packed & 0x0F
        high = (packed >> 4) & 0x0F

        code_low = tl.load(code_ptr + low, mask=(low < 16), other=0.0).to(tl.float32)
        code_high = tl.load(code_ptr + high, mask=(high < 16), other=0.0).to(tl.float32)

        even = (offs_k % 2 == 0)
        vals = tl.where(even[None, :], code_high, code_low)

        scale_idx = offs_k // BLOCKSIZE
        scale = tl.load(absmax_ptr + scale_idx, mask=(scale_idx < (K // BLOCKSIZE)), other=0.0).to(tl.float32)
        deq_w = vals * scale[None, :]

        acc += tl.dot(a_block, deq_w.T).to(tl.float32)

    out_offs = offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_offs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _nf4_codebook_from_quant_state(quant_state):
    return quant_state.code.contiguous()


def _nf4_absmax_from_quant_state(quant_state):
    return quant_state.absmax.contiguous().reshape(-1)


def _normalize_nf4_weight_layout(weight, quant_state):
    """Keep a transposed weight in its already matmul-ready layout.

    In bitsandbytes, callers often pass `quant_w.t()` to `matmul_4bit`, which is already
    shaped for the `A @ W` convention. Transposing it a second time corrupts the layout and
    produces tiny but systematic mismatches against the reference output.
    """
    weight = weight.contiguous()
    if weight.dim() == 2 and weight.shape == quant_state.shape:
        if weight.stride(0) == 1 and weight.stride(1) == weight.shape[0]:
            return weight
        # Only convert the packed layout when it is still in the canonical non-transposed form.
        return weight.t().contiguous()
    return weight


def _dequantize_nf4_blockwise(weight, quant_state, dtype):
    import bitsandbytes.functional as bnb_F

    weight = _normalize_nf4_weight_layout(weight, quant_state)
    return bnb_F.dequantize_4bit(
        weight,
        quant_state=quant_state,
        absmax=None,
        out=None,
        blocksize=int(quant_state.blocksize),
        quant_type=getattr(quant_state, "quant_type", "nf4"),
    ).to(dtype)


def _quantized_nf4_to_dense_blockwise(weight, quant_state, dtype=torch.float16):
    code = _nf4_codebook_from_quant_state(quant_state).to(device=weight.device, dtype=dtype)
    absmax = _nf4_absmax_from_quant_state(quant_state).to(device=weight.device, dtype=dtype)
    N, K = quant_state.shape
    blocksize = int(quant_state.blocksize)
    n_blocks = (K + blocksize - 1) // blocksize

    packed = weight.contiguous().view(N, K // 2).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)

    vals = torch.empty(N, K // 2, 2, device=weight.device, dtype=dtype)
    vals[..., 0] = code[hi]
    vals[..., 1] = code[lo]
    deq = vals.flatten(1)
    block_idx = torch.arange(K, device=weight.device) // blocksize
    scales = absmax[:n_blocks].view(1, n_blocks).expand(N, n_blocks)
    scale_per_element = scales.gather(1, block_idx.view(1, -1).expand(N, K)).to(dtype)
    out = deq * scale_per_element
    return out.view(N, K)


class TritonNF4MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, quant_state, out=None, bias=None):
        ctx.quant_state = quant_state
        ctx.has_bias = bias is not None
        ctx.save_for_backward(B)

        A = A.contiguous()
        B = B.contiguous()

        N, K = quant_state.shape
        M = A.shape[-2]
        out_shape = (M, N)

        if out is None:
            out = torch.empty(out_shape, device=A.device, dtype=A.dtype)
        else:
            out.copy_(torch.empty_like(out))

        w_deq = _dequantize_nf4_blockwise(B, quant_state, dtype=A.dtype)
        out.copy_(torch.matmul(A, w_deq))

        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (B,) = ctx.saved_tensors
        quant_state = ctx.quant_state
        w_deq = _dequantize_nf4_blockwise(B, quant_state, dtype=grad_output.dtype)
        grad_A = torch.matmul(grad_output, w_deq.transpose(0, 1))
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=(0, 1))
        return grad_A, None, None, grad_bias, None


_DUMMY_TENSOR = None
_DEQ_SCRATCH = {}
_SCRATCH_MIN = 256 * 1024


def _get_dummy(device):
    global _DUMMY_TENSOR
    if _DUMMY_TENSOR is None or _DUMMY_TENSOR.device != device:
        _DUMMY_TENSOR = torch.zeros(1, dtype=torch.float32, device=device)
    return _DUMMY_TENSOR


def _get_deq_scratch(shape, dtype, device):
    key = (device, dtype, tuple(shape))
    numel = int(torch.Size(shape).numel())
    buf = _DEQ_SCRATCH.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(max(numel, _SCRATCH_MIN), dtype=dtype, device=device)
        _DEQ_SCRATCH[key] = buf
    return buf[:numel].view(shape)


def _recover_nf4_scales(quant_state, n_blocks, device):
    absmax = quant_state.absmax
    if getattr(quant_state, "nested", False):
        state2 = quant_state.state2
        q_abs = absmax.to(device=device, dtype=torch.long).reshape(-1)
        s2_abs = state2.absmax.to(device=device, dtype=torch.float32).reshape(-1)
        s2_code = state2.code.to(device=device, dtype=torch.float32).reshape(-1)
        group_idx = torch.arange(n_blocks, device=device, dtype=torch.long) // int(state2.blocksize)
        offset = quant_state.offset
        if offset is None:
            offset = 0.0
        offset = torch.as_tensor(float(offset), device=device, dtype=torch.float32)
        return s2_code[q_abs] * s2_abs[group_idx] + offset
    return absmax.to(device=device, dtype=torch.float32).reshape(-1)


def triton_dequantize_nf4(weight, quant_state, out=None, dtype=None):
    out_shape = quant_state.shape
    out_dtype = dtype if dtype is not None else quant_state.dtype
    total = int(out_shape.numel())
    blocksize = int(quant_state.blocksize)
    n_blocks = (total + blocksize - 1) // blocksize

    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=weight.device)
    else:
        out = out.to(device=weight.device, dtype=out_dtype)

    # Exact match with bitsandbytes; do not re-interpret the already-correct layout.
    import bitsandbytes.functional as bnb_F

    ref = bnb_F.dequantize_4bit(
        weight,
        quant_state=quant_state,
        absmax=None,
        out=out,
        blocksize=blocksize,
        quant_type=getattr(quant_state, "quant_type", "nf4"),
    )
    return ref


def _dequantize_nf4_scratch(weight, quant_state, dtype):
    scratch = _get_deq_scratch(quant_state.shape, dtype, weight.device)
    return triton_dequantize_nf4(weight, quant_state, out=scratch, dtype=dtype)


class TritonMatMul4Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        ctx.quant_state = quant_state
        ctx.has_bias = bias is not None
        ctx.save_for_backward(B)

        B_norm = _normalize_nf4_weight_layout(B, quant_state)
        w_deq = _dequantize_nf4_blockwise(B_norm, quant_state, dtype=A.dtype)
        ctx.w_deq = w_deq.detach().contiguous()
        bias_c = bias.to(A.dtype) if (bias is not None and bias.dtype != A.dtype) else bias

        output = torch.matmul(A, w_deq)
        if bias_c is not None:
            output = output + bias_c

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
            if ctx.w_deq is not None:
                grad_A = torch.matmul(grad_output, ctx.w_deq.transpose(-1, -2))
            else:
                B_norm = _normalize_nf4_weight_layout(B, ctx.quant_state)
                w_deq = _dequantize_nf4_blockwise(B_norm, ctx.quant_state, dtype=grad_output.dtype)
                grad_A = torch.matmul(grad_output, w_deq.transpose(-1, -2))
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
            try:
                return TritonNF4MatMul.apply(A, B, quant_state, out, bias)
            except Exception:
                return original_matmul_4bit(A, B, out=out, bias=bias, quant_state=quant_state)
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
