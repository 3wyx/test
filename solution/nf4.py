"""NF4 -> fp16 dequantization of the frozen 4-bit base weights (our own Triton kernel).

Every linear layer of the model keeps its weight as packed 4-bit NF4 (two values per byte, the
first value in the high nibble) with one fp32 scale per block of 64 values. For every matmul,
forward and backward, bitsandbytes first expands the weight to a dense fp16 matrix: ~390 such
expansions per training step. This kernel does the same expansion: value = NF4_CODE[nibble] *
absmax[block]. The format is the public NF4 data type of QLoRA (Dettmers et al., 2023); the
kernel is written from that definition, not copied from bitsandbytes' CUDA code.
"""

import torch
import triton
import triton.language as tl

NF4_CODE = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141506195, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039558105, 0.7229568362236999, 1.0)
BLOCKSIZE = 64
BLOCK = 1024
NUM_WARPS = 4

_codes = {}


@triton.jit
def _nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, out_ptr, n_out, n_bytes,
                        BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n_bytes
    packed = tl.load(packed_ptr + offs, mask=in_range, other=0).to(tl.int32)
    scale = tl.load(absmax_ptr + offs // 32, mask=in_range, other=0.0)
    high = tl.load(code_ptr + (packed >> 4)) * scale
    low = tl.load(code_ptr + (packed & 15)) * scale
    pair = tl.arange(0, 2)[None, :]
    vals = tl.where(pair == 0, high[:, None], low[:, None])
    out_offs = offs[:, None] * 2 + pair
    tl.store(out_ptr + out_offs, vals.to(out_ptr.dtype.element_ty), mask=out_offs < n_out)


def _code_table(device):
    table = _codes.get(device)
    if table is None:
        table = _codes[device] = torch.tensor(NF4_CODE, dtype=torch.float32, device=device)
    return table


def dequantize_nf4(packed, absmax, out):
    n_out = out.numel()
    n_bytes = (n_out + 1) // 2
    grid = (triton.cdiv(n_bytes, BLOCK),)
    _nf4_dequant_kernel[grid](packed, absmax, _code_table(out.device), out, n_out, n_bytes,
                              BLOCK=BLOCK, num_warps=NUM_WARPS)


def make_dequant(original):
    """Replacement of bitsandbytes' CUDA `_dequantize_4bit_impl` (same signature)."""

    def dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out):
        if (quant_type == "nf4" and blocksize == BLOCKSIZE and torch.is_tensor(A) and A.is_cuda
                and out.is_cuda and out.is_contiguous() and absmax.dtype == torch.float32
                and out.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            dequantize_nf4(A.contiguous(), absmax.contiguous(), out)
            return None
        return original(A, absmax, blocksize, quant_type, dtype, out)

    dequantize_4bit_impl._is_fused = True
    return dequantize_4bit_impl
