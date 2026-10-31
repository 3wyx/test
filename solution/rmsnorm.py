"""Fused Qwen2 RMSNorm forward and backward kernels in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def qwen2_rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    stride_x_row,
    stride_y_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_row_ptr = x_ptr + row_idx * stride_x_row
    y_row_ptr = y_ptr + row_idx * stride_y_row

    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    tl.store(rstd_ptr + row_idx, rstd)
    y = (x * rstd) * w
    tl.store(y_row_ptr + col_offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qwen2_rmsnorm_bwd_kernel(
    dy_ptr,
    xy_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    stride_dy_row,
    stride_xy_row,
    stride_dx_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    RECONSTRUCT_X: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dy = tl.load(dy_ptr + row_idx * stride_dy_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    xy = tl.load(xy_ptr + row_idx * stride_xy_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

    if RECONSTRUCT_X:
        safe_w = tl.where(w == 0.0, 1.0, w)
        x_hat = tl.where(mask, xy / safe_w, 0.0)
    else:
        x_hat = tl.where(mask, xy * rstd, 0.0)
    dy_w = tl.where(mask, dy * w, 0.0)
    mean_dy_w_xhat = tl.sum(dy_w * x_hat, axis=0) / n_cols
    dx = rstd * (dy_w - x_hat * mean_dy_w_xhat)

    tl.store(
        dx_ptr + row_idx * stride_dx_row + col_offsets,
        dx.to(dx_ptr.dtype.element_ty),
        mask=mask,
    )


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, reconstruct_x=False):
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x_2d.shape

        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size <= 1024 else 8

        y_2d = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        qwen2_rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            weight,
            y_2d,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            n_cols,
            float(eps),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        use_reconstruct = bool(reconstruct_x and not weight.requires_grad)
        ctx.reconstruct_x = use_reconstruct
        if use_reconstruct:
            ctx.save_for_backward(y_2d, weight, rstd)
        else:
            ctx.save_for_backward(x_2d, weight, rstd)
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        xy_2d, weight, rstd = ctx.saved_tensors
        orig_shape = dy.shape
        dy_2d = dy.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = dy_2d.shape

        dw = None
        if weight.requires_grad:
            x_hat = xy_2d.float() * rstd.unsqueeze(1)
            dw = (dy_2d.float() * x_hat).sum(dim=0).to(weight.dtype)
            dx_2d = torch.empty_like(xy_2d)
        else:
            dx_2d = dy_2d if dy_2d.dtype == xy_2d.dtype else torch.empty_like(xy_2d)

        qwen2_rmsnorm_bwd_kernel[(n_rows,)](
            dy_2d,
            xy_2d,
            weight,
            rstd,
            dx_2d,
            dy_2d.stride(0),
            xy_2d.stride(0),
            dx_2d.stride(0),
            n_cols,
            BLOCK_SIZE=ctx.block_size,
            num_warps=ctx.num_warps,
            RECONSTRUCT_X=ctx.reconstruct_x,
        )

        return dx_2d.view(orig_shape), dw, None, None


def rmsnorm(x, weight, eps=1e-6, reconstruct_x=False):
    return TritonRMSNorm.apply(x, weight, eps, reconstruct_x)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original_forward = modeling_qwen2.Qwen2RMSNorm.forward
    if getattr(original_forward, "_is_fused", False):
        return

    def forward(self, hidden_states):
        if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
            return original_forward(self, hidden_states)
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        return rmsnorm(
            hidden_states,
            self.weight,
            eps,
            reconstruct_x=not self.weight.requires_grad,
        )

    forward._is_fused = True
    modeling_qwen2.Qwen2RMSNorm.forward = forward