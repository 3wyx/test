"""Fused SwiGLU, fused RMSNorm+MLP and whole-model active-token compaction for Qwen2."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint

from .nf4 import _dequantize_nf4_scratch
from .rmsnorm import qwen2_rmsnorm_bwd_kernel, qwen2_rmsnorm_fwd_kernel
from .rope import _CompactRoPESDPA

BLOCK = 1024
RECOMPUTE_MLP = False


@triton.jit
def example_swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def example_swiglu_bwd_kernel(grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(dgate_ptr + offsets, dgate.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offsets, dup.to(dup_ptr.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        grad = grad.to(gate.dtype).contiguous()
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate, up, gate, up, n, BLOCK=BLOCK)
        return gate, up


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


class _ScatterActive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out_valid, active_idx, n_rows):
        ctx.save_for_backward(active_idx)
        out = torch.zeros((n_rows, out_valid.shape[1]), dtype=out_valid.dtype, device=out_valid.device)
        out.index_copy_(0, active_idx, out_valid)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (active_idx,) = ctx.saved_tensors
        return grad_out.index_select(0, active_idx), None, None


def _cast_bias(bias, dtype):
    if bias is not None and bias.dtype != dtype:
        return bias.to(dtype)
    return bias


class _FusedNormMLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        norm_w,
        norm_eps,
        gate_w,
        gate_qs,
        gate_bias,
        up_w,
        up_qs,
        up_bias,
        down_w,
        down_qs,
        down_bias,
        recompute,
    ):
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x_2d.shape
        dtype = x_2d.dtype

        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size <= 1024 else 8

        x_norm = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        qwen2_rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            norm_w,
            x_norm,
            rstd,
            x_2d.stride(0),
            x_norm.stride(0),
            n_cols,
            float(norm_eps),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        gate = F.linear(x_norm, _dequantize_nf4_scratch(gate_w, gate_qs, dtype=dtype), _cast_bias(gate_bias, dtype))
        up = F.linear(x_norm, _dequantize_nf4_scratch(up_w, up_qs, dtype=dtype), _cast_bias(up_bias, dtype))
        del x_norm

        hidden = torch.empty_like(gate)
        n_elem = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n_elem, BLOCK),)](gate, up, hidden, n_elem, BLOCK=BLOCK)
        if recompute:
            gate = None
            up = None

        out = F.linear(hidden, _dequantize_nf4_scratch(down_w, down_qs, dtype=dtype), _cast_bias(down_bias, dtype))
        del hidden

        ctx.save_for_backward(x_2d, norm_w, rstd, gate_w, up_w, down_w, gate, up)
        ctx.gate_bias = gate_bias
        ctx.up_bias = up_bias
        ctx.norm_eps = float(norm_eps)
        ctx.gate_qs = gate_qs
        ctx.up_qs = up_qs
        ctx.down_qs = down_qs
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_2d, norm_w, rstd, gate_w, up_w, down_w, gate, up = ctx.saved_tensors
        orig_shape = grad_out.shape
        g_2d = grad_out.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x_2d.shape
        dtype = g_2d.dtype

        if gate is None:
            x_norm = torch.empty_like(x_2d)
            qwen2_rmsnorm_fwd_kernel[(n_rows,)](
                x_2d,
                norm_w,
                x_norm,
                rstd,
                x_2d.stride(0),
                x_norm.stride(0),
                n_cols,
                ctx.norm_eps,
                BLOCK_SIZE=ctx.block_size,
                num_warps=ctx.num_warps,
            )
            gate = F.linear(
                x_norm, _dequantize_nf4_scratch(gate_w, ctx.gate_qs, dtype=dtype), _cast_bias(ctx.gate_bias, dtype)
            )
            up = F.linear(
                x_norm, _dequantize_nf4_scratch(up_w, ctx.up_qs, dtype=dtype), _cast_bias(ctx.up_bias, dtype)
            )
            del x_norm

        d_hidden = torch.matmul(g_2d, _dequantize_nf4_scratch(down_w, ctx.down_qs, dtype=dtype))

        n_elem = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n_elem, BLOCK),)](d_hidden, gate, up, gate, up, n_elem, BLOCK=BLOCK)
        del d_hidden

        dx_norm = torch.matmul(up, _dequantize_nf4_scratch(up_w, ctx.up_qs, dtype=dtype))
        del up
        dx_norm.addmm_(gate, _dequantize_nf4_scratch(gate_w, ctx.gate_qs, dtype=dtype))
        del gate

        qwen2_rmsnorm_bwd_kernel[(n_rows,)](
            dx_norm,
            x_2d,
            norm_w,
            rstd,
            dx_norm,
            dx_norm.stride(0),
            x_2d.stride(0),
            dx_norm.stride(0),
            n_cols,
            BLOCK_SIZE=ctx.block_size,
            num_warps=ctx.num_warps,
            RECONSTRUCT_X=False,
        )

        return (dx_norm.view(orig_shape),) + (None,) * 12


def _is_frozen_nf4_proj(proj):
    if hasattr(proj, "lora_A") and not getattr(proj, "disable_adapters", False):
        if len(getattr(proj, "active_adapters", [])) > 0:
            return False
    w = getattr(proj, "weight", None)
    qs = getattr(w, "quant_state", None)
    return (
        w is not None
        and qs is not None
        and not w.requires_grad
        and w.dtype == torch.uint8
        and getattr(qs, "quant_type", None) == "nf4"
        and getattr(qs, "blocksize", 0) in (32, 64, 128)
        and qs.shape.numel() % 2 == 0
    )


def _can_use_fused_norm_mlp(mlp_mod, norm_mod):
    act = getattr(getattr(mlp_mod, "config", None), "hidden_act", "silu")
    if act != "silu":
        return False
    norm_w = getattr(norm_mod, "weight", None)
    if norm_w is None or norm_w.requires_grad:
        return False
    return (
        _is_frozen_nf4_proj(mlp_mod.gate_proj)
        and _is_frozen_nf4_proj(mlp_mod.up_proj)
        and _is_frozen_nf4_proj(mlp_mod.down_proj)
    )


def _get_orig_sdpa(modeling_qwen2):
    funcs = getattr(modeling_qwen2, "ALL_ATTENTION_FUNCTIONS", None)
    fn = None
    if funcs is not None and "sdpa" in funcs:
        fn = funcs["sdpa"]
        fn = getattr(fn, "_orig_sdpa", fn)
    if fn is None:
        try:
            from transformers.integrations.sdpa_attention import sdpa_attention_forward

            fn = sdpa_attention_forward
        except Exception:
            fn = None
    return fn


def _eligible(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, kwargs):
    if use_cache is None:
        use_cache = getattr(model.config, "use_cache", False)
    return (
        isinstance(input_ids, torch.Tensor)
        and input_ids.is_cuda
        and input_ids.dim() == 2
        and inputs_embeds is None
        and isinstance(attention_mask, torch.Tensor)
        and attention_mask.shape == input_ids.shape
        and position_ids is None
        and past_key_values is None
        and not use_cache
        and not kwargs.get("output_attentions", False)
        and not kwargs.get("output_hidden_states", False)
        and getattr(model.config, "_attn_implementation", "sdpa") == "sdpa"
        and not getattr(model, "has_sliding_layers", False)
        and not (getattr(model, "gradient_checkpointing", False) and model.training)
    )


def patch_qwen2_mlp():
    from transformers.models.qwen2 import modeling_qwen2

    orig_model_fwd = modeling_qwen2.Qwen2Model.forward
    if getattr(orig_model_fwd, "_is_fused", False):
        return

    def compact_forward(self, input_ids, mask_bool, orig_sdpa):
        bsz, seq_len = input_ids.shape
        right_padded, longest = torch.stack(
            [(mask_bool[:, 1:] <= mask_bool[:, :-1]).all().to(torch.long), mask_bool.sum(1).max()]
        ).tolist()
        if not right_padded or longest == 0:
            return None
        s_max = int(longest)

        keep = mask_bool[:, :s_max].reshape(-1)
        idx_smax = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        idx_full = (idx_smax // s_max) * seq_len + idx_smax % s_max
        flat_ids = input_ids[:, :s_max].reshape(-1).index_select(0, idx_smax)

        hidden_states = self.embed_tokens(flat_ids)
        if hidden_states.dtype == torch.float32:
            hidden_states = hidden_states.to(torch.float16)

        pos_1d = idx_smax % s_max
        cos_valid, sin_valid = self.rotary_emb(hidden_states, pos_1d.unsqueeze(0))

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            residual = hidden_states
            h_norm = decoder_layer.input_layernorm(hidden_states)

            attn = decoder_layer.self_attn
            q_valid = attn.q_proj(h_norm)
            k_valid = attn.k_proj(h_norm)
            v_valid = attn.v_proj(h_norm)
            del h_norm

            head_dim = attn.head_dim
            n_qh = q_valid.shape[-1] // head_dim
            n_kh = k_valid.shape[-1] // head_dim

            attn_valid = _CompactRoPESDPA.apply(
                q_valid,
                k_valid,
                v_valid,
                cos_valid,
                sin_valid,
                idx_smax,
                None,
                bsz,
                s_max,
                n_qh,
                n_kh,
                head_dim,
                attn,
                orig_sdpa,
            )
            del q_valid, k_valid, v_valid

            attn_out = attn.o_proj(attn_valid)
            del attn_valid
            hidden_states = residual + attn_out
            del residual, attn_out

            residual = hidden_states
            norm_mod = decoder_layer.post_attention_layernorm
            mlp_mod = decoder_layer.mlp
            if _can_use_fused_norm_mlp(mlp_mod, norm_mod):
                norm_eps = getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-6))
                gw = mlp_mod.gate_proj.weight
                uw = mlp_mod.up_proj.weight
                dw = mlp_mod.down_proj.weight
                mlp_out = _FusedNormMLP.apply(
                    hidden_states,
                    norm_mod.weight,
                    norm_eps,
                    gw,
                    gw.quant_state,
                    getattr(mlp_mod.gate_proj, "bias", None),
                    uw,
                    uw.quant_state,
                    getattr(mlp_mod.up_proj, "bias", None),
                    dw,
                    dw.quant_state,
                    getattr(mlp_mod.down_proj, "bias", None),
                    RECOMPUTE_MLP,
                )
            else:
                mlp_out = mlp_mod(norm_mod(hidden_states))

            hidden_states = residual + mlp_out
            del residual, mlp_out

        hidden_states = self.norm(hidden_states)
        full_hidden = _ScatterActive.apply(hidden_states, idx_full, bsz * seq_len).view(bsz, seq_len, -1)
        return modeling_qwen2.BaseModelOutputWithPast(last_hidden_state=full_hidden, past_key_values=None)

    def fast_qwen2_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if _eligible(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, kwargs):
            orig_sdpa = _get_orig_sdpa(modeling_qwen2)
            if orig_sdpa is not None:
                result = compact_forward(self, input_ids, attention_mask != 0, orig_sdpa)
                if result is not None:
                    return result
        return orig_model_fwd(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    fast_qwen2_model_forward._is_fused = True
    modeling_qwen2.Qwen2Model.forward = fast_qwen2_model_forward

    original_mlp = modeling_qwen2.Qwen2MLP.forward

    def mlp_forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not isinstance(x, torch.Tensor) or not x.is_cuda:
            return original_mlp(self, x)

        def _gate_up_swiglu(inp):
            return swiglu(self.gate_proj(inp), self.up_proj(inp))

        if RECOMPUTE_MLP and torch.is_grad_enabled() and x.requires_grad:
            hidden_act = checkpoint(_gate_up_swiglu, x, use_reentrant=False)
        else:
            hidden_act = _gate_up_swiglu(x)
        return self.down_proj(hidden_act)

    modeling_qwen2.Qwen2MLP.forward = mlp_forward