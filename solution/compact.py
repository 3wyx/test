import sys

import torch
import torch.nn.functional as F
from transformers.models.qwen2 import modeling_qwen2

from .rmsnorm import add_norm

_SEEN = [0]


class _Pad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, rows):
        ctx.save_for_backward(idx)
        out = x.new_zeros((rows, x.shape[1]))
        out.index_copy_(0, idx, x)
        return out

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        return grad.index_select(0, idx), None, None


class _Take(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(idx)
        ctx.rows = x.shape[0]
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        out = grad.new_zeros((ctx.rows, grad.shape[1]))
        out.index_copy_(0, idx, grad.contiguous())
        return out, None


def _attention(attn, q, k, v, idx, bsz, s_max, cos, sin):
    d = attn.head_dim
    n_heads = q.shape[1] // d
    n_kv = k.shape[1] // d
    rows = bsz * s_max
    qp = _Pad.apply(q, idx, rows).view(bsz, s_max, n_heads, d).transpose(1, 2)
    kp = _Pad.apply(k, idx, rows).view(bsz, s_max, n_kv, d).transpose(1, 2)
    vp = _Pad.apply(v, idx, rows).view(bsz, s_max, n_kv, d).transpose(1, 2)
    qp, kp = modeling_qwen2.apply_rotary_pos_emb(qp, kp, cos, sin)
    if n_kv != n_heads:
        kp = kp.repeat_interleave(n_heads // n_kv, dim=1)
        vp = vp.repeat_interleave(n_heads // n_kv, dim=1)
    out = F.scaled_dot_product_attention(
        qp, kp, vp, dropout_p=0.0, is_causal=True, scale=getattr(attn, "scaling", d ** -0.5)
    )
    out = out.transpose(1, 2).reshape(rows, n_heads * d)
    return _Take.apply(out, idx)


def _eligible(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
    if use_cache is None:
        use_cache = getattr(model.config, "use_cache", False)
    return (
        model.training
        and torch.is_grad_enabled()
        and input_ids is not None
        and inputs_embeds is None
        and input_ids.is_cuda
        and input_ids.dim() == 2
        and attention_mask is not None
        and attention_mask.shape == input_ids.shape
        and position_ids is None
        and past_key_values is None
        and not use_cache
        and getattr(model.config, "_attn_implementation", "sdpa") == "sdpa"
        and not getattr(model, "gradient_checkpointing", False)
        and not getattr(model, "has_sliding_layers", False)
    )


def make_forward(original):
    def forward(
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
        def fallback():
            return original(
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

        if (
            kwargs.get("output_hidden_states")
            or kwargs.get("output_attentions")
            or not _eligible(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache)
        ):
            return fallback()

        bsz, seq_len = input_ids.shape
        mask = attention_mask != 0
        right_padded = (mask[:, 1:] <= mask[:, :-1]).all()
        info = torch.stack([right_padded.to(torch.long), mask.sum(1).max()]).tolist()
        if not info[0] or info[1] == 0:
            return fallback()
        s_max = int(info[1])

        keep = mask[:, :s_max].reshape(-1)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        idx_full = (idx // s_max) * seq_len + idx % s_max
        if _SEEN[0] < 3:
            _SEEN[0] += 1
            share = 100.0 * idx.numel() / (bsz * seq_len)
            print(f"[compact] active tokens {idx.numel()} of {bsz * seq_len} ({share:.1f}%), s_max {s_max}",
                  file=sys.stderr)

        flat_ids = input_ids[:, :s_max].reshape(-1).index_select(0, idx)
        h = self.embed_tokens(flat_ids)
        pos = torch.arange(s_max, device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(h, pos)
        cos = cos.expand(bsz, -1, -1).contiguous()
        sin = sin.expand(bsz, -1, -1).contiguous()

        for layer in self.layers[: self.config.num_hidden_layers]:
            residual = h
            x = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_proj(x)
            k = attn.k_proj(x)
            v = attn.v_proj(x)
            del x
            a = _attention(attn, q, k, v, idx, bsz, s_max, cos, sin)
            del q, k, v
            o = attn.o_proj(a)
            del a
            fused = add_norm(residual, o, layer.post_attention_layernorm)
            if fused is None:
                h = residual + o
                y = layer.post_attention_layernorm(h)
            else:
                h, y = fused
            del residual, o
            h = h + layer.mlp(y)
            del y

        h = self.norm(h)
        full = _Pad.apply(h, idx_full, bsz * seq_len).view(bsz, seq_len, -1)
        return modeling_qwen2.BaseModelOutputWithPast(last_hidden_state=full, past_key_values=None)

    return forward


def patch_compact():
    original = modeling_qwen2.Qwen2Model.forward
    if getattr(original, "_is_compact", False):
        return
    fast = make_forward(original)
    fast._is_compact = True
    modeling_qwen2.Qwen2Model.forward = fast
