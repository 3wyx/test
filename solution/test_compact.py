import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_compact_forward_matches_original():
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution.compact import make_forward

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = modeling_qwen2.Qwen2Model(cfg).cuda().half().train()
    ids = torch.randint(0, 256, (3, 40), device="cuda")
    mask = torch.ones_like(ids)
    mask[1, 25:] = 0
    mask[2, 33:] = 0
    valid = mask.bool()

    original = modeling_qwen2.Qwen2Model.forward
    fast = make_forward(original)

    ref = original(model, input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    ref[valid].float().sum().backward()
    ref_grad = model.embed_tokens.weight.grad.clone()
    model.zero_grad()

    out = fast(model, input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    out[valid].float().sum().backward()
    grad = model.embed_tokens.weight.grad.clone()

    torch.testing.assert_close(out[valid].float(), ref[valid].float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(grad.float(), ref_grad.float(), rtol=5e-2, atol=5e-2)
