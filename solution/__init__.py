"""Soup Kernel Challenge solution entrypoint."""

import sys

import torch

from .cross_entropy import patch_chunked_loss
from .nf4 import patch_nf4_dequant
from .rmsnorm import patch_qwen2_rmsnorm
from .rope import patch_qwen2_rope
from .swiglu import patch_qwen2_mlp


def _warn(what, exc):
    print(f"[solution] patch skipped: {what}: {exc!r}", file=sys.stderr)


def _patch_peft_lora_dtype():
    from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit

    orig_cast = getattr(PeftLinear4bit, "_cast_input_dtype", None)
    if orig_cast is None or getattr(orig_cast, "_is_fused", False):
        return

    def fast_cast_input_dtype(self, x, dtype):
        if (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.dtype == torch.float16
            and dtype == torch.float32
            and torch.is_autocast_enabled()
        ):
            return x
        return orig_cast(self, x, dtype)

    fast_cast_input_dtype._is_fused = True
    PeftLinear4bit._cast_input_dtype = fast_cast_input_dtype


def _patch_frozen_fp16_params():
    from soup_cli.trainer.sft import SFTTrainerWrapper

    original_setup = SFTTrainerWrapper._setup_transformers
    if getattr(original_setup, "_is_fused", False):
        return

    def _setup_transformers_patched(self, *args, **kwargs):
        result = original_setup(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if not isinstance(model, torch.nn.Module) or not torch.cuda.is_available():
            return result
        for param in model.parameters():
            if (
                not param.requires_grad
                and param.is_cuda
                and param.dtype == torch.float32
                and param.__class__.__name__ != "Params4bit"
            ):
                param.data = param.data.to(torch.float16)
        return result

    _setup_transformers_patched._is_fused = True
    SFTTrainerWrapper._setup_transformers = _setup_transformers_patched


def apply():
    for name, patch in (
        ("compact model + fused MLP", patch_qwen2_mlp),
        ("RMSNorm", patch_qwen2_rmsnorm),
        ("RoPE", patch_qwen2_rope),
        ("chunked LM-head loss", patch_chunked_loss),
        ("NF4 dequant", patch_nf4_dequant),
        ("peft LoRA input dtype", _patch_peft_lora_dtype),
        ("frozen fp32 params to fp16", _patch_frozen_fp16_params),
    ):
        try:
            patch()
        except Exception as exc:
            _warn(name, exc)