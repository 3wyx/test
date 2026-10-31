"""Where the fused loss (fused_loss.py) is plugged into transformers, TRL and Soup.

1. transformers: LOSS_MAPPING["ForCausalLM"] -> our loss (one kernel pass instead of an fp32
   copy + cross-entropy + softmax gradient buffers).
2. TRL SFTTrainer.compute_loss: the same function as TRL's, except that entropy and token
   accuracy are read from the kernel's per-token results instead of two more passes over the
   logits (entropy_from_logits, and a contiguous copy of the shifted logits + argmax).
   It falls back to TRL's own code whenever its assumptions do not hold.
3. Soup rewind_hf.row_losses: the per-row losses for rewind.jsonl are summed from the same
   per-token results instead of a cross-entropy per row.

Every patch is guarded: a missing or faked module (Soup's regression tests use some) is skipped.
"""

import sys
from collections.abc import Mapping

import torch

from . import fused_loss, lora_fast, nf4, rmsnorm, rope


def _warn(what, exc):
    print(f"[solution] patch skipped: {what}: {exc!r}", file=sys.stderr)


def patch_transformers_loss():
    from transformers.loss import loss_utils

    original = loss_utils.LOSS_MAPPING["ForCausalLM"]
    if getattr(original, "_is_fused", False):
        return
    fused = fused_loss.make_loss(original)
    fused._is_fused = True
    loss_utils.LOSS_MAPPING["ForCausalLM"] = fused


def patch_trl_compute_loss():
    from trl.trainer import sft_trainer as sft_module

    trainer_cls = sft_module.SFTTrainer
    original = trainer_cls.compute_loss
    if getattr(original, "_is_fused", False):
        return

    def fast_path_ok(self, inputs):
        return (self.model.training and not self.args.use_liger_kernel
                and "shift_labels" not in inputs and "attention_mask" in inputs
                and "labels" in inputs and getattr(self, "num_virtual_tokens", 0) == 0
                and not getattr(self, "aux_loss_enabled", False))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        fused_loss._SIDE.clear()  # whatever is cached must belong to the forward pass below
        if not fast_path_ok(self, inputs):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train"
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False
        loss, outputs = super(trainer_cls, self).compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            side = fused_loss.lookup(outputs.logits)
            if side is not None:
                per_token_entropy = side["entropy"]
                valid = labels[:, 1:] != -100
                correct_tokens = ((side["correct"][:, :-1] != 0) & valid).sum()
                total_tokens = valid.sum()
            else:  # not our loss path: do what TRL does, without the contiguous copy
                per_token_entropy = sft_module.entropy_from_logits(outputs.logits)
                predictions = outputs.logits[:, :-1, :].argmax(dim=-1)
                valid = labels[:, 1:] != -100
                correct_tokens = ((predictions == labels[:, 1:]) & valid).sum()
                total_tokens = valid.sum()
            entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
            entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        num_tokens_in_batch = self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        with torch.no_grad():
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    compute_loss._is_fused = True
    trainer_cls.compute_loss = compute_loss


def patch_soup_row_losses():
    from soup_cli.trainer import rewind_hf

    original = rewind_hf.row_losses
    if getattr(original, "_is_fused", False):
        return

    def row_losses(logits, labels):
        side = fused_loss.lookup(logits)
        if side is None:
            return original(logits, labels)
        return rewind_hf.row_losses_from_token_losses(side["tokloss"], labels)

    row_losses._is_fused = True
    rewind_hf.row_losses = row_losses


def patch_nf4_dequant():
    from bitsandbytes.backends.cuda import ops

    original = ops._dequantize_4bit_impl
    if getattr(original, "_is_fused", False):
        return
    ops._dequantize_4bit_impl = nf4.make_dequant(original)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2 as qwen2

    original = qwen2.Qwen2RMSNorm.forward
    if getattr(original, "_is_fused", False):
        return
    qwen2.Qwen2RMSNorm.forward = rmsnorm.make_forward(original)


def patch_accelerate_fp32_logits():
    from accelerate.utils import operations as acc_ops

    original = acc_ops.convert_to_fp32
    if getattr(original, "_is_fused", False):
        return

    def convert_to_fp32(tensor):
        logits = tensor.get("logits") if isinstance(tensor, Mapping) else None
        keep = (torch.is_tensor(logits) and logits.dtype == torch.float16
                and fused_loss.lookup(logits) is not None)
        if not keep:
            return original(tensor)
        try:
            rest = type(tensor)(**{k: v for k, v in tensor.items() if k != "logits"})
            out = original(rest)
            out["logits"] = logits
            return out
        except Exception:
            return original(tensor)

    convert_to_fp32._is_fused = True
    acc_ops.convert_to_fp32 = convert_to_fp32


def patch_qwen2_decoder_layer():
    from transformers.models.qwen2 import modeling_qwen2 as qwen2

    original = qwen2.Qwen2DecoderLayer.forward
    if getattr(original, "_is_fused", False):
        return
    qwen2.Qwen2DecoderLayer.forward = rmsnorm.make_layer_forward(original)


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2 as qwen2

    original = qwen2.apply_rotary_pos_emb
    if getattr(original, "_is_fused", False):
        return
    qwen2.apply_rotary_pos_emb = rope.make_rope(original)


def patch_lora_forward():
    from peft.tuners.lora import bnb as peft_bnb

    original = peft_bnb.Linear4bit.forward
    if getattr(original, "_is_fused", False):
        return
    peft_bnb.Linear4bit.forward = lora_fast.make_forward(original)


def apply_all():
    for name, patch in (("NF4 dequant", patch_nf4_dequant),
                        ("RMSNorm", patch_qwen2_rmsnorm),
                        ("decoder layer add+norm", patch_qwen2_decoder_layer),
                        ("RoPE", patch_qwen2_rope),
                        ("LoRA fast path", patch_lora_forward),
                        ("accelerate fp32 logits", patch_accelerate_fp32_logits),
                        ("transformers loss", patch_transformers_loss),
                        ("TRL compute_loss", patch_trl_compute_loss),
                        ("Soup row_losses", patch_soup_row_losses)):
        try:
            patch()
        except Exception as exc:
            _warn(name, exc)
