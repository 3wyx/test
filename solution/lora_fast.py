import torch


def make_forward(original):
    def forward(self, x, *args, **kwargs):
        fast = (torch.is_autocast_enabled() and not args and not kwargs
                and not self.disable_adapters and not self.merged
                and len(self.active_adapters) == 1)
        if fast:
            name = self.active_adapters[0]
            fast = name in self.lora_A and name not in self.lora_variant
        if fast:
            lora_A = self.lora_A[name]
            lora_B = self.lora_B[name]
            fast = lora_A.bias is None and lora_B.bias is None
        if not fast:
            return original(self, x, *args, **kwargs)

        result = self.base_layer(x)
        scaling = self.scaling[name]
        hidden = lora_A(self.lora_dropout[name](x))
        if result.dtype != hidden.dtype:
            return result + lora_B(hidden) * scaling
        flat = torch.addmm(result.reshape(-1, result.shape[-1]),
                           hidden.reshape(-1, hidden.shape[-1]),
                           lora_B.weight.t(), alpha=scaling)
        return flat.view(result.shape)

    forward._is_fused = True
    return forward
