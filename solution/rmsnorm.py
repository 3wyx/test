"""Слитый RMSNorm (forward + backward) для Qwen2RMSNorm, свой Triton-кернел."""

import torch  # основной пакет
import triton  # компилятор кернелов
import triton.language as tl  # язык кернелов


@triton.jit
def _rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # одна программа = одна строка (один токен)
    cols = tl.arange(0, BLOCK)  # индексы столбцов внутри строки
    mask = cols < N  # маска: BLOCK больше N, лишнее не читаем
    base = row.to(tl.int64) * N  # смещение начала строки (int64, чтобы не переполнилось)
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)  # читаем строку в fp32
    var = tl.sum(x * x, 0) / N  # средний квадрат (дисперсия без среднего)
    rstd = 1.0 / tl.sqrt(var + eps)  # 1 / sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # вес нормы в fp32
    xn = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)  # как в оригинале: округление до типа входа
    y = xn * w  # умножение на вес
    tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=mask)  # пишем результат
    tl.store(rstd_ptr + row, rstd)  # сохраняем rstd для backward


@triton.jit
def _rmsnorm_bwd_kernel(x_ptr, w_ptr, g_ptr, rstd_ptr, dx_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # одна программа = одна строка
    cols = tl.arange(0, BLOCK)  # индексы столбцов
    mask = cols < N  # маска лишних столбцов
    base = row.to(tl.int64) * N  # начало строки
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)  # вход
    g = tl.load(g_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)  # градиент выхода
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # вес
    rstd = tl.load(rstd_ptr + row)  # сохранённый rstd
    xhat = x * rstd  # нормированное значение
    gw = g * w  # градиент по xhat
    c = tl.sum(gw * xhat, 0) / N  # средняя проекция градиента на xhat
    dx = rstd * (gw - xhat * c)  # формула градиента RMSNorm
    tl.store(dx_ptr + base + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)  # пишем градиент входа


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype=None):
        shape = x.shape
        n = shape[-1]
        x2 = x.reshape(-1, n).contiguous()
        rows = x2.shape[0]
        if out_dtype is None:
            out_dtype = torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty((rows, n), dtype=out_dtype, device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(n)
        warps = 8 if block >= 4096 else 4
        _rmsnorm_fwd_kernel[(rows,)](x2, weight.contiguous(), y, rstd, n, eps,
                                     BLOCK=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2, weight, rstd = ctx.saved_tensors
        rows, n = x2.shape
        g = grad_out.reshape(-1, n).contiguous()
        dx = torch.empty_like(x2)
        block = triton.next_power_of_2(n)
        warps = 8 if block >= 4096 else 4
        _rmsnorm_bwd_kernel[(rows,)](x2, weight.contiguous(), g, rstd, dx, n,
                                     BLOCK=block, num_warps=warps)
        dw = None
        if ctx.needs_input_grad[1]:
            xn = (x2.float() * rstd[:, None]).to(x2.dtype).float()
            dw = (g.float() * xn).sum(0).to(weight.dtype)
        return dx.view(ctx.shape), dw, None, None


def make_forward(original):
    def forward(self, hidden_states):
        w = self.weight
        if (hidden_states.is_cuda and w.is_cuda
                and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and w.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            out_dtype = torch.promote_types(hidden_states.dtype, w.dtype)
            if out_dtype == torch.float32 and torch.is_autocast_enabled():
                out_dtype = torch.get_autocast_dtype("cuda")
            return _RMSNorm.apply(hidden_states, w, self.variance_epsilon, out_dtype)
        return original(self, hidden_states)

    forward._is_fused = True
    return forward

_OK = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _add_rmsnorm_fwd_kernel(a_ptr, b_ptr, w_ptr, s_ptr, y_ptr, rstd_ptr, N, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row.to(tl.int64) * N
    a = tl.load(a_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    s = (a + b).to(s_ptr.dtype.element_ty)
    tl.store(s_ptr + base + cols, s, mask=mask)
    x = s.to(tl.float32)
    var = tl.sum(x * x, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xn = (x * rstd).to(s_ptr.dtype.element_ty).to(tl.float32)
    tl.store(y_ptr + base + cols, (xn * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _add_rmsnorm_bwd_kernel(s_ptr, w_ptr, gy_ptr, gs_ptr, rstd_ptr, da_ptr, db_ptr, N,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row.to(tl.int64) * N
    x = tl.load(s_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    gy = tl.load(gy_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    gs = tl.load(gs_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    gw = gy * w
    c = tl.sum(gw * xhat, 0) / N
    d = gs + rstd * (gw - xhat * c)
    tl.store(da_ptr + base + cols, d.to(da_ptr.dtype.element_ty), mask=mask)
    tl.store(db_ptr + base + cols, d.to(db_ptr.dtype.element_ty), mask=mask)


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, weight, eps, out_dtype):
        shape = a.shape
        n = shape[-1]
        a2 = a.reshape(-1, n)
        b2 = b.reshape(-1, n)
        rows = a2.shape[0]
        s = torch.empty((rows, n), dtype=torch.promote_types(a.dtype, b.dtype), device=a.device)
        y = torch.empty((rows, n), dtype=out_dtype, device=a.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=a.device)
        block = triton.next_power_of_2(n)
        warps = 8 if block >= 4096 else 4
        _add_rmsnorm_fwd_kernel[(rows,)](a2, b2, weight.contiguous(), s, y, rstd, n, eps,
                                         BLOCK=block, num_warps=warps)
        ctx.save_for_backward(s, weight, rstd)
        ctx.shape = shape
        ctx.a_dtype = a.dtype
        ctx.b_dtype = b.dtype
        return s.view(shape), y.view(shape)

    @staticmethod
    def backward(ctx, grad_s, grad_y):
        s, weight, rstd = ctx.saved_tensors
        rows, n = s.shape
        gs = grad_s.reshape(-1, n).contiguous()
        gy = grad_y.reshape(-1, n).contiguous()
        da = torch.empty((rows, n), dtype=ctx.a_dtype, device=s.device)
        db = torch.empty((rows, n), dtype=ctx.b_dtype, device=s.device)
        block = triton.next_power_of_2(n)
        warps = 8 if block >= 4096 else 4
        _add_rmsnorm_bwd_kernel[(rows,)](s, weight.contiguous(), gy, gs, rstd, da, db, n,
                                         BLOCK=block, num_warps=warps)
        dw = None
        if ctx.needs_input_grad[2]:
            xn = (s.float() * rstd[:, None]).to(s.dtype).float()
            dw = (gy.float() * xn).sum(0).to(weight.dtype)
        return da.view(ctx.shape), db.view(ctx.shape), dw, None, None


def add_norm(residual, branch, norm):
    w = norm.weight
    if not (residual.is_cuda and branch.is_cuda and w.is_cuda
            and residual.shape == branch.shape
            and residual.dtype in _OK and branch.dtype in _OK and w.dtype in _OK):
        return None
    s_dtype = torch.promote_types(residual.dtype, branch.dtype)
    out_dtype = torch.promote_types(s_dtype, w.dtype)
    if out_dtype == torch.float32 and torch.is_autocast_enabled():
        out_dtype = torch.get_autocast_dtype("cuda")
    return _AddRMSNorm.apply(residual.contiguous(), branch.contiguous(), w,
                             norm.variance_epsilon, out_dtype)


def make_layer_forward(original):
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        fused = add_norm(residual, hidden_states, self.post_attention_layernorm)
        if fused is None:
            residual = residual + hidden_states
            normed = self.post_attention_layernorm(residual)
        else:
            residual, normed = fused
        return residual + self.mlp(normed)

    forward._is_fused = True
    return forward
