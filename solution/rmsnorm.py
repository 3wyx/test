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
    def forward(ctx, x, weight, eps):
        shape = x.shape  # запоминаем исходную форму
        n = shape[-1]  # размер скрытого слоя
        x2 = x.reshape(-1, n).contiguous()  # плоская матрица (строки, n)
        rows = x2.shape[0]  # число строк
        out_dtype = torch.promote_types(x.dtype, weight.dtype)  # тип выхода как у оригинала (fp32 при fp32-весе)
        y = torch.empty((rows, n), dtype=out_dtype, device=x.device)  # буфер результата
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)  # буфер rstd
        block = triton.next_power_of_2(n)  # степень двойки >= n
        warps = 8 if block >= 4096 else 4  # число варпов
        _rmsnorm_fwd_kernel[(rows,)](x2, weight.contiguous(), y, rstd, n, eps,
                                     BLOCK=block, num_warps=warps)  # запуск
        ctx.save_for_backward(x2, weight, rstd)  # сохраняем для backward
        ctx.shape = shape  # форма для возврата градиента
        return y.view(shape)  # возвращаем в исходной форме

    @staticmethod
    def backward(ctx, grad_out):
        x2, weight, rstd = ctx.saved_tensors  # достаём сохранённое
        rows, n = x2.shape  # размеры
        g = grad_out.reshape(-1, n).contiguous()  # градиент как плоская матрица
        dx = torch.empty_like(x2)  # градиент входа в типе входа
        block = triton.next_power_of_2(n)  # степень двойки
        warps = 8 if block >= 4096 else 4  # варпы
        _rmsnorm_bwd_kernel[(rows,)](x2, weight.contiguous(), g, rstd, dx, n,
                                     BLOCK=block, num_warps=warps)  # запуск
        dw = None  # градиент веса по умолчанию не нужен
        if ctx.needs_input_grad[1]:  # вес нормы обычно заморожен, но на всякий случай
            xn = (x2.float() * rstd[:, None]).to(x2.dtype).float()  # округлённый xhat, как в forward
            dw = (g.float() * xn).sum(0).to(weight.dtype)  # сумма по строкам
        return dx.view(ctx.shape), dw, None  # градиенты для x, weight, eps


def make_forward(original):
    """Замена Qwen2RMSNorm.forward (та же сигнатура)."""

    def forward(self, hidden_states):
        w = self.weight  # вес нормы
        if (hidden_states.is_cuda and w.is_cuda
                and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and w.dtype in (torch.float16, torch.bfloat16, torch.float32)):  # условия быстрого пути
            return _RMSNorm.apply(hidden_states, w, self.variance_epsilon)  # наш кернел
        return original(self, hidden_states)  # иначе обычный код

    forward._is_fused = True  # метка, чтобы не патчить дважды
    return forward