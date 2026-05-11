"""
ozaki_fp8_emu.py — Stage 4: Ozaki Scheme DGEMM with FP64-emulated slicing
==========================================================================

实现论文 §4.2 的核心命题: **不使用硬件 FP64 算术指令** 也能完成 Ozaki Scheme.

与 Stage 2 (stage2_fp8/ozaki_fp8.py) 的差异:
  - slice_iter_kernel  → slice_iter_kernel_emu, 内部 (x+σ)-σ / x-v 都走 emu_fp64_add
  - sigma / c 的预计算: 用 PyTorch 整数位运算实现 (而非 torch.exp2, torch.ceil, torch.log2)
  - 其余 (FP8 GEMM + 加权累加) 与 Stage 2 完全一致 — 这些过程本身已不依赖 FP64 算术

**最后一块边界**: slicing kernel 最后一行做 FP64 → FP32 → FP8 cvt 时,
仍调用硬件 cvt 指令 (.to(tl.float32).to(tl.float8e4nv)). 严格来说这也是 FP64
指令的一种, 可以再用纯位运算 emulate (从 uint64 直接打 FP8 bit pattern), 但
教学上多 ~50 行而不增添新概念, 故留作 PRINCIPLE.md §5 的 reading.

配套阅读: ./PRINCIPLE.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import triton
import triton.language as tl

# 直接引用 Stage 4 的 FP64 emulator
from fp64_emu import emu_fp64_add

# 复用 Stage 2 的 FP8 GEMM + 加权累加 (它们不依赖 FP64 算术)
_STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_fp8"
if str(_STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE2_DIR))
from ozaki_fp8 import (
    compute_rho,
    matmul_fp8_to_fp32,
    accumulate_into,
)


# ============================================================================
# §1. 整数版的 per-row 预计算: row_max, c, sigma_bits
# ============================================================================
#
# Stage 2 用了 torch.exp2 / torch.ceil / torch.log2 在 FP64 域运算. Stage 4 全
# 部用位操作完成, 同样的 GPU op 但只触及整数硬件:
#   - |x|              = clear sign bit of FP64 bits
#   - max_along_dim    = int64 比较 (对正 FP64, bit-pattern 顺序就是数值顺序)
#   - ceil(log2(x))    = biased_exp - 1023 + (mantissa != 0 ? 1 : 0)
#   - sigma_bits (0.75·2^e) = 直接 pack: exp 字段 = e - 1 + 1023, mantissa = 1<<51

def per_row_preparation(X: torch.Tensor, rho: int):
    """整数位运算完成所有 per-row 预计算.

    Args:
        X: [m, k] FP64
        rho: 切片参数

    Returns:
        c           : [m] int32   per-row 指数, c = ⌈log2 max_i |x_i|⌉ (零行设 0)
        sigma_bits  : [m] uint64  per-row σ 的 IEEE bit pattern (零行设 0)
        any_nonzero : bool        是否还有任何残量需要再切一轮
    """
    assert X.dtype == torch.float64

    # — |x| 的 bit pattern: 清除 sign bit —
    x_bits = X.view(torch.int64) & 0x7FFFFFFFFFFFFFFF      # 正数 bit pattern
    # 对正 FP64, bit pattern 顺序 = 数值顺序, 所以行内 amax 直接给出 max(|x|) 的 bits
    row_max_bits = x_bits.amax(dim=1)                       # [m] int64

    is_zero = row_max_bits == 0

    # — ceil(log2(row_max)): 拆出 biased_exp 和 mantissa, 整数运算 —
    biased_exp = (row_max_bits >> 52) & 0x7FF              # [m]
    mantissa = row_max_bits & 0xFFFFFFFFFFFFF
    # ceil(log2(x)) = (biased_exp - 1023) + (1 if mantissa != 0 else 0)
    c = (biased_exp - 1023 + (mantissa != 0).to(torch.int64)).to(torch.int32)
    c = torch.where(is_zero, torch.zeros_like(c), c)

    # — sigma_bits = 0.75 · 2^(rho+c) 的 IEEE bit pattern —
    # 0.75 = 1.5 · 2^(-1) → biased_exp = (rho + c - 1) + 1023, mantissa stored = 1<<51 (= 0.5)
    sigma_exp_biased = (rho + c.to(torch.int64) - 1 + 1023) & 0x7FF
    sigma_bits = (sigma_exp_biased << 52) | (1 << 51)       # uint64 (in int64 container)
    sigma_bits = torch.where(is_zero, torch.zeros_like(sigma_bits), sigma_bits)

    return c, sigma_bits.view(torch.uint64), not bool(is_zero.all().item())


# ============================================================================
# §2. Triton slicing kernel — 仅用整数运算 (除了最后 cvt 到 FP8)
# ============================================================================

@triton.jit
def slice_iter_kernel_emu(
    X_ptr,            # *fp64       [M, K]    本轮残量 (作为 fp64 张量传入, 内部 bitcast)
    sigma_bits_ptr,   # *uint64     [M]
    c_ptr,            # *int32      [M]
    slice_out_ptr,    # *fp8e4nv    [M, K]
    X_next_ptr,       # *fp64       [M, K]    下一轮残量 (写出仍是 fp64 view)
    M, K,
    stride_xm, stride_xk,
    stride_sm, stride_sk,
    stride_nm, stride_nk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r"""
    论文 Eq. 4-8 的整数版实现:
        v        = fl_64(x + σ) - σ           # 全程 emu_fp64_add, 无 tl.float64
        x_next   = x - v                       # emu_fp64_add (b 翻 sign bit)
        slice    = v · 2^(-c)                  # 纯指数减法 (uint64 减常量)
        → cvt to FP8                           # 最后一步用硬件 cvt (见 PRINCIPLE.md §5)

    所有内部计算都是 `tl.uint64` / `tl.int32`. 不出现一次 tl.float64 加减乘.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # ─── Load: x 作为 fp64 读, 立刻 bitcast 到 uint64, 此后不再当 fp64 用 ───
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x_f64 = tl.load(x_ptrs, mask=mask, other=0.0)
    x_bits = x_f64.to(tl.uint64, bitcast=True)

    # σ_bits, c — per row, broadcast across K
    sigma_bits_row = tl.load(sigma_bits_ptr + offs_m, mask=mask_m, other=0).to(tl.uint64)
    c_row = tl.load(c_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)

    sigma_bits = sigma_bits_row[:, None]                                # [BM, 1] broadcast
    sigma_neg_bits = sigma_bits ^ 0x8000000000000000                    # -σ via sign-flip

    # ─── v = (x + σ) - σ, 全程整数模拟 ───────────────────────────────
    x_plus_sigma_bits = emu_fp64_add(x_bits, sigma_bits)
    v_bits = emu_fp64_add(x_plus_sigma_bits, sigma_neg_bits)

    # ─── x_next = x - v ───────────────────────────────────────────────
    v_neg_bits = v_bits ^ 0x8000000000000000
    x_next_bits = emu_fp64_add(x_bits, v_neg_bits)

    # ─── slice = v · 2^(-c): 拆 v_bits, 把 exp 字段减 c (纯整数) ─────
    v_sign = (v_bits >> 63) & 1
    v_exp  = ((v_bits >> 52) & 0x7FF).to(tl.int32)
    v_mant = v_bits & 0xFFFFFFFFFFFFF
    v_is_zero = v_exp == 0
    new_exp = v_exp - c_row[:, None]
    # repack (only when v != 0; otherwise keep all-zero bits)
    new_exp_field = (new_exp.to(tl.uint64) & 0x7FF) << 52
    v_scaled_bits = (v_sign << 63) | new_exp_field | v_mant
    v_scaled_bits = tl.where(v_is_zero, v_bits, v_scaled_bits)

    # ─── 最后一步: bitcast 回 fp64 view, 经 fp32 cvt 落到 fp8e4nv ───
    # 这里用硬件 cvt 指令; 严格 emulate 需再写 ~50 行位运算 (见 PRINCIPLE.md §5)
    v_scaled_f64 = v_scaled_bits.to(tl.float64, bitcast=True)
    slice_v = v_scaled_f64.to(tl.float32).to(tl.float8e4nv)

    # x_next 写回为 fp64 view (下一轮 kernel 的输入)
    x_next_f64 = x_next_bits.to(tl.float64, bitcast=True)

    s_ptrs = slice_out_ptr + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk
    n_ptrs = X_next_ptr   + offs_m[:, None] * stride_nm + offs_k[None, :] * stride_nk
    tl.store(s_ptrs, slice_v, mask=mask)
    tl.store(n_ptrs, x_next_f64, mask=mask)


# ============================================================================
# §3. 主机端封装
# ============================================================================

def slice_matrix_rows_emu(X: torch.Tensor,
                          rho: int,
                          max_slices: int = 24
                          ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """与 Stage 2 的 slice_matrix_rows 接口一致, 内部走 emu kernel."""
    assert X.dtype == torch.float64 and X.is_contiguous()
    m, k = X.shape

    X_residual = X.clone()
    slices: List[torch.Tensor] = []
    cs: List[torch.Tensor] = []

    BLOCK_M, BLOCK_K = 32, 64
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k, BLOCK_K))

    for _ in range(max_slices):
        c, sigma_bits, has_residual = per_row_preparation(X_residual, rho)
        if not has_residual:
            break

        slice_out = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=X.device)
        X_next    = torch.empty(m, k, dtype=torch.float64,        device=X.device)

        slice_iter_kernel_emu[grid](
            X_residual, sigma_bits, c, slice_out, X_next,
            m, k,
            X_residual.stride(0), X_residual.stride(1),
            slice_out.stride(0),  slice_out.stride(1),
            X_next.stride(0),     X_next.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        slices.append(slice_out)
        cs.append(c)
        X_residual = X_next

    if not slices:
        slices.append(torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=X.device))
        cs.append(torch.zeros(m, dtype=torch.int32, device=X.device))

    return slices, torch.stack(cs, dim=0)    # cs: [s, m] int32


def ozaki_dgemm_emu(A: torch.Tensor,
                    B: torch.Tensor,
                    max_slices: int = 24,
                    return_meta: bool = False):
    """
    与 Stage 2 ozaki_dgemm 接口一致, 但内部 slicing 完全用整数模拟 FP64 算术.
    GEMM 部分继续走 cuBLASLt FP8 (没有 FP64 在里面).
    累加部分仍用 Stage 2 的 accumulate_kernel — 它内部有 FP64 算术, 严格 emu 也
    需要替换 (见 PRINCIPLE.md §6 "为什么暂不 emulate accumulation"). 主张是:
      → slicing 是论文 §4.2 重点举例的部分, 这里完整 demo
      → accumulation 的 emu 与 slicing 镜像, 留作扩展
    """
    assert A.dtype == torch.float64 and B.dtype == torch.float64
    assert A.is_cuda and B.is_cuda
    m, k = A.shape
    k2, n = B.shape
    assert k == k2

    rho = compute_rho(k)

    A_slices, c_A = slice_matrix_rows_emu(A.contiguous(), rho=rho, max_slices=max_slices)

    Bt = B.T.contiguous()
    Bt_slices, c_B = slice_matrix_rows_emu(Bt, rho=rho, max_slices=max_slices)
    B_slices = [bs.T for bs in Bt_slices]    # col-major 视图, 喂 _scaled_mm

    s_A, s_B = len(A_slices), len(B_slices)

    # pow2_cA, pow2_cB 仍用 torch.exp2 (这是为了喂 Stage 2 的 accumulate kernel;
    # 严格 emu 时换成整数 pack `((c + 1023) << 52)` 即可, 见 PRINCIPLE.md)
    pow2_cA = torch.exp2(c_A.to(torch.float64))
    pow2_cB = torch.exp2(c_B.to(torch.float64))

    C = torch.zeros(m, n, dtype=torch.float64, device=A.device)
    for p in range(s_A):
        for q in range(s_B):
            G_pq = matmul_fp8_to_fp32(A_slices[p], B_slices[q])
            accumulate_into(C, G_pq,
                            pow2_cA[p].contiguous(),
                            pow2_cB[q].contiguous())

    if return_meta:
        return C, (s_A, s_B)
    return C
