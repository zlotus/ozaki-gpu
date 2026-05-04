"""
ozaki_fp16.py — Stage 1: Ozaki Scheme DGEMM via FP16 Tensor Cores
==================================================================

教学项目，目标是用最少的代码把 Ozaki Scheme 的算法骨架讲清楚。
配套阅读：./PRINCIPLE.md（本文件每个 §X 注释都对应那里的章节号）。

整体流水线（PRINCIPLE.md §6）：

    A_slices, c_A = slice_rows(A,   rho)   # FP64 → s_A 个 FP16 切片
    B_slices, c_B = slice_rows(B.T, rho)   # 切 B 的列 = 切 B.T 的行
    for p in range(s_A):
        for q in range(s_B):
            G_pq = A_slices[p] @ B_slices[q]    # FP16 in, FP32 out
            C   += 2^c_A[p] * 2^c_B[q] * G_pq   # 加权累加到 FP64 C

三个 Triton kernel：
    1. slice_iter_kernel   : 一轮切片（element-wise + 行广播）
    2. matmul_fp16_kernel  : FP16 → FP32 GEMM
    3. accumulate_kernel   : C[m,n] FP64 += s * G[m,n] FP32, s = 2^cA[m] * 2^cB[n]
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import triton
import triton.language as tl


# ============================================================================
# §4. 切片参数 rho 的选择
# ============================================================================

# IEEE 浮点格式的 mantissa 位数（含隐含位）
M_FP64 = 53
M_FP32 = 24
M_FP16 = 11


def compute_rho(k: int,
                m_type1: int = M_FP64,
                m_type2: int = M_FP16,
                m_type3: int = M_FP32) -> int:
    r"""
    计算 Ozaki Scheme 的 rho 参数（PRINCIPLE.md §4）。

    $$\xi = m_1 - m_2,\qquad \gamma = m_1 - \big\lfloor (m_3 - \log_2 k)/2 \big\rfloor$$
    $$\rho = \max(\xi,\,\gamma)$$

    - $\xi$  : 保证每片塞得进 Type2 (FP16) 的 mantissa
    - $\gamma$: 保证 GEMM 在 Type3 (FP32) 累加器里不溢出 mantissa

    注：本实现略保守 — 切片数会比论文 Table 3 多 1~2 片，但对算法正确性无影响。
    """
    # 防御性下限：k=1 时 log2 = 0
    log2_k = max(1, math.ceil(math.log2(max(2, k))))
    xi = m_type1 - m_type2
    gamma = m_type1 - (m_type3 - log2_k) // 2
    return max(xi, gamma)


# ============================================================================
# §5.1 Triton Kernel #1 — 一轮切片
# ============================================================================

@triton.jit
def slice_iter_kernel(
    # ---- 输入 / 输出指针 ----
    X_ptr,            # *fp64  [M, K]  本轮残量
    sigma_ptr,        # *fp64  [M]     每行的 sigma = 0.75 * 2^(rho + c)
    inv_2c_ptr,       # *fp64  [M]     每行的 2^(-c)
    slice_out_ptr,    # *fp16  [M, K]  本轮切片
    X_next_ptr,       # *fp64  [M, K]  下一轮残量
    # ---- 形状 / 步幅 ----
    M, K,
    stride_xm, stride_xk,
    stride_sm, stride_sk,
    stride_nm, stride_nk,
    # ---- 编译期常量 ----
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r"""
    对每个元素执行一轮切片（论文 Eq. 4–8）：

    $$
    \begin{aligned}
        v_i           &= \mathrm{fl}_{64}(x_i + \sigma) - \sigma \\
        x^{\text{next}}_i &= x_i - v_i \\
        x^{\text{slice}}_i &= \mathrm{cvt}_{16}(2^{-c}\, v_i)
    \end{aligned}
    $$

    sigma / 2^(-c) 是 **按行的标量**，按 K 维 broadcast。

    **关键**: `(x + sigma) - sigma` 必须用 FP64 的真实加减，靠 mantissa 截断
    取出 x 的高位。Triton 编译到 PTX 后保留这两条 fadd.f64 / fsub.f64 指令，不会
    被优化掉（因为 x 不是常量、且我们没启用 fast-math）。
    """
    # 每个 program 处理一个 [BLOCK_M, BLOCK_K] 的 tile
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # 加载本 tile 的 x（FP64）
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # 行广播：sigma 和 inv_2c 是 [BLOCK_M] 标量向量
    sigma  = tl.load(sigma_ptr  + offs_m, mask=mask_m, other=0.0)
    inv_2c = tl.load(inv_2c_ptr + offs_m, mask=mask_m, other=0.0)

    # ★ Ozaki 切片核心：3 行
    v        = (x + sigma[:, None]) - sigma[:, None]   # x 的高位（FP64 截断结果）
    x_next   = x - v                                    # 残量给下一轮
    slice_v  = (v * inv_2c[:, None]).to(tl.float16)    # 归一化到 [-1,1] 后降到 FP16

    # 写回
    s_ptrs = slice_out_ptr + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk
    n_ptrs = X_next_ptr   + offs_m[:, None] * stride_nm + offs_k[None, :] * stride_nk
    tl.store(s_ptrs, slice_v, mask=mask)
    tl.store(n_ptrs, x_next,  mask=mask)


# ============================================================================
# §5.2 Triton Kernel #2 — FP16 → FP32 GEMM
# ============================================================================

@triton.jit
def matmul_fp16_kernel(
    A_ptr, B_ptr, C_ptr,            # A: fp16 [M,K], B: fp16 [K,N], C: fp32 [M,N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r"""
    标准 Triton GEMM (参考 Triton tutorial 03)，唯一关键点：

    - 输入 FP16，**累加器在 FP32**（`acc = tl.zeros(..., tl.float32)`）
    - 输出 FP32（不再 round 回 FP16）

    这是 Ozaki Scheme 正确性的硬要求 — 见 PRINCIPLE.md §5.2。
    `tl.dot(a, b, acc=acc)` 会用 Tensor Core 做 mma.f16.f16.f32.f32 指令，
    硬件直接 FP32 累加，不会经过 FP16 中间态。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # FP32 累加器
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        a_ptrs = (A_ptr
                  + offs_m[:, None] * stride_am
                  + (k_start + offs_k)[None, :] * stride_ak)
        b_ptrs = (B_ptr
                  + (k_start + offs_k)[:, None] * stride_bk
                  + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K)
        b_mask = ((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc=acc)   # FP32 累加，硬件 Tensor Core

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ============================================================================
# §5.3 Triton Kernel #3 — 加权累加
# ============================================================================

@triton.jit
def accumulate_kernel(
    C_ptr,           # *fp64  [M, N]   in-place
    G_ptr,           # *fp32  [M, N]   一个 G_pq
    pow2_cA_ptr,     # *fp64  [M]      2^(c_A^(p)[i])
    pow2_cB_ptr,     # *fp64  [N]      2^(c_B^(q)[j])
    M, N,
    stride_cm, stride_cn,
    stride_gm, stride_gn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    r"""
    把一个切片乘积 $G^{(p,q)}$ 加权累加到 FP64 输出矩阵 $C$：

    $$C[i,j] \;\mathrel{+}=\; 2^{c_A^{(p)}[i]} \cdot 2^{c_B^{(q)}[j]} \cdot G^{(p,q)}[i,j]$$

    - 行/列广播 + FP32→FP64 提升 + in-place 累加 全部 fuse 在一个 kernel 里
    - 避免分配中间 [M,N] FP64 临时量（对 m=n=4096 来说每次能省 128 MB 分配）
    - 注意 G 是 FP32 输入；上探到 FP64 后再乘 scale，保证累加的精度
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    pow2_cA = tl.load(pow2_cA_ptr + offs_m, mask=mask_m, other=0.0)  # [BM] fp64
    pow2_cB = tl.load(pow2_cB_ptr + offs_n, mask=mask_n, other=0.0)  # [BN] fp64
    scale = pow2_cA[:, None] * pow2_cB[None, :]                       # [BM,BN] fp64

    g_ptrs = G_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float64)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_old  = tl.load(c_ptrs, mask=mask, other=0.0)
    tl.store(c_ptrs, c_old + scale * g, mask=mask)


# ============================================================================
# 主机端封装
# ============================================================================

def slice_matrix_rows(X: torch.Tensor,
                      rho: int,
                      max_slices: int = 24
                      ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    r"""
    把 X 沿行方向切成若干 FP16 切片。

    Args:
        X        : [m, k] FP64, contiguous
        rho      : 切片参数（见 compute_rho）
        max_slices: 安全上限，防御无穷循环

    Returns:
        slices : list of [m, k] FP16 tensors，长度 s_x
        cs     : [s_x, m] FP64 tensor of 每行指数 c_x^(p)

    迭代直到所有行的残量都为 0（见 PRINCIPLE.md §3）。
    """
    assert X.dtype == torch.float64, f"Expected fp64, got {X.dtype}"
    assert X.is_contiguous(), "X must be contiguous (caller responsibility)"
    m, k = X.shape

    X_residual: torch.Tensor = X.clone()  # clone 避免破坏 caller 的 X
    slices: List[torch.Tensor] = []
    cs: List[torch.Tensor] = []

    BLOCK_M, BLOCK_K = 32, 64
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k, BLOCK_K))

    for _ in range(max_slices):
        # 计算每行的 max(|x|) — 行 reduction 借 PyTorch 完成
        row_max = X_residual.abs().amax(dim=1)            # [m] fp64

        # 全为 0 → 残量已经被吃光，提前结束
        if torch.all(row_max == 0):
            break

        # c[i] = ceil(log2(row_max[i])); row_max=0 时设 0（该行此后不再贡献）
        c = torch.where(
            row_max > 0,
            torch.ceil(torch.log2(row_max)),
            torch.zeros_like(row_max),
        )
        sigma  = 0.75 * torch.exp2(rho + c)               # [m] fp64
        inv_2c = torch.exp2(-c)                           # [m] fp64

        slice_out = torch.empty(m, k, dtype=torch.float16, device=X.device)
        X_next    = torch.empty(m, k, dtype=torch.float64, device=X.device)

        slice_iter_kernel[grid](
            X_residual, sigma, inv_2c, slice_out, X_next,
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
        # 极端情况：输入全 0
        slices.append(torch.zeros(m, k, dtype=torch.float16, device=X.device))
        cs.append(torch.zeros(m, dtype=torch.float64, device=X.device))

    return slices, torch.stack(cs, dim=0)


def matmul_fp16_to_fp32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B with A, B in FP16 and output in FP32 (Triton)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert A.dtype == torch.float16 and B.dtype == torch.float16

    C = torch.empty(M, N, dtype=torch.float32, device=A.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_fp16_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


def accumulate_into(C: torch.Tensor,
                    G: torch.Tensor,
                    pow2_cA_p: torch.Tensor,
                    pow2_cB_q: torch.Tensor) -> None:
    """C += outer(pow2_cA_p, pow2_cB_q) * G   in-place, FP64."""
    M, N = C.shape
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    accumulate_kernel[grid](
        C, G, pow2_cA_p, pow2_cB_q,
        M, N,
        C.stride(0), C.stride(1),
        G.stride(0), G.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )


def ozaki_dgemm(A: torch.Tensor,
                B: torch.Tensor,
                max_slices: int = 24,
                return_meta: bool = False):
    r"""
    用 FP16 Tensor Core + Ozaki Scheme 计算 $C = A B$（FP64 精度等价）。

    Args:
        A : [m, k] FP64
        B : [k, n] FP64
        max_slices : 切片数安全上限
        return_meta : 若 True, 同时返回 (s_A, s_B) 用于调试 / 报告

    Returns:
        C : [m, n] FP64
        (可选) (s_A, s_B)
    """
    assert A.dtype == torch.float64 and B.dtype == torch.float64, \
        "Ozaki scheme expects FP64 inputs"
    assert A.is_cuda and B.is_cuda, "Need CUDA tensors"
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"Inner dims mismatch: {k} vs {k2}"

    # §4. 决定切片参数
    rho = compute_rho(k)

    # §3. 切片
    A_slices, c_A = slice_matrix_rows(A.contiguous(), rho=rho, max_slices=max_slices)

    # B 沿列切 == B.T 沿行切，最后再把每片 transpose 回去
    Bt = B.T.contiguous()                                   # [n, k]
    Bt_slices, c_B = slice_matrix_rows(Bt, rho=rho, max_slices=max_slices)
    B_slices = [bs.T.contiguous() for bs in Bt_slices]      # [k, n] FP16, contiguous

    s_A, s_B = len(A_slices), len(B_slices)

    # 一次性算好 2^c, 避免内层循环重复算
    pow2_cA = torch.exp2(c_A)   # [s_A, m] fp64
    pow2_cB = torch.exp2(c_B)   # [s_B, n] fp64

    # §6. 主循环：流式累加（一个 G_pq 算完立刻加到 C，不存全部 G）
    C = torch.zeros(m, n, dtype=torch.float64, device=A.device)
    for p in range(s_A):
        for q in range(s_B):
            G_pq = matmul_fp16_to_fp32(A_slices[p], B_slices[q])     # [m, n] fp32
            accumulate_into(C, G_pq, pow2_cA[p].contiguous(),
                                     pow2_cB[q].contiguous())

    if return_meta:
        return C, (s_A, s_B)
    return C
