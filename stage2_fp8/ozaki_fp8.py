"""
ozaki_fp8.py — Stage 2: Ozaki Scheme DGEMM via FP8 (E4M3) Tensor Cores
========================================================================

配套阅读: ./PRINCIPLE.md (delta 文档，假设你读过 ../stage1_fp16/)。

与 Stage 1 的对照（只列改动）:

  | 位置                       | Stage 1               | Stage 2 (本文件)           |
  |---------------------------|-----------------------|----------------------------|
  | compute_rho 默认 m_type2  | 11 (M_FP16)           | 4  (M_FP8_E4M3)            |
  | slice_iter_kernel cvt     | .to(tl.float16)       | .to(tl.float8e4nv)         |
  | slice_matrix_rows 输出    | torch.float16         | torch.float8_e4m3fn        |
  | 切片数 (典型 k=1024)      | 8 (=> 64 GEMM)        | 14 (=> 196 GEMM)           |

其他完全照搬 — slicing 公式、累加 kernel、主循环都不变。
有意识地保持代码结构、变量名、注释一致，便于和 stage1_fp16/ozaki_fp16.py diff 比对。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import triton
import triton.language as tl


# ============================================================================
# §4. 切片参数 rho 的选择（PRINCIPLE.md §2）
# ============================================================================

M_FP64 = 53
M_FP32 = 24
M_FP8_E4M3 = 4   # ★ 关键变化：FP8 E4M3 有 3 stored + 1 hidden = 4 mantissa bits


def compute_rho(k: int,
                m_type1: int = M_FP64,
                m_type2: int = M_FP8_E4M3,   # ★ 默认变成 FP8
                m_type3: int = M_FP32) -> int:
    r"""
    ρ = max(ξ, γ)，公式与 Stage 1 完全相同（PRINCIPLE.md §2）：

    $$\xi = m_1 - m_2,\quad \gamma = m_1 - \big\lfloor (m_3 - \log_2 k)/2 \big\rfloor$$

    只是 m_2 默认值从 11 (FP16) 变成 4 (FP8 E4M3)，导致 ξ 从 42 涨到 49 —
    大部分 k 下 ξ 主导，所以切片数 ≈ ⌈53/4⌉ = 14，对 k 几乎不敏感。
    """
    log2_k = max(1, math.ceil(math.log2(max(2, k))))
    xi = m_type1 - m_type2
    gamma = m_type1 - (m_type3 - log2_k) // 2
    return max(xi, gamma)


# ============================================================================
# §5.1 Triton Kernel #1 — 一轮切片（FP8 输出版本）
# ============================================================================

@triton.jit
def slice_iter_kernel(
    X_ptr,            # *fp64  [M, K]
    sigma_ptr,        # *fp64  [M]
    inv_2c_ptr,       # *fp64  [M]
    slice_out_ptr,    # *fp8e4nv [M, K]  ★ 变化：FP8 而非 FP16
    X_next_ptr,       # *fp64  [M, K]
    M, K,
    stride_xm, stride_xk,
    stride_sm, stride_sk,
    stride_nm, stride_nk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r"""
    与 Stage 1 的 slice_iter_kernel 字面意义一致，唯一区别是末尾 cvt 到 FP8 而非 FP16。

    Slicing 递推（Eq. 4–8）:
        v_i           = fl_64(x_i + σ) - σ
        x_next_i      = x_i - v_i
        slice_i       = cvt_FP8E4M3(2^(-c) · v_i)

    ★ 关键: 由于 FP8 E4M3 的 mantissa 只有 4 位，与 FP16 相比，
       同样的 v 经过 cvt 后会被舍掉更多位 —— 这部分被舍掉的位会自动留在 x_next
       里参与下一轮切片（这就是 Ozaki 递推的本质：低精度截断的"残量"由后续切片补回）。
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    sigma  = tl.load(sigma_ptr  + offs_m, mask=mask_m, other=0.0)
    inv_2c = tl.load(inv_2c_ptr + offs_m, mask=mask_m, other=0.0)

    v        = (x + sigma[:, None]) - sigma[:, None]
    x_next   = x - v
    # ★ FP64 → FP8 在 Triton 没有直接 cvt 指令，经 FP32 中转
    # 这一步不会引入额外精度损失：slice 已归一化到 [-1, 1]，FP32 (m3=24) 完全 cover
    slice_v  = (v * inv_2c[:, None]).to(tl.float32).to(tl.float8e4nv)

    s_ptrs = slice_out_ptr + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk
    n_ptrs = X_next_ptr   + offs_m[:, None] * stride_nm + offs_k[None, :] * stride_nk
    tl.store(s_ptrs, slice_v, mask=mask)
    tl.store(n_ptrs, x_next,  mask=mask)


# ============================================================================
# §5.2 GEMM 路径选择 — 这里有一个重要的教学故事
# ============================================================================
#
# 直觉上 Stage 2 的 GEMM kernel 应该和 Stage 1 几乎一样, 只把输入 dtype 从
# FP16 换成 FP8e4nv. 实际写出来 (见下方 matmul_fp8_TRITON_kernel) 也能编译运行,
# 生成的 PTX 看着完全对 (mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32),
# 但在 RTX 4090 (sm_89, Ada Lovelace) + Triton 3.6 这个组合下, **当 k ≥ 512
# 时累加结果会与 IEEE-FP32 ground truth 偏离**, 大概率是 Triton 在该 GPU 上的
# codegen 缺陷 (Hopper sm_90 无此问题):
#
#       k     Triton-FP8 误差    cuBLASLt-FP8 误差
#       64        0                  0
#      256        0                  0
#      512     3.5e-2                0
#     1024     6.3e-1                0
#     2048     5.0e+0                0
#
# 这一误差量级 (10^-2 ~ 10^0) 完全破坏 Ozaki Scheme 的"无舍入累加"前提.
#
# 论文 §4.1 实际使用 cuBLASLt 的 FP8 路径 (cublasLtMatmul); PyTorch 把它包成了
# torch._scaled_mm. 我们也走这条路 — 教学上仍然成立: Stage 2 的故事是"切片精度
# 变了, 但 GEMM 仍可用现成最快的 FP8 后端". Triton kernel 留在下方作为反面教材.

@triton.jit
def matmul_fp8_TRITON_kernel(   # NOTE: 主流程不调用, 仅作教学对照
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r"""
    教学用 — Triton FP8 GEMM. **在 Triton 3.6 + sm_89 上有已知精度问题**, 见上方注释.
    结构与 stage1_fp16/ozaki_fp16.py 中 matmul_fp16_kernel 完全一致, 只差输入 dtype.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

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

        # max_num_imprecise_acc=0 本意是禁用"imprecise accumulator", 实测在
        # sm_89 上**这个 flag 不生效** — 即使设了 0, k≥512 仍然误差爆炸.
        # 这正是我们改走 cuBLASLt 的根本原因.
        acc = tl.dot(a, b, acc=acc, max_num_imprecise_acc=0)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ============================================================================
# §5.3 Triton Kernel #3 — 加权累加（与 Stage 1 完全相同）
# ============================================================================

@triton.jit
def accumulate_kernel(
    C_ptr,           # *fp64  [M, N]
    G_ptr,           # *fp32  [M, N]   ← G 仍是 FP32，因为 GEMM 累加器是 FP32
    pow2_cA_ptr,     # *fp64  [M]
    pow2_cB_ptr,     # *fp64  [N]
    M, N,
    stride_cm, stride_cn,
    stride_gm, stride_gn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    r"""
    与 Stage 1 字面一致 —— 因为 GEMM 的输出仍是 FP32（accumulator dtype），
    所以累加 kernel 不需要任何变化。这是 Ozaki Scheme 的一个优雅性质：
    切片精度变了，但 GEMM 累加器和最终加权累加都不动。

    $$C[i,j] \mathrel{+}= 2^{c_A^{(p)}[i]} \cdot 2^{c_B^{(q)}[j]} \cdot G^{(p,q)}[i,j]$$
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    pow2_cA = tl.load(pow2_cA_ptr + offs_m, mask=mask_m, other=0.0)
    pow2_cB = tl.load(pow2_cB_ptr + offs_n, mask=mask_n, other=0.0)
    scale = pow2_cA[:, None] * pow2_cB[None, :]

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
    沿行方向切片。输出每片为 FP8 E4M3 张量（vs Stage 1 的 FP16）。

    其余逻辑完全与 Stage 1 相同：
      - 主机端 row-reduce 算 row_max, sigma, inv_2c
      - 调 Triton kernel 写出切片 + 残量
      - 迭代到残量全 0 退出
    """
    assert X.dtype == torch.float64, f"Expected fp64, got {X.dtype}"
    assert X.is_contiguous()
    m, k = X.shape

    X_residual = X.clone()
    slices: List[torch.Tensor] = []
    cs: List[torch.Tensor] = []

    BLOCK_M, BLOCK_K = 32, 64
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k, BLOCK_K))

    for _ in range(max_slices):
        row_max = X_residual.abs().amax(dim=1)
        if torch.all(row_max == 0):
            break

        c = torch.where(
            row_max > 0,
            torch.ceil(torch.log2(row_max)),
            torch.zeros_like(row_max),
        )
        sigma  = 0.75 * torch.exp2(rho + c)
        inv_2c = torch.exp2(-c)

        # ★ 输出 dtype 是 FP8 而不是 FP16
        slice_out = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=X.device)
        X_next    = torch.empty(m, k, dtype=torch.float64,        device=X.device)

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
        slices.append(torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=X.device))
        cs.append(torch.zeros(m, dtype=torch.float64, device=X.device))

    return slices, torch.stack(cs, dim=0)


_ONE_FP32 = None  # 延迟初始化, 避免 import 时就要求 cuda


def _scale_one():
    """torch._scaled_mm 要求传 scale_a/scale_b 标量张量, 我们用 1.0."""
    global _ONE_FP32
    if _ONE_FP32 is None:
        _ONE_FP32 = torch.tensor(1.0, device='cuda', dtype=torch.float32)
    return _ONE_FP32


def matmul_fp8_to_fp32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A @ B with A, B in FP8 E4M3 and output in FP32, **via cuBLASLt (torch._scaled_mm)**.

    为什么不用 Triton: 见上方 §5.2 注释 — Triton 3.6 在 sm_89 上 FP8 GEMM 有精度
    bug. cuBLASLt 是论文实际使用的后端, 精度 = IEEE FP32 ground truth.

    Layout 要求: torch._scaled_mm 需要 B 是 column-major (即 stride[0]=1).
    我们直接在 ozaki_dgemm() 里把 B_slice 构造成 col-major 视图喂进来, 这里直接传.
    """
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    assert A.shape[1] == B.shape[0]
    # B 必须 col-major (stride[0] == 1)
    assert B.stride(0) == 1, f"B must be col-major, got strides {B.stride()}"

    one = _scale_one()
    return torch._scaled_mm(
        A, B,
        scale_a=one, scale_b=one,
        out_dtype=torch.float32,
        use_fast_accum=False,   # ★ 关键: False = IEEE FP32 累加, 论文也用这个
    )


def accumulate_into(C: torch.Tensor,
                    G: torch.Tensor,
                    pow2_cA_p: torch.Tensor,
                    pow2_cB_q: torch.Tensor) -> None:
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
    """
    用 FP8 E4M3 Tensor Core + Ozaki Scheme 计算 C = A B (FP64 精度等价)。

    主流程与 Stage 1 完全相同，仅切片为 FP8、GEMM 走 FP8 Tensor Core。
    """
    assert A.dtype == torch.float64 and B.dtype == torch.float64
    assert A.is_cuda and B.is_cuda
    m, k = A.shape
    k2, n = B.shape
    assert k == k2

    rho = compute_rho(k)

    A_slices, c_A = slice_matrix_rows(A.contiguous(), rho=rho, max_slices=max_slices)

    Bt = B.T.contiguous()
    Bt_slices, c_B = slice_matrix_rows(Bt, rho=rho, max_slices=max_slices)
    # ★ 关键优化: Bt_slices[q] 形状是 [n, k] row-major (strides = (k, 1)).
    # 取 .T 得到 [k, n] 视图, strides = (1, k) —— 这就是 column-major,
    # 也是 torch._scaled_mm 想要的 B 布局, **无需 .contiguous() 拷贝**.
    # (Stage 1 因为 Triton kernel 要 row-major, 必须 .contiguous(); 这里反而省了)
    B_slices = [bs.T for bs in Bt_slices]

    s_A, s_B = len(A_slices), len(B_slices)

    pow2_cA = torch.exp2(c_A)
    pow2_cB = torch.exp2(c_B)

    C = torch.zeros(m, n, dtype=torch.float64, device=A.device)
    for p in range(s_A):
        for q in range(s_B):
            G_pq = matmul_fp8_to_fp32(A_slices[p], B_slices[q])
            accumulate_into(C, G_pq, pow2_cA[p].contiguous(),
                                     pow2_cB[q].contiguous())

    if return_meta:
        return C, (s_A, s_B)
    return C
