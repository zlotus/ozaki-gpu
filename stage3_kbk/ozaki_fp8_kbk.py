"""
ozaki_fp8_kbk.py — Stage 3: Inner-product Blocking on FP8 Ozaki Scheme
========================================================================

配套阅读: ./PRINCIPLE.md (delta 文档, 假设你已经吃透 Stage 1 和 Stage 2)。

Stage 3 = Stage 2 + 一个外层 k-block 循环。**完全没有新 Triton kernel**,
只在 host 端把 k 维切成长度 k_bk 的多块, 每块独立调 Stage 2 的整套流水线,
块间结果用 FP64 加法在同一个 C 上累加 (PRINCIPLE.md §6.4)。

代码意图：教学上演示"一个 host-side 改造就能 unblock 全新的能力"
        —— 不动 GPU 代码,Stage 2 装不下的大 k 现在能跑了。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

# 直接复用 Stage 2 的部件（PRINCIPLE.md §6.1）。把 stage2_fp8 加进 sys.path。
_STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_fp8"
if str(_STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE2_DIR))
from ozaki_fp8 import (    # noqa: E402  (允许 path 插入后再 import)
    compute_rho,
    slice_matrix_rows,
    matmul_fp8_to_fp32,
    accumulate_into,
)


# ============================================================================
# 主入口
# ============================================================================

def ozaki_dgemm_kbk(A: torch.Tensor,
                    B: torch.Tensor,
                    k_bk: int = 1024,
                    max_slices: int = 24,
                    return_meta: bool = False):
    r"""
    k-blocked Ozaki Scheme DGEMM via FP8 Tensor Cores (cuBLASLt).

    把 k 维切成长度 k_bk 的多块, 对每块独立做完整 Ozaki, 块间结果用 FP64 累加。

    数学:
        $$C \;=\; \sum_{i=0}^{b-1} A_{[:,\,i k_{bk}:(i+1)k_{bk}]} \cdot B_{[i k_{bk}:(i+1)k_{bk},\,:]}$$

    Args:
        A          : [m, k] FP64
        B          : [k, n] FP64
        k_bk       : 块大小; 设为 >= k 即退化成 Stage 2 (无 blocking)
        max_slices : 单块内的切片数上限 (传给 slice_matrix_rows)
        return_meta: 若 True, 返回 (C, dict with profiling info)

    Returns:
        C : [m, n] FP64
        (可选) meta dict: { 'n_blocks', 'total_gemm_count', 's_A_first_block', ... }
    """
    assert A.dtype == torch.float64 and B.dtype == torch.float64
    assert A.is_cuda and B.is_cuda
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    assert k_bk >= 1

    C = torch.zeros(m, n, dtype=torch.float64, device=A.device)

    n_blocks = 0
    total_gemm = 0
    sA_record, sB_record = -1, -1   # 记录第一块的切片数, 方便 verify 打印

    for k_start in range(0, k, k_bk):
        k_end = min(k_start + k_bk, k)
        kb = k_end - k_start    # 本块实际 k (最后一块可能 < k_bk)

        # —— 取出本块的 A、B 切片(view + contiguous 拷贝, 见 PRINCIPLE.md §6.3) ——
        A_block = A[:, k_start:k_end].contiguous()       # [m, kb] FP64
        B_block = B[k_start:k_end, :].contiguous()       # [kb, n] FP64

        # 这一块自己的 rho。kb 小时 gamma 也小; 在我们的保守公式下 rho 一般还是 = ξ.
        rho = compute_rho(kb)

        # —— Stage 2 的切片流水线, 完全复用 ——
        A_slices, c_A = slice_matrix_rows(A_block, rho=rho, max_slices=max_slices)

        Bt_block = B_block.T.contiguous()                # [n, kb] FP64
        Bt_slices, c_B = slice_matrix_rows(Bt_block, rho=rho, max_slices=max_slices)
        # bs 形状 [n, kb] row-major, .T 视图就是 [kb, n] col-major(零拷贝),
        # 正是 torch._scaled_mm 想要的格式(Stage 2 PRINCIPLE.md §4.2 坑 3)
        B_slices = [bs.T for bs in Bt_slices]

        s_A, s_B = len(A_slices), len(B_slices)
        if n_blocks == 0:
            sA_record, sB_record = s_A, s_B

        pow2_cA = torch.exp2(c_A)    # [s_A, m] FP64
        pow2_cB = torch.exp2(c_B)    # [s_B, n] FP64

        # —— 内层双循环 → 累加到 C ——
        # 注意: C 在所有块之间共用, 所以 accumulate_into 隐式完成块间 FP64 求和
        # (PRINCIPLE.md §6.4)
        for p in range(s_A):
            for q in range(s_B):
                G_pq = matmul_fp8_to_fp32(A_slices[p], B_slices[q])
                accumulate_into(C, G_pq,
                                pow2_cA[p].contiguous(),
                                pow2_cB[q].contiguous())

        total_gemm += s_A * s_B
        n_blocks   += 1

        # —— 显式 free, 防止下一块切片峰值内存翻倍 (PRINCIPLE.md §6.2) ——
        del A_slices, Bt_slices, B_slices, c_A, c_B, pow2_cA, pow2_cB
        del A_block, B_block, Bt_block

    if return_meta:
        meta = {
            "n_blocks":          n_blocks,
            "total_gemm_count":  total_gemm,
            "s_A_first_block":   sA_record,
            "s_B_first_block":   sB_record,
            "k_bk":              k_bk,
        }
        return C, meta
    return C
