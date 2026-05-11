"""
verify.py — Stage 3 精度 / 内存 / 速度验证

跑法:
    conda activate tridev
    python verify.py

三个观察目标:
  (A) **精度**: 不管 k_bk 怎么选, max_rel_err 都应保持 ~1e-15
       (块间额外 FP64 求和误差 ~b·ulp, 仍达到标准 DGEMM 量级)
  (B) **内存**: 峰值显存随 k_bk 线性下降 — 演示"为啥需要 blocking"
  (C) **速度**: 论文 Fig 5 暗示存在 sweet spot k_bk (∼4096), 我们扫一遍验证
"""

import sys
import time
import torch
from pathlib import Path

# 让 Stage 2 (供 Stage 3 import) 找得到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage2_fp8"))
from ozaki_fp8 import ozaki_dgemm as ozaki_no_blocking
from ozaki_fp8_kbk import ozaki_dgemm_kbk    # noqa: E402

torch.manual_seed(0)
DEV = "cuda"


def time_call(fn, warmup=1):
    """跑 warmup+1 次, 返回 (返回值, 第二次的耗时)。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def measure_peak_mem(fn):
    """返回函数运行期间的峰值显存增量 (MB)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return peak / 1024 / 1024


def run_block_sweep(m, k, n):
    """对一个 (m, k, n) 跑多个 k_bk, 报精度/速度/内存."""
    A = torch.rand(m, k, device=DEV, dtype=torch.float64) * 9 + 1
    B = torch.rand(k, n, device=DEV, dtype=torch.float64) * 9 + 1

    # —— Reference (cuBLAS DGEMM) ——
    C_ref, t_ref = time_call(lambda: A @ B)

    print(f"\n=== m={m}, k={k}, n={n} ===   cuBLAS DGEMM = {t_ref*1e3:.2f} ms")
    print(f"{'kind':<18} {'k_bk':>6} {'blocks':>7} {'slices':>10} "
          f"{'gemms':>7} {'max_rel':>10} {'time(ms)':>10} {'peak_mem(MB)':>14}")

    rows = []

    # —— Stage 2 baseline (no blocking) ——
    def f_noblk(): return ozaki_no_blocking(A, B, return_meta=True)
    res, t = time_call(f_noblk)
    if isinstance(res, tuple):
        C_oz, (sA, sB) = res
    else:
        C_oz = res; sA = sB = -1
    rel = ((C_oz - C_ref).abs() / C_ref.abs().clamp(min=1e-300)).max().item()
    mem = measure_peak_mem(lambda: ozaki_no_blocking(A, B))
    rows.append(("stage2-noblock", k, 1, f"{sA}x{sB}", sA*sB, rel, t*1e3, mem))

    # —— Stage 3 with various k_bk ——
    candidates = sorted({kb for kb in [256, 512, 1024, 2048, 4096] if kb <= k})
    for kb in candidates:
        def f_kbk(kb=kb): return ozaki_dgemm_kbk(A, B, k_bk=kb, return_meta=True)
        (C_oz, meta), t = time_call(f_kbk)
        rel = ((C_oz - C_ref).abs() / C_ref.abs().clamp(min=1e-300)).max().item()
        mem = measure_peak_mem(lambda kb=kb: ozaki_dgemm_kbk(A, B, k_bk=kb))
        sA1, sB1 = meta["s_A_first_block"], meta["s_B_first_block"]
        rows.append((
            "stage3-kbk", kb, meta["n_blocks"],
            f"{sA1}x{sB1}", meta["total_gemm_count"], rel, t*1e3, mem,
        ))

    # 打印
    for kind, kb, nb, slices, gemms, rel, tms, mem in rows:
        flag = "OK" if rel < 1e-13 else ("WARN" if rel < 1e-10 else "FAIL")
        print(f"{kind:<18} {kb:>6} {nb:>7} {slices:>10} "
              f"{gemms:>7} {rel:>10.2e} {tms:>10.2f} {mem:>14.1f}  [{flag}]")


def main():
    print("=" * 115)
    print("Stage 3 verify: inner-product blocking on FP8 Ozaki Scheme")
    print("=" * 115)

    # 中等尺寸: 看 k_bk 怎么影响精度/速度/内存
    for (m, k, n) in [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (2048, 4096, 2048),    # 高瘦 k — blocking 的典型适用场景
    ]:
        run_block_sweep(m, k, n)

    print()
    print("=" * 115)
    print("观察清单:")
    print("  (A) 同一行 max_rel 不应 > 1e-13 — blocking 不掉精度")
    print("  (B) Stage 3 的 peak_mem 应随 k_bk 减小而下降 — 这是 blocking 的核心价值")
    print("  (C) k_bk 太小时 time 反而上升 — kernel launch 开销 + 小 GEMM 效率低")
    print("=" * 115)


if __name__ == "__main__":
    main()
