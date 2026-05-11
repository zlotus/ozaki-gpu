"""
verify.py — Stage 2 精度 / 正确性验证脚本

跑法:
    conda activate tridev
    python verify.py

期望:
  - 相对误差 ~ 1e-15 (与 Stage 1 持平，理由见 PRINCIPLE.md §5)
  - 切片数 ≈ 14 × 14 = 196 (vs Stage 1 的 7-9 × 7-9)
  - 总耗时比 Stage 1 慢 ~1.5x (PRINCIPLE.md §3 推导)
"""

import sys
import time
import torch

# 同时导入 Stage 1 做对照（路径需要加 stage1_fp16 的父目录）
sys.path.insert(0, "..")
from stage1_fp16.ozaki_fp16 import ozaki_dgemm as ozaki_fp16
from ozaki_fp8 import compute_rho, ozaki_dgemm as ozaki_fp8

torch.manual_seed(0)
DEV = "cuda"


def fmt_err(name, err):
    threshold_good, threshold_warn = 1e-13, 1e-10
    if err < threshold_good:
        tag = "OK"
    elif err < threshold_warn:
        tag = "WARN"
    else:
        tag = "FAIL"
    return f"[{tag:<4}] {name}: max_rel_err = {err:.3e}"


def time_call(fn, *args, warmup=1):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def run_case(m, k, n, label=""):
    A = torch.rand(m, k, device=DEV, dtype=torch.float64) * 9 + 1
    B = torch.rand(k, n, device=DEV, dtype=torch.float64) * 9 + 1

    # ---- Reference: cuBLAS DGEMM ----
    C_ref, t_ref = time_call(lambda: A @ B)

    # ---- Stage 1 FP16 (for comparison) ----
    C16, t16 = time_call(lambda: ozaki_fp16(A, B, return_meta=False))

    # ---- Stage 2 FP8 ----
    res, t8 = time_call(lambda: ozaki_fp8(A, B, return_meta=True))
    C8, (s_A, s_B) = res

    rel_fp16 = (C16 - C_ref).abs() / C_ref.abs().clamp(min=1e-300)
    rel_fp8  = (C8  - C_ref).abs() / C_ref.abs().clamp(min=1e-300)

    print(f"[{label:<8}] m={m:<5} k={k:<5} n={n:<5} | "
          f"rho={compute_rho(k):<3} slices={s_A}x{s_B}={s_A*s_B:<3} | "
          f"FP16 max_rel={rel_fp16.max().item():.2e} t={t16*1e3:6.2f}ms | "
          f"FP8  max_rel={rel_fp8.max().item():.2e} t={t8*1e3:6.2f}ms | "
          f"t_cuBLAS={t_ref*1e3:5.2f}ms")
    return rel_fp8.max().item()


def main():
    print("=" * 130)
    print("Stage 2 verify: FP8 E4M3 + FP32 accum  vs  Stage 1 (FP16) vs cuBLAS DGEMM")
    print("=" * 130)
    print(f"compute_rho(k=128)  = {compute_rho(128)}    (ξ=49 主导, FP8 在小 k 下切片数对 k 不敏感)")
    print(f"compute_rho(k=1024) = {compute_rho(1024)}")
    print(f"compute_rho(k=4096) = {compute_rho(4096)}")
    print(f"compute_rho(k=8192) = {compute_rho(8192)}    (γ 开始 catch up)")
    print()

    cases = [
        ( 128,  128,  128, "tiny"),
        ( 256,  256,  256, "small"),
        ( 512,  512,  512, "med"),
        (1024, 1024, 1024, "large"),
        (2048, 2048, 2048, "xlarge"),
    ]

    max_errors = []
    for (m, k, n, label) in cases:
        max_errors.append(run_case(m, k, n, label))

    print()
    print("=" * 130)
    print(fmt_err("OVERALL (FP8 path)", max(max_errors)))
    print("通过门槛: 1e-13 (~1000 ULP of FP64)")
    print("=" * 130)


if __name__ == "__main__":
    main()
