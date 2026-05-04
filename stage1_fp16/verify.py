"""
verify.py — Stage 1 精度 / 正确性验证脚本

跑法：
    conda activate tridev
    python verify.py

期望输出：相对误差 ~ 1e-15（接近 FP64 机器精度）。
"""

import time
import torch

from ozaki_fp16 import compute_rho, ozaki_dgemm

torch.manual_seed(0)
DEV = "cuda"


def fmt_err(name, err):
    """带颜色的简易格式化（如果不在 tty 也无所谓）。"""
    threshold_good, threshold_warn = 1e-13, 1e-10
    if err < threshold_good:
        tag = "OK"
    elif err < threshold_warn:
        tag = "WARN"
    else:
        tag = "FAIL"
    return f"[{tag:<4}] {name}: max_rel_err = {err:.3e}"


def run_case(m, k, n, label=""):
    """单一规模的精度对比 + 速度对比。"""
    # 论文 §5.1 用 (1, 10) 均匀分布，使得切片数恰好填满 mantissa
    A = torch.rand(m, k, device=DEV, dtype=torch.float64) * 9 + 1
    B = torch.rand(k, n, device=DEV, dtype=torch.float64) * 9 + 1

    # ---- Reference: cuBLAS DGEMM (warm-up 一次再计时, 排除 cuBLAS 初次编译开销) ----
    _ = A @ B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C_ref = A @ B
    torch.cuda.synchronize()
    t_ref = time.perf_counter() - t0

    # ---- Ozaki (warm-up 一次, 让 Triton 把 kernel 编译完) ----
    _ = ozaki_dgemm(A, B, return_meta=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C_oz, (s_A, s_B) = ozaki_dgemm(A, B, return_meta=True)
    torch.cuda.synchronize()
    t_oz = time.perf_counter() - t0

    # ---- 误差 ----
    diff = (C_oz - C_ref).abs()
    denom = C_ref.abs().clamp(min=1e-300)
    rel = diff / denom
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    rho = compute_rho(k)
    print(
        f"[{label:<8}] m={m:<5} k={k:<5} n={n:<5} | "
        f"rho={rho:<3} slices={s_A}x{s_B}={s_A*s_B:<3} | "
        f"max_rel={max_rel:.2e} mean_rel={mean_rel:.2e} | "
        f"t_ref={t_ref*1e3:6.2f}ms t_ozaki={t_oz*1e3:7.2f}ms slowdown={t_oz/max(t_ref,1e-9):.1f}x"
    )
    return max_rel


def main():
    print("=" * 100)
    print("Stage 1 verify: FP16 + FP32 accum  →  emulated DGEMM via Ozaki Scheme")
    print("=" * 100)
    print(f"compute_rho(k=128)   = {compute_rho(128)}   (xi=42, gamma_k=128 主导)")
    print(f"compute_rho(k=512)   = {compute_rho(512)}")
    print(f"compute_rho(k=1024)  = {compute_rho(1024)}")
    print(f"compute_rho(k=4096)  = {compute_rho(4096)}")
    print()

    print("-" * 100)
    print("精度 + 速度对比 (vs torch.matmul aka cuBLAS DGEMM)")
    print("-" * 100)

    # 几个梯度，看精度 / 速度怎么走
    cases = [
        ( 128,  128,  128, "tiny"),
        ( 256,  256,  256, "small"),
        ( 512,  512,  512, "med"),
        (1024, 1024, 1024, "large"),
        (2048, 2048, 2048, "xlarge"),
    ]

    max_errors = []
    for (m, k, n, label) in cases:
        try:
            err = run_case(m, k, n, label)
            max_errors.append(err)
        except Exception as e:
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
            raise

    print()
    print("=" * 100)
    worst = max(max_errors) if max_errors else float("inf")
    print(fmt_err("OVERALL", worst))
    print(f"FP64 unit roundoff ≈ 1.11e-16；通过门槛设为 1e-13（约 1000 ULP）")
    print("=" * 100)


if __name__ == "__main__":
    main()
