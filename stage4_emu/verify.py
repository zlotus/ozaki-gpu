"""
verify.py — Stage 4 三层验证

  Level 1: Python ref FP64 add/mul == 硬件 FP64        (fp64_emu.py 内置)
  Level 2: Triton emu_fp64_add == Python ref / 硬件   (本文件 test_triton_add)
  Level 3: emu slicing 与 Stage 2 slicing 输出**逐元素一致**, 末端 Ozaki 精度持平

跑法:
    conda activate tridev
    python verify.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

# Stage 4 emulator
from fp64_emu import emu_fp64_add, fp64_add, f64_to_bits, bits_to_f64
from ozaki_fp8_emu import (
    ozaki_dgemm_emu, slice_matrix_rows_emu, per_row_preparation,
)

# Stage 2 for comparison
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage2_fp8"))
from ozaki_fp8 import (
    ozaki_dgemm as ozaki_fp8_stage2,
    slice_matrix_rows as slice_matrix_rows_stage2,
    compute_rho,
)

torch.manual_seed(0)
DEV = "cuda"


# ============================================================================
# Level 2: Triton emu_fp64_add 大批量随机对拍硬件
# ============================================================================

@triton.jit
def _add_kernel(A_ptr, B_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0).to(tl.uint64)
    b = tl.load(B_ptr + offs, mask=mask, other=0).to(tl.uint64)
    out = emu_fp64_add(a, b)
    tl.store(OUT_ptr + offs, out, mask=mask)


def test_triton_add():
    import random
    random.seed(0xCAFE)
    EXP_MASK = 0x7FF

    cases = []
    # Hand-picked
    for (a, b) in [(1.0, 1.0), (1.5, 2.0), (1.0, -1.0), (3.14, 2.71)]:
        cases.append((a, b))
    # Uniform [1, 10]
    for _ in range(500):
        cases.append((random.uniform(1, 10), random.uniform(1, 10)))
        cases.append((random.uniform(1, 10), -random.uniform(1, 10)))
    # Wide exponent
    for _ in range(500):
        ea, eb = random.randint(-100, 100), random.randint(-100, 100)
        a = random.uniform(0.5, 2) * 2.0**ea
        b = random.uniform(0.5, 2) * 2.0**eb
        if random.random() < 0.5: b = -b
        cases.append((a, b))
    # Ozaki (x+σ)-σ pattern
    for _ in range(500):
        sigma = 0.75 * 2.0**random.randint(40, 55)
        x = random.uniform(1, 10)
        cases.append((x, sigma))
        cases.append((x + sigma, -sigma))

    # Filter out subnormals/inf/nan inputs and results
    valid = []
    for (a, b) in cases:
        for x in (a, b, a + b):
            bits = f64_to_bits(x)
            exp = (bits >> 52) & EXP_MASK
            if exp == 0 or exp == EXP_MASK:
                break
        else:
            valid.append((a, b))
    N = len(valid)

    a_bits = torch.from_numpy(np.array([f64_to_bits(a) for (a, _) in valid], dtype=np.uint64)).cuda()
    b_bits = torch.from_numpy(np.array([f64_to_bits(b) for (_, b) in valid], dtype=np.uint64)).cuda()
    out = torch.zeros(N, dtype=torch.uint64, device='cuda')

    BLOCK = 256
    _add_kernel[(triton.cdiv(N, BLOCK),)](a_bits, b_bits, out, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    out_cpu = out.cpu().numpy()

    n_hw  = sum(int(out_cpu[i]) == f64_to_bits(a + b) for i, (a, b) in enumerate(valid))
    n_ref = sum(int(out_cpu[i]) == fp64_add(f64_to_bits(a), f64_to_bits(b))
                for i, (a, b) in enumerate(valid))
    flag_hw  = "OK" if n_hw  == N else "FAIL"
    flag_ref = "OK" if n_ref == N else "FAIL"
    print(f"[L2] Triton emu_fp64_add vs hardware FP64 :  {n_hw }/{N}  [{flag_hw }]")
    print(f"[L2] Triton emu_fp64_add vs Python ref    :  {n_ref}/{N}  [{flag_ref}]")
    return n_hw == N and n_ref == N


# ============================================================================
# Level 3a: emu slicing 输出 vs Stage 2 slicing — 逐元素一致 (FP8 bit-exact)
# ============================================================================

def test_slice_identity():
    """Stage 4 emu slicing 与 Stage 2 slicing 应产生**完全相同**的 FP8 切片."""
    m, k = 64, 256
    X = torch.rand(m, k, device=DEV, dtype=torch.float64) * 9 + 1
    X = X.contiguous()
    rho = compute_rho(k)

    # Stage 2 (硬件 FP64 slicing)
    slices_hw, c_A_hw = slice_matrix_rows_stage2(X, rho=rho)
    # Stage 4 (整数模拟 FP64 slicing)
    slices_emu, c_A_emu = slice_matrix_rows_emu(X, rho=rho)

    print(f"[L3a] Stage 2 vs Stage 4 slicing  (X: {m}x{k}, rho={rho})")
    print(f"     slice counts: stage2={len(slices_hw)}, stage4={len(slices_emu)}",
          "[OK]" if len(slices_hw) == len(slices_emu) else "[FAIL]")

    n_layers_to_check = min(len(slices_hw), len(slices_emu))
    all_match = True
    for p in range(n_layers_to_check):
        # FP8 张量按 bit pattern 比较
        s_hw  = slices_hw[p].view(torch.uint8)
        s_emu = slices_emu[p].view(torch.uint8)
        diff = (s_hw != s_emu).sum().item()
        c_match = (c_A_hw[p].to(torch.int64) == c_A_emu[p].to(torch.int64)).all().item()
        flag = "OK" if (diff == 0 and c_match) else "FAIL"
        print(f"     slice[{p:2d}]: FP8 diff bytes = {diff}, c match = {c_match}  [{flag}]")
        if diff or not c_match:
            all_match = False
    return all_match


# ============================================================================
# Level 3b: 端到端 Ozaki DGEMM 精度 — emu 与 Stage 2 同等水平
# ============================================================================

def test_e2e_accuracy():
    print(f"\n[L3b] 端到端 DGEMM 精度对比 (vs torch.matmul aka cuBLAS DGEMM)")
    print(f"{'size':>6} {'slices':>8} {'stage2 max_rel':>18} {'stage4 max_rel':>18} "
          f"{'stage2 ms':>10} {'stage4 ms':>10}")

    cases = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    ok = True
    for (m, k, n) in cases:
        A = torch.rand(m, k, device=DEV, dtype=torch.float64) * 9 + 1
        B = torch.rand(k, n, device=DEV, dtype=torch.float64) * 9 + 1
        C_ref = A @ B

        # warmup
        ozaki_fp8_stage2(A, B)
        ozaki_dgemm_emu(A, B)

        torch.cuda.synchronize(); t = time.perf_counter()
        C_hw, (sA, sB) = ozaki_fp8_stage2(A, B, return_meta=True)
        torch.cuda.synchronize(); t_hw = time.perf_counter() - t

        torch.cuda.synchronize(); t = time.perf_counter()
        C_emu, (sA2, sB2) = ozaki_dgemm_emu(A, B, return_meta=True)
        torch.cuda.synchronize(); t_emu = time.perf_counter() - t

        rel_hw  = ((C_hw  - C_ref).abs() / C_ref.abs().clamp(min=1e-300)).max().item()
        rel_emu = ((C_emu - C_ref).abs() / C_ref.abs().clamp(min=1e-300)).max().item()
        # 同样的算法 + bit-exact 切片 → 末端结果应几乎一致
        emu_passes = rel_emu < 1e-13
        flag = "OK" if emu_passes else "FAIL"
        print(f"  {m:>4}  {sA}x{sB}    {rel_hw:>14.2e}   {rel_emu:>14.2e}   "
              f"{t_hw*1e3:>8.2f}   {t_emu*1e3:>8.2f}   [{flag}]")
        ok = ok and emu_passes
    return ok


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 100)
    print("Stage 4 verify: integer-emulated FP64 in Ozaki Scheme")
    print("=" * 100)

    # Level 1 is run by `python fp64_emu.py` (10004 pass on Python ref, already verified)
    print("[L1] Python ref fp64_add / fp64_mul vs hardware  →  run `python fp64_emu.py`")
    print()

    ok_l2 = test_triton_add()
    print()
    ok_l3a = test_slice_identity()
    ok_l3b = test_e2e_accuracy()

    print()
    print("=" * 100)
    overall = "PASS" if (ok_l2 and ok_l3a and ok_l3b) else "FAIL"
    print(f"Overall: [{overall}]")
    print("=" * 100)


if __name__ == "__main__":
    main()
