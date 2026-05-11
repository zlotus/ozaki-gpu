"""
fp64_emu.py — Pure-integer FP64 emulation
==========================================

实现论文 §4.2 的核心: 仅用 uint32 / uint64 整数运算模拟 IEEE 754 binary64
的算术 (add, sub, mul). 目的是让 Ozaki Scheme 跑在没有 FP64 硬件的处理器上.

本文件分两部分:
  Part A — **Python 参考实现** (本文件主体)
           操作 Python int (无限精度), 算法清晰, 用 struct 做 bit-cast 跟 hardware FP64 对拍.
           作为后面 Triton 端口的 "黄金标准".

  Part B — **Triton @jit 助手** (在 ozaki_fp8_emu.py 中 inline 使用)
           只翻译 fp64_add (slicing kernel 唯一需要的 op).
           严格只用 tl.uint64 / tl.int32, **不出现一次 tl.float64 算术**.

简化假设 (与论文一致):
  - 不处理 subnormals (exp == 0 视为 ±0)
  - 不处理 NaN / inf
  - 不处理 overflow / underflow
  - 输入限于 normal range. 对我们的 Ozaki 测试 (uniform [1, 10] + 残量) 完全够用.
"""

from __future__ import annotations

import struct
import math
from typing import Tuple

# ============================================================================
# Bit-level helpers
# ============================================================================

MASK_52 = (1 << 52) - 1          # 0xFFFFFFFFFFFFF, 52-bit mantissa mask
HIDDEN_BIT = 1 << 52              # 0x10000000000000, the implicit leading 1
EXP_MASK = 0x7FF                  # 11-bit exponent mask
BIAS = 1023


def f64_to_bits(x: float) -> int:
    """double -> uint64 bit pattern (Python int)."""
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def bits_to_f64(b: int) -> float:
    """uint64 bit pattern -> double."""
    return struct.unpack('<d', struct.pack('<Q', b & 0xFFFFFFFFFFFFFFFF))[0]


def unpack(bits: int) -> Tuple[int, int, int]:
    """Extract (sign, biased_exp, mantissa_53bit_with_hidden).

    For exp==0 (zero/subnormal in our simplification), returns mantissa=0.
    """
    sign = (bits >> 63) & 1
    exp = (bits >> 52) & EXP_MASK
    if exp == 0:
        return sign, 0, 0
    mant = (bits & MASK_52) | HIDDEN_BIT  # 53 bits
    return sign, exp, mant


def pack(sign: int, exp: int, mant_53: int) -> int:
    """Inverse of unpack. mant_53 should be normalized (top bit at position 52),
    exp is biased. Caller's responsibility to ensure no overflow.
    """
    # Strip the hidden bit; keep 52 stored bits
    stored_mant = mant_53 & MASK_52
    return ((sign & 1) << 63) | ((exp & EXP_MASK) << 52) | stored_mant


# ============================================================================
# §A.1 FP64 multiplication (paper Listing 1 adapted)
# ============================================================================
#
# 论文 Listing 1 的 `mul_mantissa` 用 uint32×4 schoolbook 把 53-bit 乘以 53-bit
# 得到 ~106-bit 乘积. Python 里 `a_mant * b_mant` 已经精确, 但我们**显式**走
# uint32×4 那条路 — 这恰好是 Triton 端要做的事, 也是论文展示的那段代码.

def _mul_mantissa(a_mant: int, b_mant: int) -> int:
    """53×53 mantissa multiplication via uint32×4 schoolbook (paper Listing 1).

    Returns the 106-bit product as a Python int.

    Algorithm (a = a_hi || a_lo, b = b_hi || b_lo where lo is 32 bits):
        product = (a_hi · b_hi) · 2^64
                + (a_hi · b_lo + a_lo · b_hi) · 2^32
                + (a_lo · b_lo)

    Each partial product fits in uint64:
      - a_lo, b_lo  : 32 bits → product 64 bits
      - a_hi, b_hi  : ≤ 21 bits (since mantissa is 53-bit) → product ≤ 42 bits
      - cross terms : 21·32 = 53 bits each, sum ≤ 54 bits, fits in uint64 cleanly
    """
    UMASK = 0xFFFFFFFF
    a_lo = a_mant & UMASK
    a_hi = a_mant >> 32   # ≤ 21 bits for 53-bit mantissa
    b_lo = b_mant & UMASK
    b_hi = b_mant >> 32

    p00 = a_lo * b_lo     # ≤ 64 bits
    p01 = a_lo * b_hi     # ≤ 53 bits
    p10 = a_hi * b_lo
    p11 = a_hi * b_hi     # ≤ 42 bits

    # Combine: 128-bit result = (high << 64) | low
    middle = p01 + p10                              # ≤ 54 bits, no overflow
    low_result = p00 + ((middle & UMASK) << 32)     # may exceed 64 bits in worst case
    carry = (low_result >> 64) & 1
    low_result &= ((1 << 64) - 1)
    high_result = p11 + (middle >> 32) + carry      # ≤ 43 bits

    product = (high_result << 64) | low_result      # 106 bits at most
    return product


def fp64_mul(a_bits: int, b_bits: int) -> int:
    """FP64 multiplication a*b, IEEE round-to-nearest-even, integer-only.

    Caveats: no subnormals/NaN/inf/overflow handling.
    """
    a_sign, a_exp, a_mant = unpack(a_bits)
    b_sign, b_exp, b_mant = unpack(b_bits)

    result_sign = a_sign ^ b_sign

    # Zero handling (assuming no subnormals: exp=0 ⇒ value=0)
    if a_exp == 0 or b_exp == 0:
        return result_sign << 63

    # Multiply mantissas (uint32×4 schoolbook)
    product = _mul_mantissa(a_mant, b_mant)
    new_exp = a_exp + b_exp - BIAS    # unbias one of them

    # Normalize: product is 105 or 106 bits
    #   - top bit at 105 ⇒ 106-bit case (need +1 to exp, drop bottom 53 bits)
    #   - top bit at 104 ⇒ 105-bit case (no exp change, drop bottom 52 bits)
    if product >> 105:
        # 106-bit case
        new_exp += 1
        true_mant_53 = product >> 53
        round_bit  = (product >> 52) & 1
        sticky     = (product & ((1 << 52) - 1)) != 0
    else:
        # 105-bit case
        true_mant_53 = product >> 52
        round_bit  = (product >> 51) & 1
        sticky     = (product & ((1 << 51) - 1)) != 0

    # Round-to-nearest-even
    lsb = true_mant_53 & 1
    round_up = (round_bit == 1) and (sticky or lsb == 1)
    if round_up:
        true_mant_53 += 1
        if true_mant_53 >> 53:    # mantissa overflowed → renormalize
            true_mant_53 >>= 1
            new_exp += 1

    return pack(result_sign, new_exp, true_mant_53)


# ============================================================================
# §A.2 FP64 addition (the hard one — alignment + GRS rounding + normalize)
# ============================================================================
#
# 思路:
#  1. 保证 |a| ≥ |b|, 必要时 swap.
#  2. 把两边 mantissa 各往 LSB 方向移 3 位, 留 G/R/S 三个 round/sticky 位.
#  3. 按 exp_diff 把 b 对齐到 a 的指数, 移出的低位 OR 进 sticky.
#  4. 同号加, 异号减.
#  5. 加: 若 carry 出位, 右移 1, 调整 exp; 减: 左移直到首位归位.
#  6. round-to-nearest-even by GRS + LSB.

def fp64_add(a_bits: int, b_bits: int) -> int:
    """FP64 addition, RTNE. Integer-only. Skips subnormals/NaN/inf."""
    a_sign, a_exp, a_mant = unpack(a_bits)
    b_sign, b_exp, b_mant = unpack(b_bits)

    if a_exp == 0 and b_exp == 0:
        return 0
    if a_exp == 0:
        return b_bits
    if b_exp == 0:
        return a_bits

    # Step 1: ensure |a| ≥ |b|
    if (a_exp, a_mant) < (b_exp, b_mant):
        a_sign, b_sign = b_sign, a_sign
        a_exp,  b_exp  = b_exp,  a_exp
        a_mant, b_mant = b_mant, a_mant

    # Step 2: extend to 56-bit (mantissa shifted left 3 to leave room for G/R/S below)
    #   Bit layout after << 3:
    #     bit 55 = hidden bit
    #     bits 54..3 = 52 stored mantissa bits
    #     bits 2..0 = 0 (room for G/R/S below)
    a_ext = a_mant << 3
    b_ext = b_mant << 3

    # Step 3: align b right by exp_diff, accumulate sticky from shifted-out bits
    exp_diff = a_exp - b_exp
    sticky_from_align = 0
    if exp_diff > 0:
        if exp_diff >= 64:
            b_aligned = 0
            sticky_from_align = 1 if b_ext != 0 else 0
        else:
            mask = (1 << exp_diff) - 1
            shifted_out = b_ext & mask
            b_aligned = b_ext >> exp_diff
            sticky_from_align = 1 if shifted_out != 0 else 0
    else:
        b_aligned = b_ext

    result_sign = a_sign

    # Step 4: same-sign add OR different-sign subtract
    if a_sign == b_sign:
        result = a_ext + b_aligned
        sticky_acc = sticky_from_align
        # Step 5a: carry into bit 56? Shift right 1 and capture LSB as sticky.
        if result >> 56:
            sticky_acc |= result & 1
            result >>= 1
            a_exp += 1
        # result is in [2^55, 2^56), normalized
    else:
        # |a| >= |b|; we subtract b_aligned + sticky portion from a_ext.
        # If sticky_from_align == 1, b's true value > b_aligned (by some sub-LSB amount),
        # so we owe an extra 1 ULP. After subtracting that, the residue (1 - tail) is a
        # positive sub-LSB amount → recorded as sticky on the result.
        if sticky_from_align:
            result = a_ext - b_aligned - 1
            sticky_acc = 1
        else:
            result = a_ext - b_aligned
            sticky_acc = 0

        if result == 0 and sticky_acc == 0:
            return 0     # exact cancellation

        # Step 5b: normalize (shift left until top bit at position 55).
        # Bounded by ~56 iters even in catastrophic cancellation.
        while result != 0 and not (result >> 55):
            result <<= 1
            a_exp -= 1

        # If result became 0 but sticky != 0, we have a tiny residue — would be subnormal,
        # which we don't model. Just return 0.
        if result == 0:
            return result_sign << 63

    # Step 6: round-to-nearest-even using GRS bits at positions 2, 1, 0 (+ sticky_acc)
    guard_bit  = (result >> 2) & 1
    round_bit  = (result >> 1) & 1
    sticky_bit = (result & 1) | sticky_acc
    lsb_kept   = (result >> 3) & 1

    if guard_bit == 0:
        round_up = False
    elif round_bit == 1 or sticky_bit == 1:
        round_up = True                       # strictly above half
    else:
        round_up = (lsb_kept == 1)            # tie → round to even

    result_53 = result >> 3
    if round_up:
        result_53 += 1
        if result_53 >> 53:                   # mantissa overflowed (53 → 54 bits)
            result_53 >>= 1
            a_exp += 1

    return pack(result_sign, a_exp, result_53)


def fp64_sub(a_bits: int, b_bits: int) -> int:
    """a - b = a + (-b). Just flip b's sign bit."""
    return fp64_add(a_bits, b_bits ^ (1 << 63))


# ============================================================================
# Bit-exact unit test against hardware FP64
# ============================================================================

def _hw_mul(a: float, b: float) -> int:
    return f64_to_bits(a * b)


def _hw_add(a: float, b: float) -> int:
    return f64_to_bits(a + b)


def _run_self_test():
    """Bit-exact comparison against hardware FP64 over a battery of test cases."""
    import random
    random.seed(0xC0FFEE)

    n_tested = 0
    n_failed = 0

    def case(a: float, b: float, label: str = ""):
        nonlocal n_tested, n_failed
        a_bits = f64_to_bits(a)
        b_bits = f64_to_bits(b)
        # Skip subnormals / NaN / inf (we don't support them)
        for x_bits in (a_bits, b_bits):
            exp = (x_bits >> 52) & EXP_MASK
            if exp == 0 or exp == EXP_MASK:
                return

        emu_mul = fp64_mul(a_bits, b_bits)
        hw_mul = _hw_mul(a, b)
        emu_add = fp64_add(a_bits, b_bits)
        hw_add = _hw_add(a, b)

        # Skip if hw result is subnormal/inf/nan (out of our scope)
        for r in (hw_mul, hw_add):
            exp = (r >> 52) & EXP_MASK
            if exp == 0 or exp == EXP_MASK:
                return

        n_tested += 1
        if emu_mul != hw_mul:
            n_failed += 1
            print(f"  MUL MISMATCH [{label}]: a={a!r}, b={b!r}")
            print(f"    emu  = 0x{emu_mul:016x} = {bits_to_f64(emu_mul)!r}")
            print(f"    hw   = 0x{hw_mul:016x} = {bits_to_f64(hw_mul)!r}")
        if emu_add != hw_add:
            n_failed += 1
            print(f"  ADD MISMATCH [{label}]: a={a!r}, b={b!r}")
            print(f"    emu  = 0x{emu_add:016x} = {bits_to_f64(emu_add)!r}")
            print(f"    hw   = 0x{hw_add:016x} = {bits_to_f64(hw_add)!r}")

    # 1. Hand-picked easy cases
    for (a, b) in [(1.0, 1.0), (1.5, 2.0), (1.0, -1.0), (1.0, 0.5),
                   (3.14, 2.71), (-3.14, 2.71),
                   (1e10, 1e-10), (1e-10, 1e10),
                   (1.0 + 2**-52, 1.0)]:
        case(a, b, "hand")

    # 2. Random uniform in [1, 10] — our Ozaki test range
    for _ in range(2000):
        a = random.uniform(1, 10)
        b = random.uniform(1, 10)
        case(a, b, "uniform[1,10]")
        case(a, -b, "uniform[1,10] mix sign")

    # 3. Large dynamic range (test alignment / sticky)
    for _ in range(2000):
        ea = random.randint(-100, 100)
        eb = random.randint(-100, 100)
        a = random.uniform(0.5, 2) * (2.0 ** ea)
        b = random.uniform(0.5, 2) * (2.0 ** eb)
        if random.random() < 0.5: b = -b
        case(a, b, f"wide e=({ea},{eb})")

    # 4. Near-cancellation (challenging for add)
    for _ in range(1000):
        a = random.uniform(1, 10)
        # b very close to -a (or a)
        b = -a * (1 + random.uniform(-1e-10, 1e-10))
        case(a, b, "cancellation")

    # 5. Specific Ozaki-relevant: sigma-like values (0.75 · 2^e) added to small x
    for _ in range(1000):
        e = random.randint(40, 55)
        sigma = 0.75 * (2.0 ** e)
        x = random.uniform(1, 10)
        case(x, sigma, "Ozaki sigma")
        case(x, -sigma, "Ozaki -sigma")
        case(x + sigma, -sigma, "Ozaki (x+σ)-σ")

    print(f"\nfp64 emu self-test: {n_tested - n_failed} / {n_tested} pass, {n_failed} fail")
    return n_failed == 0


# ============================================================================
# Part B — Triton @jit FP64 add helper (uint64-only)
# ============================================================================
#
# 严格逐字翻译上面 `fp64_add` 的 Python 参考, 用 tl.uint64 / tl.int32 操作.
# 唯一变化: normalize 那个 `while` 循环 → 56 次 unrolled `tl.where` 条件移位
# (因为 Triton 3.6 没有 tl.math.clz, 也没法 break-out-of-loop).
#
# 这个 @triton.jit 函数被其他 kernel 调用时 inline 展开 — 没有 device function call 开销.

import triton
import triton.language as tl


@triton.jit
def emu_fp64_add(a_bits, b_bits):
    """Emulated FP64 add in Triton, IEEE round-to-nearest-even.

    Args / Return: all `tl.uint64` (block-shaped). Algorithm matches fp64_add() above
    but uses only integer operations (no tl.float64 arithmetic).

    Caveats inherited from the Python ref: no subnormals / NaN / inf / overflow.
    """
    # ─── Unpack ───────────────────────────────────────────────────────────
    a_sign = (a_bits >> 63) & 1
    a_exp  = (a_bits >> 52) & 0x7FF
    a_mant = (a_bits & 0xFFFFFFFFFFFFF) | (1 << 52)    # 53-bit

    b_sign = (b_bits >> 63) & 1
    b_exp  = (b_bits >> 52) & 0x7FF
    b_mant = (b_bits & 0xFFFFFFFFFFFFF) | (1 << 52)

    a_zero = a_exp == 0
    b_zero = b_exp == 0
    # If either is "zero" (exp=0 in our simplified model) we'll patch the result at the end

    # ─── Step 1: ensure |a| >= |b| (sort by exp then mantissa, tl.where swap) ──
    # 在 uint64 域比较: (exp, mant) lexicographic compare.
    a_is_bigger = (a_exp > b_exp) | ((a_exp == b_exp) & (a_mant >= b_mant))

    big_sign = tl.where(a_is_bigger, a_sign, b_sign)
    big_exp  = tl.where(a_is_bigger, a_exp,  b_exp )
    big_mant = tl.where(a_is_bigger, a_mant, b_mant)

    sml_sign = tl.where(a_is_bigger, b_sign, a_sign)
    sml_exp  = tl.where(a_is_bigger, b_exp,  a_exp )
    sml_mant = tl.where(a_is_bigger, b_mant, a_mant)

    # ─── Step 2: extend to 56-bit (mantissa << 3 leaves room for G/R/S below) ─
    big_ext = big_mant << 3
    sml_ext = sml_mant << 3

    # ─── Step 3: align sml right by exp_diff, track sticky from shifted-out bits ──
    exp_diff = big_exp - sml_exp                         # >= 0

    # Triton 的变量量移位安全范围是 [0, 63]; >=64 视为 "全部移出"
    safe_shift = tl.minimum(exp_diff, 63)
    shifted_out = sml_ext & ((1 << safe_shift) - 1)
    sml_aligned = sml_ext >> safe_shift
    # 当 exp_diff >= 64 时, sml 整个被移出 — 全部进入 sticky
    out_of_range = exp_diff >= 64
    sml_aligned = tl.where(out_of_range, tl.zeros_like(sml_aligned), sml_aligned)
    sticky_from_align = tl.where(out_of_range,
                                 (sml_ext != 0).to(tl.uint64),
                                 (shifted_out != 0).to(tl.uint64))

    result_sign = big_sign

    # ─── Step 4: same-sign add  /  diff-sign sub ──────────────────────────
    same_sign = big_sign == sml_sign

    # 4a: same-sign add result
    add_result = big_ext + sml_aligned
    add_carry = (add_result >> 56) & 1                  # 是否进位到 bit 56
    add_sticky = sticky_from_align | (add_result & add_carry)   # 进位时把 LSB 当 sticky
    add_result_norm = tl.where(add_carry == 1, add_result >> 1, add_result)
    add_exp_adj = add_carry.to(tl.int32)                # +1 if carry

    # 4b: diff-sign subtract result
    # sml_aligned 不含 sticky_from_align 那一 ULP-以下的残量;
    # 若 sticky_from_align=1, 真实 sml 比 sml_aligned 多 (一点点), 要多减 1 ULP,
    # 余下的 (1 - tail) 形成新的 sub-ULP 残量 → sticky_acc = 1
    sub_borrow = sticky_from_align    # 0 or 1
    sub_result = big_ext - sml_aligned - sub_borrow
    sub_sticky_acc = sticky_from_align    # 1 if borrowed, else 0

    # ─── Step 5: pick add/sub branch & normalize ──────────────────────────
    result = tl.where(same_sign, add_result_norm, sub_result)
    sticky_acc = tl.where(same_sign, add_sticky, sub_sticky_acc)
    exp_adj = tl.where(same_sign, add_exp_adj, tl.zeros_like(add_exp_adj))
    result_exp_signed = big_exp.to(tl.int32) + exp_adj

    # —— normalize: 仅在减法分支(可能首位 < 55) 时需要左移直到首位归位 ——
    # 用 56 次 unrolled tl.where 实现 (Triton 没有 tl.math.clz)
    for _ in tl.static_range(56):
        # 当 result != 0 且 result's top bit < 55 时, 左移 1, exp -= 1
        needs_shift = (result != 0) & ((result >> 55) == 0)
        result = tl.where(needs_shift, result << 1, result)
        result_exp_signed = tl.where(needs_shift, result_exp_signed - 1, result_exp_signed)

    # ─── Step 6: round-to-nearest-even (GRS bits at positions 2, 1, 0) ────
    guard_bit  = (result >> 2) & 1
    round_bit  = (result >> 1) & 1
    sticky_bit = (result & 1) | sticky_acc
    lsb_kept   = (result >> 3) & 1

    # 三态: guard=0 → 下舍; guard=1 且 (round|sticky) → 上舍; guard=1 且其余 → tie, 看 lsb_kept
    is_above_half = (guard_bit == 1) & ((round_bit == 1) | (sticky_bit == 1))
    is_tie        = (guard_bit == 1) & (round_bit == 0) & (sticky_bit == 0)
    round_up = is_above_half | (is_tie & (lsb_kept == 1))

    result_53 = (result >> 3) + round_up.to(tl.uint64)
    # round_up 后可能从 53 bit 溢出到 54 bit, 此时右移 1 + exp += 1
    overflow = (result_53 >> 53) & 1
    result_53 = tl.where(overflow == 1, result_53 >> 1, result_53)
    result_exp_signed = result_exp_signed + overflow.to(tl.int32)

    # ─── Pack ──────────────────────────────────────────────────────────────
    # 特殊: 若 result==0 (减法精确抵消), 输出 ±0
    is_zero = (result == 0) & (sticky_acc == 0)
    final_exp = tl.where(is_zero, tl.zeros_like(result_exp_signed),
                          result_exp_signed).to(tl.uint64) & 0x7FF
    final_mant = result_53 & 0xFFFFFFFFFFFFF
    final_bits = (result_sign << 63) | (final_exp << 52) | final_mant

    # 处理 a 或 b 为零的特殊情况 (直接返回另一边)
    final_bits = tl.where(a_zero, b_bits, final_bits)
    final_bits = tl.where(b_zero, a_bits, final_bits)
    return final_bits


@triton.jit
def emu_fp64_sub(a_bits, b_bits):
    """a - b via emu_fp64_add(a, -b). Sign flip on b's MSB."""
    return emu_fp64_add(a_bits, b_bits ^ (1 << 63))


if __name__ == "__main__":
    ok = _run_self_test()
    if not ok:
        raise SystemExit(1)
