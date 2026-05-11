# Stage 4 — 用整数运算模拟 FP64 (论文 §4.2)

> 本文是 Stage 2 → Stage 4 的 delta 文档。假设你已经吃透 Stage 1、Stage 2、Stage 3。
>
> **核心命题**: GPU kernel 里 *一条 FP64 算术指令都不用*，仍然能完成 Ozaki Scheme。

---

## 0. 为什么要做这个

论文 §4.2 给的动机：

- 现代 AI 卡 (GB200 / GB300 / AMD MI350) 的 FP64 算力已经被砍到接近无；GB300 完全无 FP64 Tensor Core
- Ozaki Scheme 本来就只在 slicing + accumulation 处用 FP64，**这部分原则上完全可以用整数算术模拟**
- 这样 DGEMM 可以彻底跑在"FP64-less"硬件上

教学上这是个**漂亮的反推**：当我们用了 FP8/FP16 Tensor Core 加速 GEMM 主体后，剩下的 FP64 都是位级 housekeeping，完全可以"假装它不存在"。

---

## 1. FP64 (IEEE 754 binary64) bit layout 速览

```
 63 62                  52 51                                                   0
 ┌──┬─────────────────────┬─────────────────────────────────────────────────────┐
 │ S│   biased exp (11)   │              stored mantissa (52)                   │
 └──┴─────────────────────┴─────────────────────────────────────────────────────┘
```

- `value = (-1)^S · 2^(exp - 1023) · (1.m₅₁m₅₀...m₀)₂`   （normal 数）
- "53-bit mantissa" = 隐含的前导 1 + 存储的 52 bits
- bias = 1023, normal 指数范围 [1, 2046] (biased)

简化假设（与论文一致）:
- 不处理 subnormals (exp == 0 视为 ±0)
- 不处理 NaN / inf
- 不处理 overflow

对 Ozaki 的输入 (uniform [1, 10]) 和中间残量, 这些假设都成立。

---

## 2. FP64 加法 — 难点

加法的"整数算法"由 5 个 step 组成。变量名引用 `fp64_emu.py` 中的实现:

### Step 1. Unpack & swap

确保 `|a| ≥ |b|`。比较 `(a_exp, a_mant) ≥ (b_exp, b_mant)` 字典序——对正 FP64 来说，bit pattern 顺序就是数值顺序。

### Step 2. Extend 到 56-bit

把 53-bit mantissa 左移 3 位，腾出 **G/R/S** 三个位置：

```
                          ┌─ G (guard)
                          │ ┌─ R (round)
                          │ │ ┌─ S (sticky)
 55 54                  3 │ │ │
 ┌──┬───────────────────┬─┴─┴─┴┐
 │ 1│ stored mant (52)  │ 0 0 0│   ← 56-bit ext
 └──┴───────────────────┴──────┘
```

### Step 3. Align b → a 的指数

`exp_diff = a_exp - b_exp`. 把 b_ext 右移 `exp_diff` 位, **移出去的非零位 OR 进 sticky**:

```python
shifted_out = b_ext & ((1 << exp_diff) - 1)
b_aligned   = b_ext >> exp_diff
sticky_from_align = (shifted_out != 0)
```

边界: `exp_diff ≥ 64` 时 b 整个被移出, 只剩 sticky.

### Step 4. 同号 add / 异号 sub

**同号**: `result = a_ext + b_aligned`. 可能进位到 bit 56, 此时右移 1 + 指数 +1, 被移出的 LSB 进 sticky.

**异号** (`|a| > |b|` 已保证): 
```python
if sticky_from_align == 1:
    # b 真实值比 b_aligned 多一点 sub-LSB 残量
    # 多减 1, 留下 (1 - tail) 作为新 sticky
    result = a_ext - b_aligned - 1
    sticky_acc = 1
else:
    result = a_ext - b_aligned
    sticky_acc = 0
```

> 这里 sticky 的语义最容易出错。如果你看 Python ref `fp64_emu.py:fp64_add` 的 `# Step 4` 注释, 它写得最清楚 — 关键观察是: 当 `sticky_from_align = 1` 时, 我们"多减一 ULP, 但产生了一个 sub-LSB 残量, 这残量对最终舍入有影响, 所以记为 sticky"。

### Step 5. Normalize (减法 catastrophic cancellation)

异号减法后 result 可能远远小于 a (e.g., `a ≈ b` 时几乎全抵消), 需要左移直到首位回到 bit 55:

```python
while result != 0 and not (result >> 55):
    result <<= 1
    exp -= 1
```

最多 56 次。Triton 没有 `while`, 用 **静态 unroll 56 次的条件移位** 实现 (见 §4)。

### Step 6. Round-to-nearest-even

读取 GRS + LSB:

```
guard = bit 2 of result
round = bit 1 of result
sticky_bit = (bit 0 of result) | sticky_acc
lsb_kept = bit 3 of result
```

判定:
- `guard == 0`: round down (no change)
- `guard == 1` 且 `round | sticky != 0`: round up (strictly above half)
- `guard == 1` 且 `round == sticky == 0`: tie → round to even (only round up if `lsb_kept == 1`)

最后丢掉低 3 位 + 隐含 bit, 打包 sign / exp / 52-bit stored mantissa.

---

## 3. FP64 乘法 — 论文 Listing 1

53-bit × 53-bit → 106-bit 乘积. CPU/GPU 整数最多 64-bit, 所以 **schoolbook 拆 uint32×4**:

设 $a = a_{hi} \cdot 2^{32} + a_{lo}$, $b$ 同理:

$$
a \cdot b \;=\; \underbrace{(a_{hi} b_{hi})}_{p_{11}} \cdot 2^{64} + \underbrace{(a_{hi} b_{lo} + a_{lo} b_{hi})}_{\text{middle}} \cdot 2^{32} + \underbrace{a_{lo} b_{lo}}_{p_{00}}
$$

每个 partial product 都 ≤ 64 bits, 组合时要处理跨 32-bit 边界的进位。论文 Listing 1 是逐字 C++ 实现, 我们的 Python ref `_mul_mantissa` 在 `fp64_emu.py` 里也照抄。

之后:

1. **Normalize**: 乘积 105 或 106 bits, 看 top bit 在哪决定指数 ±1
2. **Round**: 取顶 53 位, 用 round bit + sticky 做 RTNE
3. **Pack**: 标准 IEEE 字段拼接

Ozaki slicing 内部其实**用不到 FP64 乘法** — `v · inv_2c` 中 `inv_2c = 2^(-c)` 是纯指数, 等价于 uint64 减 `c << 52`. 但论文 Listing 1 是 §4.2 的招牌, 我们仍把它完整实现在 `fp64_emu.py` 里作为教学示例 + bit-exact 测试样本。

---

## 4. Triton 端口的两个关键决策

### 4.1 Normalize 循环 → 56 次静态 unroll

Triton 3.6 没有 `tl.math.clz` (count-leading-zeros), 也不允许 `while` 动态循环。
所以 normalize 步骤变成:

```python
for _ in tl.static_range(56):
    needs_shift = (result != 0) & ((result >> 55) == 0)
    result = tl.where(needs_shift, result << 1, result)
    exp    = tl.where(needs_shift, exp - 1, exp)
```

每次迭代每元素**条件移位**, 56 次后必定归位。代价是 ~56 × 3 个 tl.where + shift, 比硬件 FP64 add 慢一两个数量级 — 但只用在 slicing kernel, 不在 GEMM hot loop, 总体仍可接受。

### 4.2 Slice 末端 cvt 到 FP8 仍用硬件 cvt 指令

整个 emu kernel 里, 唯一**不**用整数实现的就是最后:

```python
v_scaled_f64 = v_scaled_bits.to(tl.float64, bitcast=True)
slice_v = v_scaled_f64.to(tl.float32).to(tl.float8e4nv)
```

严格来说 `.to(tl.float32)` 和 `.to(tl.float8e4nv)` 都是 FP64-类型的硬件 cvt 指令,
也可以再用整数位运算 emulate (大约 +50 行: 提取 fp64 字段 → 新 exp = exp_unbiased + fp8_bias → 取顶 3 位 mantissa + RTNE round → 打包 FP8)。
但这部分**算法上不新增内容**, 只是 FP64 → FP8 的位级翻译, 教学价值低。本文档把它留为"reading exercise"。

---

## 5. 主机端 sigma / c 的整数预计算

Stage 2 主机端用了:
```python
c = torch.ceil(torch.log2(row_max))
sigma = 0.75 * torch.exp2(rho + c)
```
都是 FP64 算术。

Stage 4 (`ozaki_fp8_emu.py:per_row_preparation`) 全用位运算:

```python
# |x|: 清 sign bit
abs_bits = X.view(int64) & 0x7FFFFFFFFFFFFFFF
# row_max(|x|): 正 FP64 的 bit pattern 顺序就是数值顺序 → int 比较 amax 即可
row_max_bits = abs_bits.amax(dim=1)
# ceil(log2(row_max)): biased_exp - 1023 + (mantissa != 0)
biased_exp = (row_max_bits >> 52) & 0x7FF
c = biased_exp - 1023 + (row_max_bits & 0xFFFFFFFFFFFFF != 0).int()
# σ = 0.75 · 2^(rho+c) → IEEE bits: ((rho+c-1+1023) << 52) | (1 << 51)
sigma_bits = ((rho + c - 1 + 1023) << 52) | (1 << 51)
```

这些都是 GPU 上的整数 op, 不触发 FP64 ALU.

---

## 6. 为什么暂不 emulate accumulation kernel

Stage 4 主路径继续用 Stage 2 的 `accumulate_kernel`, 内部含 FP64 加法 (`c_old + scale * g`) 和 FP64 乘法 (`pow2_cA * pow2_cB * G`). 严格"FP64-less"则需要也替换为 emulated 版本。

不做的理由:
- **算法上是 §2 / §3 的镜像** — accumulation kernel 的 FP64 op 是同一套 add/mul, 只是 fuse 在 tl.where + 累加循环里
- 实现上多 ~200 行, 主要是工程苦劳
- **教学讯息已通过 slicing 达成** — 即 "可以彻底用整数 emulate, bit-exact"

留为读者扩展。

---

## 7. 实测验证

`verify.py` 三层验证:

### Level 1: Python ref vs 硬件 (在 `fp64_emu.py` 内置)
```
fp64 emu self-test: 10004 / 10004 pass, 0 fail
```
覆盖: hand-picked / uniform[1,10] / 宽 exponent / 近抵消 / Ozaki σ 模式

### Level 2: Triton emu_fp64_add vs Python ref / 硬件
```
[L2] Triton emu_fp64_add vs hardware FP64 :  2497/2497  [OK]
[L2] Triton emu_fp64_add vs Python ref    :  2497/2497  [OK]
```

### Level 3a: emu slicing 与 Stage 2 slicing 逐 FP8 字节相同
```
slice[0..11]: FP8 diff bytes = 0, c match = True   (全 OK)
```

### Level 3b: 端到端 Ozaki DGEMM
```
size   slices     stage2 max_rel     stage4 max_rel
 256   12x12         2.21e-15            2.21e-15        ← 完全相同
 512   12x12         2.90e-15            2.90e-15
1024   12x12         3.97e-15            3.97e-15
```

末端精度 **逐位相同** — 因为切片是 bit-exact 的, 后面的 FP8 GEMM + FP64 accumulation 又对每个切片确定性运算, 所以总结果必然位级一致。

性能上 Stage 4 比 Stage 2 慢约 1.5-2× (slicing kernel 多 56 次 unrolled 循环 + emu_fp64_add 内部 ~50 ops), 仍处可接受范围。
