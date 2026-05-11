# Stage 3 — Inner-product Blocking

> **本文是 Stage 2 → Stage 3 的 delta 文档**。假设你读过 `../stage2_fp8/PRINCIPLE.md`。
> 本 stage 不改 Ozaki 算法本身，**只在 $k$ 维加一层外层循环**。

---

## 0. 一句话差异

把 $k$ 维切成长度为 $k_{bk}$ 的小块，对每块跑一次完整的 Ozaki 切片 + GEMM + 累加，
块间结果用 FP64 加法叠加。

```
普通 Ozaki (Stage 2):                    Inner-product blocking (Stage 3):
                                         k=0..k_bk → Ozaki → ΔC₁
[A:m×k] [B:k×n]                          k=k_bk..2k_bk → Ozaki → ΔC₂
   ↓ slice + s_A·s_B 次 GEMM                ...
   C                                      C = ΔC₁ + ΔC₂ + ... + ΔC_b   (FP64 求和)
```

---

## 1. 算法

设输入 $A \in \mathbb{R}^{m \times k}$、$B \in \mathbb{R}^{k \times n}$，块大小 $k_{bk}$，分 $b = \lceil k / k_{bk} \rceil$ 块。

$$
C \;=\; \sum_{i=0}^{b-1} A_{[:,\,i k_{bk}:(i+1)k_{bk}]} \cdot B_{[i k_{bk}:(i+1)k_{bk},\,:]}
$$

对每一项内积，**独立**应用 Ozaki Scheme（沿用 Stage 2 的全套机器）。
外层求和在 FP64 算（直接 `C += ΔC_i`）。

伪代码：

```python
C = zeros(m, n, fp64)
for k_start in range(0, k, k_bk):
    A_block = A[:, k_start:k_start+k_bk]
    B_block = B[k_start:k_start+k_bk, :]
    C += ozaki_dgemm(A_block, B_block)   # 内部还是 Stage 2 那套切片+累加
return C
```

---

## 2. 误差分析

**块内**（沿 $k_{bk}$ 维）：Ozaki Scheme 仍然误差自由（论文 Eq. 11 在 $k_{bk}$ 上成立，因为 $\rho$ 是按 $k_{bk}$ 算的）。

**块间**（外层 $\sum$）：每次 `C += ΔC_i` 是 FP64 加法，**有舍入**。

把 $b$ 个相对量级 $\sim O(k_{bk})$ 的中间结果加起来，误差是

$$
\mathrm{err}(C) \;\le\; b \cdot \mathrm{ulp}(C) \;=\; \frac{k}{k_{bk}} \cdot \mathrm{ulp}(C)
$$

这恰好是**标准 FP64 GEMM 的误差量级**（标准 GEMM 也是 $O(k \cdot \mathrm{ulp})$）。
所以 Stage 3 的精度与 cuBLAS DGEMM **同等**，只不过：

- Stage 1 / Stage 2 (no blocking): 误差 $\approx \mathrm{ulp}(C)$（仅累加 $s_A s_B$ 次, 远小于 $k$）
- Stage 3 (with blocking): 误差 $\approx (k/k_{bk}) \cdot \mathrm{ulp}(C)$（约等于标准 DGEMM）

**结论**：blocking 微降精度但仍达标准 FP64 GEMM 水平，可接受。

---

## 3. 为什么需要 Blocking？

论文给了两类动机：

### 3.1 内存（主因）

非 blocking 实现下，所有切片同时驻留显存：

$$
\text{mem}(\text{slices}) \;=\; s_A \cdot mk + s_B \cdot kn \quad\text{bytes (FP8)}
$$

对 $s_A = s_B = 14$、$m = n = k = 16384$，这是 $14 \cdot 16384^2 \cdot 2 \approx 7.5$ GB。
RTX 4090 的 16 GB 装得下但很紧。$k=32768$ 就装不下了。

Blocking 后，**任何时刻只有一块的切片驻留**：

$$
\text{mem per block} \;=\; s_A \cdot m k_{bk} + s_B \cdot k_{bk} n
$$

$k_{bk} = 1024$ 时是 470 MB。$k = 65536$ 也轻松装下。

### 3.2 速度（次因，且依赖实现）

论文里 blocking 还能**减少切片数**：每块的 $k_{bk}$ 比 $k$ 小，
Stage 2 §4 公式里的 $\gamma = m_1 - \lfloor (m_3 - \log_2 k_{bk})/2 \rfloor$ 跟着变小，
当 $\gamma < \xi$ 时 $\rho = \xi$ 不变 → 切片数不变；
当 $\gamma > \xi$ 时 $\rho$ 更小 → 切片数更少。

**对于 FP8 + FP32 这条路**：$\xi = 49$ 主导，$\gamma$ 只有在 $k > 65536$ 才会 catch up。
所以我们的实现里 blocking **不减切片数**，只省内存。

论文 Fig. 5 在 $m = k = n = 16384$ 上扫 $k_{bk} \in \{1024, 4096, 8192, 16384\}$，
最优是 $k_{bk} = 4096$。原因是论文的 $\rho$ 公式更精细，
对小 $k_{bk}$ 真的能减少切片数。

---

## 4. 性能权衡（"为什么不是越小越好"）

| $k_{bk}$ 变小 | 影响 |
|---|---|
| 切片内存 | ⬇ 减少 |
| 单次 GEMM size | ⬇ 减少 → cuBLASLt FP8 吞吐下降（小 GEMM 离不开 peak） |
| 总 GEMM 个数 | $b \times s_A s_B$，**当切片数不变时反而 ⬆ 增加** |
| 累加 kernel 调用数 | ⬆ 增加 → kernel launch 开销变大 |
| Outer FP64 求和误差 | ⬆ 略大但仍 $\le$ 标准 DGEMM 误差 |

所以 $k_{bk}$ 太小会因 launch + 小 GEMM 效率拖慢；太大就退化成 Stage 2。
**最优 $k_{bk}$ 取决于 $m, n, k, $ GPU 架构**，论文没给自动调参，本 stage 也手工扫一遍。

---

## 5. 与 outer-product blocking 的区别

论文 §4.3 提到另一种 blocking 策略 — **outer-product blocking** —
把 $m$ 或 $n$ 维切块。OzBLAS 的早期版本就是这么做的。

| 对比项 | inner-product (本 stage) | outer-product |
|---|---|---|
| 切的方向 | $k$ 维 | $m$ 或 $n$ 维 |
| 每块 GEMM 形状 | $m \times k_{bk}$ · $k_{bk} \times n$ | $m_{bk} \times k$ · $k \times n_{bk}$ |
| 块间累加 | ✗ 需 FP64 加法 | ✓ 完全独立（不同输出区块） |
| 切片是否重复 | $B$ 的同列可能在不同块独立切 | 同一 $k$ 维切片被多块复用 |
| 适用场景 | $k$ 巨大但 $m, n$ 中等 | $m \cdot n$ 输出区巨大 |
| 误差 | 略增（块间舍入） | 零增（块独立） |

实际系统会**两种 blocking 组合用**。Stage 3 只做 inner-product, 保持简单。

---

## 6. 实现注意点

### 6.1 直接 import Stage 2 的部件

Stage 3 的算法 = **Stage 2 + 外层 k-block 循环**。代码上没新 Triton kernel，
直接复用 `stage2_fp8/ozaki_fp8.py` 里的 `slice_matrix_rows`、`matmul_fp8_to_fp32`、
`accumulate_into`、`compute_rho`。

```python
from stage2_fp8.ozaki_fp8 import (
    compute_rho, slice_matrix_rows, matmul_fp8_to_fp32, accumulate_into,
)
```

教学上这是个清晰示范：**Stage 3 是 Stage 2 的纯 host-side 改造**，
不动 GPU kernel 也能拿到全新的能力（处理更大的 $k$）。

### 6.2 切完块就释放切片内存

每块循环结束前显式 `del A_slices, B_slices`，触发 PyTorch 的 caching allocator
释放显存。否则峰值内存可能比理论值高一倍。

### 6.3 A_block / B_block 的连续性

切片之前的 `A[:, k_start:k_end]` 是 strided view，**不**连续。
slicing kernel 期望 contiguous 输入，所以做一次 `.contiguous()`。这是必要拷贝。

### 6.4 累加进入同一个 C

每块循环不要分配新的 ΔC 然后再加。直接把 `accumulate_into(C, G_pq, ...)`
在共用的 `C` 上 in-place 操作。块间求和**隐式**完成在 Triton accumulate kernel 里
（因为 `C += scale·G_pq` 累加的就是同一个 C）。这等价于先 ΔC₁ 再 ΔC₂... 的语义。

---

阅读完毕，进 `ozaki_fp8_kbk.py`。
