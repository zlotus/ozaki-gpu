# Stage 2 — Ozaki Scheme + FP8 (E4M3) Tensor Core 模拟 DGEMM

> **本文是 Stage 1 → Stage 2 的 delta 文档**，假设你已经读过 `../stage1_fp16/PRINCIPLE.md`。
> 算法骨架完全相同，本文只讲"哪里变了、为什么变了、变了以后会怎样"。

---

## 0. 一句话差异

把 Type2 从 FP16 换成 FP8 (E4M3)。**算法不变**，但：

- 每片可承载的有效位数从 **11 bits → 4 bits**
- 切片数从 $\sim 8 \times 8$ 翻到 $\sim 14 \times 14$（GEMM 数从 $\sim 64$ → $\sim 196$）
- 单次 GEMM 在硬件上快 **2×**（FP8 TC 比 FP16 TC 吞吐高一倍）
- 论文里 FP8 路径在大 $k$（$\gtrsim 8192$）上能反超 FP16；本 stage 在 $k \lesssim 4096$ 上**不会更快**（详见 §3 时间复杂度分析）

为什么仍然要做？— **为 Stage 3 (inner-product blocking) 打底**。Stage 3 会把大的 $k$ 切成小块，每块独立做 Ozaki，那时 FP8 的优势才能展现。本 stage 先把 FP8 路径走通。

---

## 1. FP8 (E4M3) 数值格式速览

| 字段 | 位数 | 说明 |
|---|---|---|
| sign | 1 | |
| exponent | 4 | bias = 7 |
| mantissa | 3 (+ 1 隐含位) | $m_2 = 4$ |

特殊值：
- **没有 ±∞**（"FN" = Finite, No-inf）
- 只有 NaN（位模式 `S.1111.111`）
- 最大正规数：$(1 + 7/8) \cdot 2^8 = 448$
- 最小正规数：$2^{-6} = 0.015625$
- 在 $[1, 2)$ 区间的步长：$2^{-3} = 0.125$ ← **这就是切片之间的分辨率**

PyTorch 名字：`torch.float8_e4m3fn`
Triton 名字：`tl.float8e4nv`（"nv" 表示 NVIDIA 变种，无 inf）

---

## 2. $\rho$ 变了

回顾 Stage 1 §4 公式：

$$\rho = \max(\xi,\;\gamma),\qquad \xi = m_1 - m_2,\qquad \gamma = m_1 - \big\lfloor (m_3 - \log_2 k)/2 \big\rfloor$$

把 $m_2$ 从 11 换成 4：

| 量 | Stage 1 (FP16) | Stage 2 (FP8) |
|---|---|---|
| $m_2$ | 11 | **4** |
| $\xi = m_1 - m_2$ | 42 | **49** |
| bits per slice ($\approx m_1 - \rho$) | 11 (when $k$ 小) | **4** (when $k$ 小) |
| 切片数 $s_x = \lceil m_1 / \text{bits} \rceil$ | 5–9 | **14**（$k \le 65536$ 范围内基本恒定） |

**有趣的现象**：FP8 切片数对 $k$ 不敏感（直到 $k > 65536$ 才开始增长），因为 $\xi = 49$ 在大多数情况下 dominate $\gamma$。Stage 1 在 $k$ 从 128 到 2048 上 $s_x$ 从 7 增到 9，Stage 2 大概率全程 $s_x = 14$。

> **关于我们的 $\rho$ 比论文 Table 3 保守**：论文 FP8/FP32 在 $k=128$ 时给出 121 GEMM = 11×11，我们的公式估算 14×14 = 196。差距来自我们没有精确建模 $(x+\sigma)-\sigma$ 操作的"位重叠"（精确公式见 Ozaki 2012 原论文）。
> 对**算法正确性**没影响，只是多做 ~60% GEMM。

---

## 3. 时间复杂度对比

设 $G_T$ 表示一次 GEMM 的耗时。

| | Stage 1 (FP16) | Stage 2 (FP8) | 比例 |
|---|---|---|---|
| GEMM 次数 | $\sim 64$ ($k=1024$) | $\sim 196$ | $3.1\times$ 更多 |
| 单次 GEMM 时间 | $G_T^{16}$ | $G_T^8 \approx 0.5 \cdot G_T^{16}$ | $0.5\times$ |
| 总 GEMM 时间 | $64 \cdot G_T^{16}$ | $196 \cdot 0.5 \cdot G_T^{16} = 98 \cdot G_T^{16}$ | $\mathbf{1.53\times}$ 更慢 |

**这说明**：在保守的 $\rho$ 下，FP8 路径 **在小到中等的 $k$ 上不会更快**。要解锁 FP8 的速度优势，需要：
1. 更紧的 $\rho$ 公式（精确建模重叠位） → 减少切片数到 $\sim 11$
2. 或者 **inner-product blocking** (Stage 3) → 把大 $k$ 切成小块，让 FP8 在每个小块上跑

---

## 4. 代码变化清单（Stage 1 → Stage 2）

实际写下来踩了三个坑，整理出来的"真正改动"如下：

### 4.1 算法层（"应该是这样"）

| 文件 / 位置 | Stage 1 | Stage 2 |
|---|---|---|
| `compute_rho` 默认参数 | `m_type2=11 (M_FP16)` | **`m_type2=4 (M_FP8_E4M3)`** |
| `slice_iter_kernel` 末尾 cvt | `.to(tl.float16)` | **`.to(tl.float32).to(tl.float8e4nv)`** ← 见 §4.2 第 1 坑 |
| `slice_matrix_rows` 输出 dtype | `torch.float16` | **`torch.float8_e4m3fn`** |
| GEMM 后端 | Triton FP16 kernel | **`torch._scaled_mm` (cuBLASLt)** ← 见 §4.2 第 2 坑 |
| B 切片 layout | `.T.contiguous()` (row-major) | **`.T` (col-major view, 零拷贝)** ← 见 §4.2 第 3 坑 |

### 4.2 三个工程坑

#### 坑 1 — FP64 → FP8 直接 cvt 在 Triton 编译失败

```
RuntimeError: PassManager::run failed   (在 make_llir 阶段)
```

Triton 3.6 没有 FP64 → FP8 的直接 cvt 指令。**修复**：经 FP32 中转

```python
slice_v = (v * inv_2c[:, None]).to(tl.float32).to(tl.float8e4nv)
```

这一步无额外精度损失：slice 已归一化到 $[-1, 1]$，FP32 的 24-bit mantissa 远多于
FP8 的 4-bit。

#### 坑 2 — Triton 3.6 FP8 GEMM 在 sm_89 (Ada Lovelace) 上精度 bug

写了一个和 Stage 1 几乎一模一样的 FP8 GEMM kernel（仅输入 dtype 不同），编译通过、
能跑、生成的 PTX 也是正确的 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
指令（FP32 累加），**但实测结果**：

| $k$ | Triton-FP8 误差 vs FP64 ref | cuBLASLt-FP8 误差 |
|---|---|---|
| 64–256 | $0$ | $0$ |
| 512 | $3.5 \times 10^{-2}$ | $0$ |
| 1024 | $6.3 \times 10^{-1}$ | $0$ |
| 2048 | $5.0$ | $0$ |

误差量级 $10^{-2} \sim 10^{0}$，完全破坏 Ozaki Scheme 的"无舍入累加"前提。
试过 `max_num_imprecise_acc=0` flag，无效。猜测是 Triton 3.6 在 Ada Lovelace 上
的 codegen 缺陷（Hopper sm_90 上据报无此问题）。

**修复**：换用 `torch._scaled_mm`，它走 cuBLASLt 的 FP8 路径，**这也是论文 §4.1
明确使用的后端**（"Computations are performed with cuBLASLt FP8"）。
Triton kernel 留在源码里作反面教材（命名 `matmul_fp8_TRITON_kernel`，主流程不
调用）。

教学价值：算法骨架不变 — Triton 写 slicing + accumulation，GEMM 调最快的 FP8
后端。这正是论文实现的形态。

#### 坑 3 — `_scaled_mm` 要求 B 是 column-major

`torch._scaled_mm` 文档要求 B 张量满足 `stride(0) == 1`（列优先）。我们在
slicing 阶段拿到的 `Bt_slices[q]` 形状是 $[n, k]$ 的 row-major（strides $= (k, 1)$）。

**朴素做法**：`Bt_slices[q].T.contiguous().T` 拷贝转列优先 — 但 144 次内层循环
每次都拷贝一次，开销显著。

**省事做法**：注意 row-major $[n, k]$ 张量 `bs` 的内存布局，**用 `bs.T` 视图
重新解释成 $[k, n]$ col-major 即可（strides 变成 $(1, k)$）**，零拷贝。
代码：

```python
B_slices = [bs.T for bs in Bt_slices]   # .T 是视图, 不是拷贝
```

实测优化后 FP8 路径耗时从 ~40ms 降到 ~20ms (m=k=n=2048)。

---

> **注意**：`torch.float8_e4m3fn` 张量支持 `.T` 视图，但 `.amax()`、`.abs()`
> 等部分 reduction op 不支持。我们的切片 reduction 都在 FP64 上做（用残量），
> 对 FP8 张量只做 cvt + store + load + matmul，所以没踩到这个雷。

---

## 5. FP8 路径的精度预期

每片有效位 ~4，所以两片乘积有 ~8 位 mantissa；累加 $k$ 个有 $8 + \log_2 k$ 位。

- $k = 128$：$8 + 7 = 15$ bits，远 < FP32 的 24 bits。**无舍入**。
- $k = 2048$：$8 + 11 = 19$ bits。**仍无舍入**。
- $k = 16384$：$8 + 14 = 22$ bits。**接近边界**。
- $k = 65536$：$8 + 16 = 24$ bits。**恰好填满 FP32**，此后开始溢出（这就是 Table 3 中 FP8/FP32 列在 $k > 65536$ 时打 "—"的原因）。

所以本 stage 在 $k \le 16384$ 应该和 Stage 1 一样达到 $\sim 10^{-15}$ 相对误差。

---

## 6. 实测性能（vs Stage 1 / cuBLAS DGEMM）

| 尺寸 m=k=n | 切片数 | FP16 (Stage 1) | FP8 (Stage 2) | cuBLAS DGEMM |
|---|---|---|---|---|
| 128 | 12×12 = 144 | 5 ms | 19 ms | 0.2 ms |
| 256 | 12×12 = 144 | 5 ms | 12 ms | 0.1 ms |
| 512 | 12×12 = 144 | 6 ms | 12 ms | 0.3 ms |
| 1024 | 12×12 = 144 | 7 ms | 11 ms | 2.0 ms |
| 2048 | 12×12 = 144 | 18 ms | 21 ms | 16 ms |

**观察**：

- 切片数恒为 12×12 = 144（$\rho = 49 = \xi$ 主导，对 $k$ 不敏感），比 Stage 1 的 $7\!-\!9 \times 7\!-\!9$ 多 2× 左右
- FP8 路径目前**没有跑赢 FP16**。原因 §3 已分析（GEMM 总耗时 ≈ 1.5× FP16）
- 大尺寸下 FP8 路径与 cuBLAS DGEMM 持平 (~1.3×)，证明算法在硬件上是 viable 的

## 7. 工程注意点

- **16 字节对齐**：cuBLASLt 的 FP8 路径要求 $m, n, k$ 是 16 的倍数。本 stage 测试集 $m=n=k \in \{128, 256, 512, 1024, 2048\}$ 都满足
- **DRAM 带宽减半**：FP8 切片是 FP16 的一半大小。`slice_iter_kernel` 写带宽减半；累加阶段不变（输入仍是 FP32）
- **`torch.float8_e4m3fn` 张量限制**：少量 PyTorch op 不支持（如 `.abs()`、`.amax()`、按位运算）。本 stage 的所有 reduction 都在 FP64 残量上做，没踩雷

---

阅读完毕，进 `ozaki_fp8.py`。
