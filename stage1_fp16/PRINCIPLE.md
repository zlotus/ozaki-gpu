# Stage 1 — Ozaki Scheme + FP16 Tensor Core 模拟 DGEMM

> 本文档把 Ozaki Scheme 的数学骨架展开讲到"读完就能看懂代码"的程度。
> 阅读顺序建议：本文 → `ozaki_fp16.py`（边读边对照本文章节号） → `verify.py` 跑一次。

---

## 0. 一句话概括

把 FP64 矩阵 $A$、$B$ **按位拆成若干 FP16 切片**，每对切片用 FP16 Tensor Core 算一个 GEMM
（FP32 累加，**保证无舍入误差**），最后把所有切片乘积 **按 2 的整数次幂加权**累加回一个 FP64 矩阵 $C$。

整个过程 **不调用任何 FP64 GEMM**，但能复现 FP64 GEMM 的精度。

---

## 1. 为什么需要这个东西

| 硬件指标 | RTX 4090 (sm_89) |
|---|---|
| FP64 (CUDA core) | ≈ 1.3 TFlops/s |
| FP16 Tensor Core (FP32 accum) | ≈ 165 TFlops/s |
| FP8 Tensor Core (FP32 accum) | ≈ 660 TFlops/s |

FP64 比 FP16 慢两个数量级。AI 卡（GB200/GB300、AMD MI350）甚至砍掉或大幅削弱 FP64。
Ozaki Scheme 提供了一座桥：**用低精度算力 + 一点 FP64 标量算术，得到 FP64 GEMM 的结果**。

---

## 2. 数学骨架

### 2.1 误差自由的内积分解

设 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^{k}$ 是两个 FP64 向量。我们想要它们的内积 $\mathbf{x}^{T}\mathbf{y}$。

Ozaki Scheme 的核心恒等式是把每个向量分解成

$$
\mathbf{x} \;=\; \sum_{p=1}^{s_x} 2^{c_x^{(p)}} \, \mathbf{x}^{(p)},
\qquad
\mathbf{y} \;=\; \sum_{q=1}^{s_y} 2^{c_y^{(q)}} \, \mathbf{y}^{(q)}
$$

其中

- $\mathbf{x}^{(p)}, \mathbf{y}^{(q)}$ 是 **低精度切片**（本 stage 是 FP16），数值范围被归一化到 $[-1, 1]$
- $c_x^{(p)}, c_y^{(q)}$ 是 **整数指数**（FP64 标量）

代入内积：

$$
\boxed{\;\mathbf{x}^{T}\mathbf{y} \;=\; \sum_{p=1}^{s_x}\sum_{q=1}^{s_y} 2^{\,c_x^{(p)}+c_y^{(q)}} \; \underbrace{\mathbf{x}^{(p)\,T} \mathbf{y}^{(q)}}_{\text{低精度 GEMM 计算}}\;}
$$

**关键事实（论文 Eq. 11）**：只要每个切片保留的位数足够少，每个内积 $\mathbf{x}^{(p)\,T}\mathbf{y}^{(q)}$ 在 **Type3（FP32）累加器**里能 **精确计算（无舍入误差）**。所以最终的 FP64 精度只取决于外层的 $\sum_{p,q}$ 累加（用 FP64 完成）。

### 2.2 推广到矩阵

对 $A \in \mathbb{R}^{m \times k}$ 和 $B \in \mathbb{R}^{k \times n}$：

- $A$ 沿 **行方向** 切片（每行是一个长度为 $k$ 的向量）：
  得到 $\{A^{(p)}\}_{p=1}^{s_A}$，每个 shape $[m, k]$ FP16；
  指数矩阵 $c_A \in \mathbb{R}^{s_A \times m}$（每行一个指数）
- $B$ 沿 **列方向** 切片（每列是一个长度为 $k$ 的向量）：
  得到 $\{B^{(q)}\}_{q=1}^{s_B}$，每个 shape $[k, n]$ FP16；
  指数矩阵 $c_B \in \mathbb{R}^{s_B \times n}$（每列一个指数）

然后

$$
G^{(p,q)} \;=\; A^{(p)} B^{(q)} \in \mathbb{R}^{m \times n} \text{ in FP32}
$$

最终输出：

$$
C[i,j] \;=\; \sum_{p,q} 2^{\,c_A^{(p)}[i] + c_B^{(q)}[j]} \cdot G^{(p,q)}[i,j]
$$

---

## 3. 切片算法（论文 Eq. 4–8）

这是 **递归** 算法。从 $\mathbf{x}^{(1)} = \mathbf{x}$ 开始，第 $p$ 轮：

$$
\begin{aligned}
c_x^{(p)} &\;:=\; \left\lceil \log_2 \max_{1 \le i \le k}\big|x^{(p)}_i\big| \right\rceil
&&\text{(本轮残量的 "幅值")} \\
\sigma &\;:=\; 0.75 \cdot 2^{\,\rho + c_x^{(p)}}
&&\text{(舍入常数, } \rho \text{ 见 §4)} \\
v_i &\;:=\; \mathrm{fl}_{\text{Type1}}\!\big(x^{(p)}_i + \sigma\big) - \sigma
&&\text{(取 } x^{(p)}_i \text{ 的高位)} \\
x^{(p+1)}_i &\;:=\; x^{(p)}_i - v_i
&&\text{(残量传给下一轮)} \\
x^{(p)\,\text{slice}}_i &\;:=\; \mathrm{cvt}_{\text{Type2}}\!\big(2^{-c_x^{(p)}} v_i\big)
&&\text{(归一化到 } [-1,1] \text{ 后降到 FP16)}
\end{aligned}
$$

直到 $\mathbf{x}^{(p+1)} = \mathbf{0}$ 退出循环。

### 3.1 为什么 $(x + \sigma) - \sigma$ 能 "取出高位"？

这是 **Dekker / Veltkamp split** 的同款技巧。

设 $x$ 是 FP64，$\sigma$ 选成 $\sigma \gg |x|$，并且 $\sigma$ 的指数大约比 $|x|$ 大 $\rho$ 位。
浮点加法 $x + \sigma$ 必须 **舍入到 FP64 mantissa**，那就把 $x$ 中权重 $< \mathrm{ulp}(\sigma)$ 的低位
整体丢掉（按 round-to-nearest-even）。再减去 $\sigma$，剩下的就是 $x$ 的 **高 $\sim m_1 - \rho$ 位**。

**直觉图（mantissa 位）**：
```
x:    [b52 b51 ... b8 b7 ... b1 b0]    (53 位)
σ:    [指数足够大, 使得 ulp(σ) 落在 b8 的位置]
x+σ:  [σ 部分][x 的 b52..b8, 四舍五入]   ← b7..b0 被丢掉
v:    [   0   ][x 的 b52..b8           ]   ← 减回 σ 后剩下高位
x_next = x - v = [0, 0, ..., 0, b7, ..., b0]  ← 残量
```

**为什么用 0.75 而不是 1.0？** 出自 Minamihata et al. 2016。$0.75 \cdot 2^{e} = 2^{e-1} + 2^{e-2}$ 是个非平局的小数，可以避免 $x + \sigma$ 落在两个浮点数正中间时按"偶数偏好"舍入产生不可控行为。

### 3.2 为什么要把 $v_i$ 乘以 $2^{-c}$ 再降到 FP16？

切片要存到 FP16，但 FP16 的指数范围只有 $[-14, 15]$，能表示的最大值约 $6.5 \times 10^4$。
若 $|x| \sim 10^{10}$，直接 cvt 会溢出。所以先用 $2^{-c}$ 把切片归一化到 $[-1, 1]$ 区间，再降到 FP16，
然后把 $2^{c}$ 这个"指数"单独存在 $c_x^{(p)}$ 里，等累加阶段再乘回去。

**这就是 "exponent-mantissa 拆开存" 的本质**。

---

## 4. 切片参数 $\rho$ 的选择

$\rho$ 决定 **每个切片保留多少位 mantissa**：粗略地 $\text{bits per slice} \approx m_1 - \rho$。
$\rho$ 越大 → 每片越短 → 需要更多片 → 更多 GEMM。

两个约束（论文 Eq. 1–3）：

1. **切片必须能塞进 Type2（FP16）的 mantissa**：
   $$\text{bits per slice} \le m_2 \quad\Longrightarrow\quad \rho \;\ge\; m_1 - m_2 \;=:\; \xi$$

2. **GEMM 在 Type3 累加器（FP32）里不能溢出 mantissa**：
   每个切片乘积有 $\le 2(m_1 - \rho)$ 位；累加 $k$ 个有 $\le 2(m_1 - \rho) + \log_2 k$ 位；
   要塞进 $m_3$ 位：
   $$2(m_1 - \rho) + \log_2 k \;\le\; m_3 \quad\Longrightarrow\quad \rho \;\ge\; m_1 - \big\lfloor \tfrac{m_3 - \log_2 k}{2} \big\rfloor \;=:\; \gamma$$

最终：

$$\boxed{\;\rho \;=\; \max(\xi,\; \gamma)\;}$$

### 4.1 数值代入（FP64 → FP16, accum FP32）

$m_1 = 53,\; m_2 = 11,\; m_3 = 24,\; \xi = 42$。

| $k$ | $\log_2 k$ | $\gamma = 53 - \lfloor (24 - \log_2 k)/2 \rfloor$ | $\rho = \max(\xi, \gamma)$ | bits/slice $\approx 53 - \rho$ | 估算切片数 $s_x$ | GEMM 数 $s_x s_y$ |
|---|---|---|---|---|---|---|
| 128  | 7  | $53 - 8 = 45$ | 45 | 8 | 7  | 49  |
| 1024 | 10 | $53 - 7 = 46$ | 46 | 7 | 8  | 64  |
| 16384 | 14 | $53 - 5 = 48$ | 48 | 5 | 11 | 121 |

我们的估算偏保守，论文 Table 3 在同样条件下给出 25 / 49 / 81。差异来自每片可携带的 "重叠位"
和归一化常数（OzBLAS 的实际实现略复杂）。**对算法正确性而言， $\rho$ 偏大没影响，只是切片数偏多**。
本 stage 用上面的公式即可。

---

## 5. 三个 Triton kernel 的角色

### 5.1 `slice_iter_kernel` — 一轮切片
- **形状**：输入 $X$ FP64 $[m, k]$、$\sigma$ FP64 $[m]$、$2^{-c}$ FP64 $[m]$
- **输出**：本轮切片 FP16 $[m, k]$、残量 FP64 $[m, k]$
- **算子**：纯 element-wise + 行 broadcast；唯一的"奇技淫巧"是 $(x + \sigma) - \sigma$ 必须在 FP64 下做（不能被编译器优化掉！）

> **为什么 $\sigma$、$2^{-c}$ 在主机端预先算好？**
> 它们是按行的 reduction 结果（依赖 $\max_i |x_i|$），跨 K 维 block 的 reduction 在 Triton 里要么写两遍 kernel，要么用 atomic — 都不优雅。
> Stage 1 直接借 PyTorch 的 `amax(dim=1)` 算这两个标量，再喂给 Triton。教学价值更高。

### 5.2 `matmul_fp16_kernel` — FP16 → FP32 GEMM
- **形状**：$A$ FP16 $[M, K]$，$B$ FP16 $[K, N]$ → $C$ FP32 $[M, N]$
- **关键**：`tl.dot(a, b, acc=acc)` 用 FP32 累加器，确保论文 Eq. 11 成立
- **教学点**：这就是教科书版 Triton matmul（参考 Triton tutorial 03），给出 FP32 输出而不是 FP16，是 Ozaki 正确性的必要条件

> **能不能省去这个 kernel，直接用 `torch.matmul`？**
> 不能直接用 — `torch.matmul(a_fp16, b_fp16)` 默认返回 FP16 输出（虽然内部 cuBLAS 是 FP32 累加），
> 那一步 FP32 → FP16 的最后舍入会破坏 Eq. 11，引入 $\sim 2^{-11}$ 量级的相对误差。
> 我们要 FP32 输出，PyTorch 没有简便 API 直出，所以自己写 Triton 最干净。

### 5.3 `accumulate_kernel` — 加权累加
- **形状**：$C$ FP64 $[m, n]$（in-place 累加）、$G$ FP32 $[m, n]$、$2^{c_A^{(p)}}$ FP64 $[m]$、$2^{c_B^{(q)}}$ FP64 $[n]$
- **算子**：$C[i,j] \mathrel{+}= 2^{c_A[i]} \cdot 2^{c_B[j]} \cdot G[i,j]$
- **教学点**：演示 Triton 里如何 fuse 行/列广播 + FP32→FP64 提升 + in-place 累加

> **为什么传 $2^{c}$ 而不是 $c$？**
> 避免在 Triton 里调用 `tl.exp2`（对 FP64 不一定原生支持，且数值上重复算 $2^c$ 浪费）。
> 主机端一次性 `torch.exp2(c)` 算好，传 FP64 标量进 kernel，乘法即可。

---

## 6. 主流程伪代码

```python
def ozaki_dgemm(A, B):
    rho = compute_rho(k=A.shape[1])        # §4
    A_slices, c_A = slice_rows(A, rho)     # 切 A 的行
    Bt_slices, c_B = slice_rows(B.T, rho)  # 切 B 的列 == 切 B.T 的行
    B_slices = [bs.T for bs in Bt_slices]
    pow2_cA, pow2_cB = exp2(c_A), exp2(c_B)

    C = zeros(m, n, fp64)
    for p in range(s_A):
        for q in range(s_B):
            G_pq = matmul_fp16_to_fp32(A_slices[p], B_slices[q])   # kernel 2
            accumulate(C, G_pq, pow2_cA[p], pow2_cB[q])            # kernel 3
    return C
```

**复杂度**：$O(s_A \cdot s_B \cdot m n k)$ FP16 操作 + $O(s_A \cdot s_B \cdot m n)$ FP64 加法。

---

## 7. 期望精度

测试用 `torch.matmul(A, B)`（cuBLAS DGEMM）作为参考。
对均匀 $(1, 10)$ 随机的 FP64 矩阵，相对误差应该在 $10^{-15}$ 量级（接近机器精度）。

如果误差明显大于 $10^{-13}$，大概率是：
1. $\rho$ 太小，切片不够 → 切片数估算 `max_slices` 不够
2. GEMM 输出不慎被截到了 FP16
3. 累加里 `exp2(c_A + c_B)` 上溢（$c$ 极大时）

---

## 8. 局限 & 下一步

- **本 stage 不做** inner-product blocking（论文 §4.3）→ Stage 3
- **本 stage 不做** FP8 路径 → Stage 2
- **本 stage 不做** uint64 模拟 FP64 算术 → Stage 4
- **GEMM kernel 没调优**：BLOCK_M=BLOCK_N=128, BLOCK_K=32 是教学默认值，没用 autotune；性能大约 cuBLAS 的 30–50%

阅读完毕，进 `ozaki_fp16.py`。代码每一段的 docstring 会引用本文的章节号。
