# ozaki-gpu — Triton 实现 Ozaki Scheme DGEMM 的教学项目

## 项目目标

通过真实可运行的 Triton + Python 代码，**从零讲解 Ozaki Scheme** —
即"用低精度 Tensor Core（FP16 / FP8）模拟出 FP64 GEMM 精度"的算法。

参考论文：`2508.00441v3.pdf`（Mukunoki, *DGEMM without FP64 Arithmetic*, 2025）。

定位：**教学优先 > 性能**。允许 naïve 实现，但必须算法上正确、注释充分。

## 学习路线（每个 stage 独立可运行；前一个跑通+答疑后再做下一个）

| Stage | 目录 | 内容 |
|---|---|---|
| 1 | `stage1_fp16/` | 标准 Ozaki + FP16 Tensor Core，验证算法骨架 |
| 2 | `stage2_fp8/` (待) | 切换到 FP8 (E4M3) 路径 |
| 3 | `stage3_kbk/` (待) | inner-product blocking |
| 4 | `stage4_emu/` (待) | uint64 模拟 FP64 算术（论文 §4.2） |

每个 stage 必含：
- `PRINCIPLE.md` — 算法原理讲解，公式用 MathJax 格式（`$...$` / `$$...$$`），方便复制到任何 Markdown 渲染器
- 主代码文件（含大量 WHY 注释）
- `verify.py` — 与 `torch.matmul`（cuBLAS DGEMM）对比精度

## 环境

- conda env: **`tridev`** — 用户偏好；**所有 Python / pip 操作必须在此环境内**，禁止系统 Python
- 硬件：RTX 4090 (sm_89, 支持 FP8 Tensor Core)
- Triton 3.6.0，PyTorch 2.11.0+cu126

## 运行方法

```bash
conda activate tridev
cd stage1_fp16
python verify.py
```

## 代码风格约定

- **大量注释解释 WHY**（不只是 WHAT）。变量命名遵循论文符号：`c`, `rho`, `sigma`, `s_A`, `s_B` …
- 公式注释直接用 MathJax，方便 IDE / Markdown 预览
- 每个 Triton kernel 顶部用 docstring 写算法语义 + 形状 + dtype 表
- 不追求性能优化，但若做了优化要标注"WHY"
- 优先单文件 `ozaki_fp{xx}.py` 容纳一个 stage 的核心逻辑（kernel + host），方便对照阅读

## 维护规则（给未来的 Claude）

- 每完成一个 stage，更新本文件"当前状态"小节
- 遇到 PR-level 改动（重构、加新 stage）才需要写新 markdown；小修不要凭空创建文档
- 所有公式优先 MathJax，不用 ASCII art 模拟数学
- 论文 `2508.00441v3.pdf` 在项目根目录，作为算法 ground truth

## 当前状态

- 🟢 **Stage 1（FP16 + FP32 accum）** — 已实现 + 验证通过
  - 三个 Triton kernel：`slice_iter_kernel`、`matmul_fp16_kernel`、`accumulate_kernel`
  - 主入口 `ozaki_dgemm(A, B)`；流式累加，内存占用 ≈ $s_A \cdot mk \cdot 2$ B（FP16 切片） + $mn \cdot 8$ B（FP64 输出）
  - 精度验证：`m=n=k ∈ {128, 256, 512, 1024, 2048}`，`max_rel_err` 全部 $\le 7 \times 10^{-15}$
  - 性能：xlarge (2048³) 与 cuBLAS DGEMM 持平（1.1× 慢）

- 🟢 **Stage 2（FP8 E4M3 + FP32 accum）** — 已实现 + 验证通过
  - 算法骨架与 Stage 1 一致；切片 dtype 改 FP8，GEMM 后端改 cuBLASLt
  - **GEMM 后端选择故事**：写 Triton FP8 kernel 时发现 Triton 3.6 在 sm_89 (Ada Lovelace)
    上 FP8 GEMM 有 codegen 缺陷（k≥512 时累加精度爆炸，详见 stage2_fp8/PRINCIPLE.md §4.2）。
    主路径改用 `torch._scaled_mm` (cuBLASLt FP8)，正是论文 §4.1 的实际后端。
    Triton FP8 kernel 留作反面教材
  - 关键实现细节：FP64→FP8 cvt 须经 FP32 中转；B 切片用 `.T` 视图（零拷贝）转 col-major 喂给 `_scaled_mm`
  - 精度：与 Stage 1 持平，`max_rel_err ≤ 7e-15` 跨同样测试集
  - 性能：xlarge 21 ms（FP16 是 18 ms，cuBLAS DGEMM 16 ms），FP8 慢一点的根因是
    切片数 12×12=144 比 Stage 1 的 9×9=81 多得多 — 我们的 $\rho$ 公式较保守
- ⏳ 等待用户 Stage 2 答疑 → 决定是否进入 Stage 3 (inner-product blocking)
