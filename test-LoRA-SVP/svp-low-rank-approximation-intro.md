# SVD 与低秩近似（Low-Rank Approximation）简介

本文说明：在已经通过矩阵相减得到差异矩阵 $\Delta W$ 之后，如何用 **SVD（奇异值分解）** 做最佳低秩近似，再把结果整理成 LoRA 所需的两个 skinny 矩阵 $A$、$B$。

> 约定：下文中的 “SVP / SVD pipeline” 指这条 **差分权重 → 截断 SVD → LoRA 因子** 的流程；数学对象是标准的 **SVD**。

假设你已经得到：

$$
\Delta W \in \mathbb{R}^{d \times k}
$$

---

## 1. SVD 的数学原理与低秩近似

### 第一步：标准的 SVD 分解

根据线性代数定理，任何矩阵 $\Delta W$ 都可以被完美、无损地分解为三个矩阵的乘积：

$$
\Delta W = U \cdot \Sigma \cdot V^{T}
$$

其中：

- $U \in \mathbb{R}^{d \times d}$：左奇异矩阵（Left Singular Vectors）。其列向量是 $\Delta W \Delta W^{T}$ 的特征向量（Eigenvectors）。
- $\Sigma \in \mathbb{R}^{d \times k}$：对角矩阵（Singular Values）。对角线上的元素 $\sigma_{i}$ 是奇异值（即 $\Delta W \Delta W^{T}$ 特征值的平方根），并且从大到小排列：

$$
\sigma_1 \ge \sigma_2 \ge \dots \ge 0
$$

- $V^{T} \in \mathbb{R}^{k \times k}$：右奇异矩阵（Right Singular Vectors）。其行向量是 $\Delta W^{T} \Delta W$ 的特征向量。

### 第二步：截断 SVD（Truncated SVD）提取 Skinny 矩阵

目标是把 $\Delta W$ 转化为 rank 为 $r$ 的 LoRA 形式。为此只保留 $\Sigma$ 中前 $r$ 个最大的奇异值，以及 $U$ 与 $V^{T}$ 中对应的列和行。

通过这种“截断”，就得到对 $\Delta W$ 的**最佳低秩近似**（Eckart–Young–Mirsky Theorem）：

$$
\Delta W \approx U_{r} \cdot \Sigma_{r} \cdot V_{r}^{T}
$$

截断后各矩阵维度变为：

- $U_r \in \mathbb{R}^{d \times r}$（瘦长矩阵）
- $\Sigma_r \in \mathbb{R}^{r \times r}$（小对角矩阵）
- $V_r^{T} \in \mathbb{R}^{r \times k}$（矮扁矩阵）

---

## 2. 转换成 LoRA 格式（Matrix A & Matrix B）

标准 LoRA 结构只需要两个矩阵：

$$
\Delta W = B \times A
$$

要把 SVD 的三个矩阵合并成两个，有两种常见分配方案。

### 方案一：对称分配（最常用）

把对角矩阵 $\Sigma_{r}$ 的平方根均匀分给左边和右边：

- Matrix $B = U_r \cdot \sqrt{\Sigma_r}$，维度为 $d \times r$  
  （LoRA 的后半部分，负责升维）
- Matrix $A = \sqrt{\Sigma_r} \cdot V_r^{T}$，维度为 $r \times k$  
  （LoRA 的前半部分，负责降维）

于是：

$$
B A = U_r \sqrt{\Sigma_r} \cdot \sqrt{\Sigma_r} V_r^{T} = U_r \Sigma_r V_r^{T} \approx \Delta W
$$

### 方案二：吸收给单侧

把奇异值完全乘进某一边（例如乘给 $V_{r}^{T}$）：

- Matrix $B = U_{r}$
- Matrix $A = \Sigma_r \cdot V_r^{T}$

同样有 $B A = U_r \Sigma_r V_r^{T} \approx \Delta W$。

---

## 3. Serving 时的 Scaling Factor 注意点

在 Serving 框架（如 vLLM、LoRAX）中加载时，LoRA 通常还会再乘一个缩放因子：

$$
\frac{\alpha}{r}
$$

因此，在你通过 SVD 算好 $A$ 和 $B$ 之后：

- 若框架会**自动**再乘 $\frac{\alpha}{r}$，你需要反向给矩阵除以该系数；**或者**
- 直接在框架配置里把 $\alpha$ 设为与 $r$ 相等，即 $\frac{\alpha}{r} = 1$，这样就完全不需要额外的数学缩放。

---

## 4. 和本仓库脚本的关系（避免混淆）

| 做法 | 含义 |
|---|---|
| **本文描述的 SVP/SVD 路径** | 先有 $\Delta W$（例如两个微调模型相减），再 **纯计算** 截断 SVD → 得到 $A,B$ |
| **PEFT 训练 LoRA** | 不先算 $\Delta W$；用梯度下降直接学习 $A,B$ |

两者最终都能得到 $d \times r$ 与 $r \times k$ 的 skinny factors，但来源不同：一个是 **分解算出来的**，一个是 **优化学出来的**。
