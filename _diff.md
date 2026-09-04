========== CELL 0 (markdown) ==========
---------- EN ----------
# Triton Puzzles

Programming for accelerators such as GPUs is critical for modern AI systems.
This often means programming directly in proprietary low-level languages such as CUDA. [Triton](https://github.com/openai/triton/) is an alternative open-source language that allows you to code at a higher-level and compile to accelerators like GPU.

[IMG]

Coding for Triton is very similar to Numpy and PyTorch in both syntax and semantics. However, as a lower-level language there are a lot of details that you need to keep track of. In particular, one area that learners have trouble with is memory loading and storage which is critical for speed on low-level devices.

This set is puzzles is meant to teach you how to use Triton from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks. These puzzles **do not** need to run on GPU since they use a Triton interpreter.

---------- ZH ----------
# Triton 谜题

在为 GPU 等加速器编程时，性能对现代 AI 系统至关重要。
过去我们通常直接使用 CUDA 之类的低层次专有语言来写。[Triton](https://github.com/openai/triton/) 是一个开源的替代方案，它让你能以更高层次的语法编写代码，并编译到 GPU 等加速器上运行。

[IMG]

Triton 的语法和语义与 Numpy、PyTorch 非常相似。但作为一门更贴近硬件的语言，它有很多细节需要你亲自把控。学习者通常最容易卡住的点，就是内存的加载与存储 —— 而这恰恰是决定性能的关键。

这套 puzzle 的目的是以互动的方式，从最基本的原理开始教你使用 Triton。你会从最简单的例子出发，一步步实现 Flash Attention、量化神经网络等真实算法。这些 puzzle **不需要** 真的在 GPU 上运行，它们使用的是 Triton 解释器。


========== CELL 4 (markdown) ==========
---------- EN ----------
## Introduction

To begin with, we will only use `tl.load` and `tl.store` in order to build simple programs.

Here's an example of load. It takes an `arange` over the memory. By default the indexing of torch tensors with column, rows, depths or right-to-left. It also takes in a mask as the second argument. Mask is critically important because all shapes in Triton need to be powers of two.

**Note:** If you're using triton-viz, make sure to click on the link that says `Cloudflare tunnel URL` to see the visualizer.
[IMG]
Triton-viz should look like this. Each cell represents a value in the input tensor, and the blue cells are the elements loaded by the `tl.load` call:
[IMG]

---------- ZH ----------
## 入门介绍

作为起步，我们只会使用 `tl.load` 和 `tl.store` 这两个函数来构造简单的程序。

下面是一个 load 的示例。它在内存上做了一个 `arange`。PyTorch 张量默认按 列-行-深度 的顺序、从右到左建立索引。第二个参数是一个 mask（掩码）。mask 至关重要，因为在 Triton 中所有形状都必须是 2 的幂。

**注意：** 如果你在使用 triton-viz，别忘了点击提示中的 `Cloudflare tunnel URL` 链接来查看可视化界面。
[IMG]
triton-viz 大致长这样。每个格子代表输入张量中的一个元素，蓝色格子表示被 `tl.load` 读取的元素：
[IMG]


========== CELL 6 (markdown) ==========
---------- EN ----------
You can also use this trick to read in a 2d array.

---------- ZH ----------
同样的技巧也可以用来读入一个 2D 数组。


========== CELL 8 (markdown) ==========
---------- EN ----------
The `tl.store` function is quite similar. It allows you to write to a tensor.
In triton-viz, cells that are written to the tensor are colored orange. It should look like this:
[IMG]

---------- ZH ----------
`tl.store` 函数用法非常类似，它让你把数据写回张量。

在 triton-viz 里，被写入的格子会被染成橙色，效果大致如下：

[IMG]


========== CELL 10 (markdown) ==========
---------- EN ----------
You can only load in relatively small `blocks` at a time in Triton. to work with larger tensors you need to use a program id axis to run multiple blocks in parallel. Here is an example with one program axis with 3 blocks. You can use the visualizer to scroll over it.

---------- ZH ----------
在 Triton 中，你一次只能加载相对较小的 `block`（数据块）。要处理更大的张量，就需要沿着某个 program id 轴（并行程序编号）来同时启动多个 block。下面是一个例子：使用一个程序轴、共启动 3 个 block。你可以用可视化工具滚动查看它们。


========== CELL 12 (markdown) ==========
---------- EN ----------
See the [Triton Docs](https://triton-lang.org/main/index.html) for further information.

---------- ZH ----------
更多细节请查阅 [Triton 官方文档](https://triton-lang.org/main/index.html)。


========== CELL 13 (markdown) ==========
---------- EN ----------
## Puzzle 1: Constant Add

Add a constant to a vector. Uses one program id axis. Block size `B0` is always the same as vector `x` with length `N0`.

$$z_i = 10 + x_i \text{ for } i = 1\ldots N_0$$

---------- ZH ----------
## 谜题 1：常数加法（Constant Add）

给一个向量的每个元素都加上一个常数。只使用一个 program id 轴。block 大小 `B0` 恒等于向量 `x` 的长度 `N0`。

$$z_i = 10 + x_i \text{  当  } i = 1\ldots N_0$$


========== CELL 14 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 16 (markdown) ==========
---------- EN ----------
## Puzzle 2: Constant Add Block

Add a constant to a vector. Uses one program block axis (no `for` loops yet). Block size `B0` is smaller than the vector length `N0`, so one axis is used to tile across the input.

$$z_i = 10 + x_i \text{ for } i = 1\ldots N_0$$

---------- ZH ----------
## 谜题 2：分块常数加法（Constant Add Block）

给向量的每个元素加上一个常数。这次只使用一个 program 块轴（先不用 `for` 循环）。block 大小 `B0` 小于向量长度 `N0`，因此需要用这一个轴来对输入做分块。

$$z_i = 10 + x_i \text{  当  } i = 1\ldots N_0$$


========== CELL 17 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 19 (markdown) ==========
---------- EN ----------
## Puzzle 3: Outer Vector Add

Add two vectors.

Uses one program block axis. Block size `B0` is always the same as vector `x` length `N0`.
Block size `B1` is always the same as vector `y` length `N1`.

$$z_{j, i} = x_i + y_j\text{ for } i = 1\ldots B_0,\ j = 1\ldots B_1$$

---------- ZH ----------
## 谜题 3：外积加法（Outer Vector Add）

把两个向量相加，形成一个 2D 矩阵。

只使用一个 program 块轴。block 大小 `B0` 恒等于向量 `x` 的长度 `N0`，`B1` 恒等于向量 `y` 的长度 `N1`。

$$z_{j, i} = x_i + y_j\text{  当  } i = 1\ldots B_0,\ j = 1\ldots B_1$$


========== CELL 20 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 22 (markdown) ==========
---------- EN ----------
## Puzzle 4: Outer Vector Add Block

Add a row vector to a column vector.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

$$z_{j, i} = x_i + y_j\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$

---------- ZH ----------
## 谜题 4：分块外积加法（Outer Vector Add Block）

把一个行向量和一个列向量相加。

使用两个 program 块轴。block 大小 `B0` 小于向量 `x` 的长度 `N0`；block 大小 `B1` 小于向量 `y` 的长度 `N1`。

$$z_{j, i} = x_i + y_j\text{  当  } i = 1\ldots N_0,\ j = 1\ldots N_1$$


========== CELL 23 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 25 (markdown) ==========
---------- EN ----------
## Puzzle 5: Fused Outer Multiplication

Multiply a row vector to a column vector and take a relu.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

$$z_{j, i} = \text{relu}(x_i \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$

---------- ZH ----------
## 谜题 5：融合外积乘法（Fused Outer Multiplication）

把一个行向量和一个列向量做外积并加上 relu。

使用两个 program 块轴。block 大小 `B0` 小于向量 `x` 的长度 `N0`；block 大小 `B1` 小于向量 `y` 的长度 `N1`。

$$z_{j, i} = \text{relu}(x_i \times y_j)\text{  当  } i = 1\ldots N_0,\ j = 1\ldots N_1$$


========== CELL 26 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 28 (markdown) ==========
---------- EN ----------
## Puzzle 6: Fused Outer Multiplication - Backwards

Backwards of a function that multiplies a matrix with a row vector and take a relu.

Uses two program blocks. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`. Chain rule backward `dz`
is of shape `N1` by `N0`

$$f(x, y) = \text{relu}(x_i \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$

$$dx_{i, j} = f_x'(x, y)_{i, j} \times dz_{i,j}$$

---------- ZH ----------
## 谜题 6：融合外积乘法 —— 反向传播（Fused Outer Multiplication - Backwards）

对一个「矩阵乘以行向量再取 relu」的函数做反向传播。

使用两个 program 块。block 大小 `B0` 小于向量 `x` 的长度 `N0`；block 大小 `B1` 小于向量 `y` 的长度 `N1`。链式法则中的上游梯度 `dz` 形状为 `N1` × `N0`。

$$f(x, y) = \text{relu}(x_i \times y_j)\text{  当  } i = 1\ldots N_0,\ j = 1\ldots N_1$$

$$dx_{i, j} = f_x'(x, y)_{i, j} \times dz_{i,j}$$


========== CELL 29 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 31 (markdown) ==========
---------- EN ----------
## Puzzle 7: Long Sum

Sum of a batch of numbers.

Uses one program blocks. Block size `B0` represents a range of batches of `x` of length `N0`.
Each element is of length `T`. Process it `B1 < T` elements at a time.

$$z_{i} = \sum^{T}_j x_{i,j} =  \text{ for } i = 1\ldots N_0$$

Hint: You will need a for loop for this problem. These work and look the same as in Python.

---------- ZH ----------
## 谜题 7：长向量求和（Long Sum）

对一个 batch 的数字求和。

使用一个 program 块。block 大小 `B0` 表示对 `x` 的一段 batch 范围，`x` 的总长为 `N0`。每个元素本身长度为 `T`，我们每次处理 `B1 < T` 个元素。

$$z_{i} = \sum^{T}_j x_{i,j} \text{  当  } i = 1\ldots N_0$$

提示：这个题需要用到 for 循环。写法和语义都与普通 Python 一致。


========== CELL 32 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 34 (markdown) ==========
---------- EN ----------
## Puzzle 8: Long Softmax

Softmax of a batch of logits.

Uses one program block axis. Block size `B0` represents the batch of `x` of length `N0`.
Block logit length `T`. Process it `B1 < T` elements at a time.

$$z_{i, j} = \text{softmax}(x_{i,1} \ldots x_{i, T}) \text{ for } i = 1\ldots N_0$$

Note softmax needs to be computed in numerically stable form as in Python. In addition in Triton they recommend not using `exp` but instead using `exp2`. You need the identity

$$\exp(x) = 2^{\log_2(e) x}$$

Advanced: there one way to do this with 3 loops. You can also do it with 2 loops if you are clever. Hint: you will find this identity useful:

$$\exp(x_i - m) =  \exp(x_i - m/2 - m/2) = \exp(x_i - m/ 2) /  \exp(m/2) $$

---------- ZH ----------
## 谜题 8：长向量 Softmax（Long Softmax）

对一个 batch 的 logits 计算 softmax。

使用一个 program 块轴。block 大小 `B0` 表示 `x` 的一段 batch，`x` 的总长为 `N0`；每一行 logit 长度为 `T`，每次处理 `B1 < T` 个元素。

$$z_{i, j} = \text{softmax}(x_{i,1} \ldots x_{i, T}) \text{  当  } i = 1\ldots N_0$$

注意 softmax 需要使用数值稳定的形式（先减最大值），这在 Python 里的实现也是一样。此外在 Triton 中，官方推荐不要用 `exp`，而是用 `exp2`。这里会用到一个恒等式：

$$\exp(x) = 2^{\log_2(e) \, x}$$

进阶：常规写法需要 3 次循环。如果你够机智，用 2 次循环也可以做到。提示：下面这个恒等式会很有用：

$$\exp(x_i - m) = \exp(x_i - m/2 - m/2) = \exp(x_i - m/2) / \exp(m/2)$$


========== CELL 35 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 37 (markdown) ==========
---------- EN ----------
## Puzzle 9: Simple FlashAttention

A scalar version of FlashAttention.

Uses zero programs. Block size `B0` represents `k` of length `N0`.
Block size `B0` represents `q` of length `N0`. Block size `B0` represents `v` of length `N0`.
Sequence length is `T`. Process it `B1 < T` elements at a time.

$$z_{i} = \sum_{j} \text{softmax}(q_1 k_1, \ldots, q_T k_T)_j v_{j} \text{ for } i = 1\ldots N_0$$

This can be done in 1 loop using a similar trick from the last puzzle.

---------- ZH ----------
## 谜题 9：简化版 FlashAttention（Simple FlashAttention）

一个标量版本的 FlashAttention。

不使用任何 program 轴。block 大小 `B0` 表示 `k` 的长度 `N0`，同时也表示 `q` 和 `v` 的长度 `N0`。序列长度为 `T`，每次处理 `B1 < T` 个元素。

$$z_{i} = \sum_{j} \text{softmax}(q_1 k_1, \ldots, q_T k_T)_j \cdot v_{j} \text{  当  } i = 1\ldots N_0$$

利用上一题里的技巧，这个题可以只用 1 层循环就完成。


========== CELL 38 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 40 (markdown) ==========
---------- EN ----------
## Puzzle 10: Two Dimensional Convolution

A batched 2D convolution.

Uses one program id axis. Block size `B0` represent the batches to process out of `N0`.
Image `x` is size is `H` by `W` with only 1 channel, and kernel `k` is size `KH` by `KW`.

$$z_{i, j, k} = \sum_{oj, ok} k_{oj,ok} \times x_{i,j + oj, k + ok} \text{ for } i = 1\ldots N_0$$

---------- ZH ----------
## 谜题 10：二维卷积（Two Dimensional Convolution）

批量二维卷积。

使用一个 program id 轴。block 大小 `B0` 表示要处理的 batch，`N0` 为总 batch 数。图像 `x` 的空间大小为 `H` × `W`，通道数为 1；卷积核 `k` 的大小为 `KH` × `KW`。

$$z_{i, j, k} = \sum_{oj, ok} k_{oj,ok} \times x_{i,j + oj,\, k + ok} \text{  当  } i = 1\ldots N_0$$


========== CELL 41 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 43 (markdown) ==========
---------- EN ----------
## Puzzle 11: Matrix Multiplication

A blocked matrix multiplication.

Uses three program id axes. Block size `B2` represent the batches to process out of `N2`.
Block size `B0` represent the rows of `x` to process out of `N0`. Block size `B1` represent the cols of `y` to process out of `N1`. The middle shape is `MID`.

$$z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$$

You are allowed to use `tl.dot` which computes a smaller mat mul.

Hint: the main trick is that you can split a matmul into smaller parts.

$$z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} +  \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k} $$

---------- ZH ----------
## 谜题 11：矩阵乘法（Matrix Multiplication）

分块矩阵乘法。

使用三个 program id 轴。block 大小 `B2` 表示 batch 维度上的处理块（总数为 `N2`）；`B0` 表示 `x` 的行方向上的处理块（总行数为 `N0`）；`B1` 表示 `y` 的列方向上的处理块（总列数为 `N1`）。中间维度为 `MID`。

$$z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k} \text{  当  } i = 1\ldots N_2,\ j = 1\ldots N_0,\ k = 1\ldots N_1$$

你可以使用 `tl.dot` 来计算子块的矩阵乘法。

提示：核心技巧是把一次大矩阵乘法拆成两次（或多次）小的求和：

$$z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} + \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k}$$


========== CELL 44 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


========== CELL 46 (markdown) ==========
---------- EN ----------
## Puzzle 12: Quantized Matrix Mult

When doing matrix multiplication with quantized neural networks a common strategy is to store the weight matrix in lower precision, with a shift and scale term.

For this problem our `weight` will be stored in 4 bits. We can store `FPINT` of these in a 32 bit integer. In addition for every `group` weights in order we will store 1 `scale` float value and 1 `shift` 4 bit value. We store these for the column of weight. The `activation`s are stored separately in standard floats.

Mathematically it looks like.

$$z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$$

However, it is a bit more complex since we need to also extract the 4-bit values into floats to begin.

---------- ZH ----------
## 谜题 12：量化矩阵乘法（Quantized Matrix Mult）

在量化神经网络的矩阵乘法中，一个常见做法是把权重矩阵以更低精度存储，并额外保存 shift（偏移）与 scale（缩放）两个量。

这个题里，我们把 `weight` 用 4 bit 存储。一个 32 bit 整数可以打包 `FPINT` 个这样的 4 bit 权重。此外，每 `group` 个连续权重会共享一个 `scale`（float）和一个 `shift`（4 bit）。这些统计量按列存储。`activation`（激活值）则用普通的 float 存储。

数学形式上：

$$z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} \, (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k} \text{  当  } i = 1\ldots N_2,\ j = 1\ldots N_0,\ k = 1\ldots N_1$$

不过实际实现会稍微复杂一些 —— 我们首先要把那些 4 bit 的值解包成 float。


========== CELL 47 (markdown) ==========
---------- EN ----------
[IMG]

---------- ZH ----------
[IMG]


