# Triton 谜题

作者：[Tejas Ramesh](https://tejas3070.github.io/)、[Keren Zhou](https://www.jokeren.tech/)，基于 [Triton-Viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) 构建

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb)


在为 GPU 等加速器编程时，性能对现代 AI 系统至关重要。
过去我们通常直接使用 CUDA 之类的低层次专有语言来写。[Triton](https://github.com/openai/triton/) 是一个开源的替代方案，它让你能以更高层次的语法编写代码，并编译到 GPU 等加速器上运行。

Triton 的语法和语义与 Numpy、PyTorch 非常相似。但作为一门更贴近硬件的语言，它有很多细节需要你亲自把控。学习者通常最容易卡住的点，就是内存的加载与存储 —— 而这恰恰是决定性能的关键。

这套 puzzle 的目的是以互动的方式，从最基本的原理开始教你使用 Triton。你会从最简单的例子出发，一步步实现 Flash Attention、量化神经网络等真实算法。这些 puzzle **不需要** 真的在 GPU 上运行，它们使用的是 Triton 解释器。

Discord: https://discord.gg/gpumode #triton-puzzles

<img width="2397" height="1195" alt="image" src="https://github.com/user-attachments/assets/a7e0219b-9df7-4640-a73e-35c1fa50b0f2" />



如果你喜欢这类风格，这已经是这个系列的第 7 套 puzzle 了：

* https://github.com/srush/gpu-puzzles
* https://github.com/srush/tensor-puzzles
* https://github.com/srush/autodiff-puzzles
* https://github.com/srush/transformer-puzzles
* https://github.com/srush/GPTworld
* https://github.com/srush/LLM-Training-Puzzles
