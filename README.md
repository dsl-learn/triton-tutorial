<h3 align="center">
Hands-On Triton Tutorial 📖
</h3>

<h4 align="center">
Learn Triton: No GPU Experience Required
</h4>

<p align="center">
<a href="https://tt-tut.top"><b>🔗 tt-tut.top</b></a>
</p>

<p align="center">
<a href="README.en.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

本教程面向没有 GPU 经验的的Triton初学者，带你从基础的向量加到RoPE、matmul_ogs、topk、Gluon Attention 等大模型算子进阶学习之路。只写到了矩阵转置，建议使用和 LLM 对话或者在 Agent 环境中学。

> [!WARNING]
> 本项目已停止更新，并进入归档状态。

作者：[BobHuang](https://github.com/sBobHuang)，[知乎专栏](https://www.zhihu.com/column/c_1948447902964901167)，[Triton算子开发及编译器资源整理](https://zhuanlan.zhihu.com/p/2018815133590271874)

## Triton简介

[OpenAI/Triton](https://github.com/openai/triton) 是一个让你用 Python 写高性能 GPU 算子的编程语言(DSL)。目前有[NVIDIA](https://github.com/triton-lang/triton/tree/main/third_party/nvidia)、[AMD](https://github.com/triton-lang/triton/tree/main/third_party/amd)、[华为昇腾](https://github.com/Ascend/triton-ascend)、[寒武纪](https://github.com/FlagTree/flagtree/tree/main/third_party/cambricon)、[摩尔线程](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads)、[沐曦](https://github.com/FlagTree/flagtree/tree/main/third_party/metax)等多个后端，一个kernel**多种硬件**均可以运行，具体见[FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems)。

优势：保持 Python 的易用性，同时充分发挥 GPU 并行计算和内存优化能力。

应用：加速深度学习算子和自定义算子，提升大模型训练和推理性能。

本教程使用 Triton 3.4.0(released on 2025, Jul 31)，只需安装 torch==2.8.0。若使用较低版本的 PyTorch，可自行升级 Triton版本。Triton具有还不错的版本兼容，大部分算子对Triton版本**没有要求**。

# 目录

* 一、 Triton 简介
* 二、 [向量加算子实战](ex1-vector_add/blog.md)
  * 1、 [torch的向量加法](ex1-vector_add/blog.md#1torch的向量加法)
  * 2、 [单program 16个元素加法和验证](ex1-vector_add/blog.md#2单program-16个元素加法和验证)
  * 3、 [通过mask控制元素访问](ex1-vector_add/blog.md#3通过mask控制元素访问)
  * 4、 [多Block(program)运行](ex1-vector_add/blog.md#4多blockprogram运行)
  * 5、 [使用参数化的BLOCK_SIZE](ex1-vector_add/blog.md#5使用参数化的block_size)
* 三、 [练习1课后作业题解](ex1-vector_add/homework_solution.md)
  * 1、 [Matrix Copy 矩阵拷贝](ex1-vector_add/homework_solution.md#1matrix-copy)
  * 2、 [Color Inversion 颜色反转](ex1-vector_add/homework_solution.md#2color-inversion)
  * 3、 [Reverse Array 数组反转](ex1-vector_add/homework_solution.md#3reverse-array)
* 四、 [矩阵转置实践](ex2-matrix_transpose/blog.md)
  * 1、[1D grid 进行native转置](ex2-matrix_transpose/blog.md#11d-grid-进行native转置)
  * 2、[2D grid 进行native转置](ex2-matrix_transpose/blog.md#22d-grid-进行native转置)
  * 3、[tl.trans 原语](ex2-matrix_transpose/blog.md#3tltrans-原语)
  * 4、[维度扩展的1x1 block实践](ex2-matrix_transpose/blog.md#4维度扩展的1x1-block实践)
  * 5、[通过mask控制元素访问](ex2-matrix_transpose/blog.md#5通过mask控制元素访问)
  * 6、[性能对比](ex2-matrix_transpose/blog.md#6性能对比)
  * 7、[使用参数化的block_size](ex2-matrix_transpose/blog.md#7使用参数化的block_size)

* 附录
  * 附录1、[Triton kernel 原语覆盖情况](language_cover.md)
  * 附录2、kernel 优化机制覆盖情况(未覆盖)
  * 附录3、[关联项目及学习资料](other_repo.md)
