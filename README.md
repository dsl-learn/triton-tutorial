<h3 align="center">
Hands-On Triton Tutorial 🚀
</h3>

<h4 align="center">
Learn Triton with Basic Python Knowledge
</h4>

<p align="center">
<a href="https://tt-tut.top"><b>🔗 tt-tut.top</b></a>
</p>

<p align="center">
<a href="README.en.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

本教程面向仅有Python基础的Triton初学者，带你从基础的向量加到RoPE、matmul_ogs、topk等大模型算子进阶学习之路。如果没有Python基础，可以看下[Python编程入门教程(以在线评测平台为载体) ](https://www.cnblogs.com/BobHuang/p/14341687.html)，把Python 基础语法搞通。

[BobHuang](https://github.com/sBobHuang) - [OpenMLIR](https://mlir.top)

tt@bobhuang.xyz

* 1. [Triton 简介](#Triton-简介)

* 2. [向量加算子实战](#向量加算子实战)

##  1. <a name='Triton-简介'></a>Triton-简介

[OpenAI/Triton](https://github.com/openai/triton) 是一个让你用 Python 写高性能 GPU 算子的编程语言(DSL)。

优势：写法像 NumPy，轻松利用 GPU 并行和优化特性。

应用：加速深度学习算子和自定义算子，提升大模型训练和推理性能。

##  2. <a name='向量加算子实战'></a>向量加算子实战
