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
<a ><b>English</b></a> | <a href="README.md"><b>中文</b></a>
</p>


This tutorial is dedicated to beginners who are not familiar with Python and GPU programming, offering a walkthrough from vector addition to RoPE, matmul_ogs, Top-K, Gluon Attention, and other important GPU kernels for Large Language Models (LLMs).

For those who haven't installed Python yet or want to learn some baic Python syntax before getting started, [Python.org](https://www.python.org/about/gettingstarted/) provides a comprehensive tutorial.

Author: [BobHuang](https://github.com/sBobHuang) - [OpenMLIR](https://mlir.top)

## Introduction of Triton

[OpenAI/Triton](https://github.com/openai/triton)  is a Python dialect that allows developers to write efficient GPU kernels using Python syntax instead of coding in C++. As to now, Triton supports multiple backends, including: [NVIDIA](https://github.com/triton-lang/triton/tree/main/third_party/nvidia), [AMD](https://github.com/triton-lang/triton/tree/main/third_party/amd), [华为昇腾](https://github.com/Ascend/triton-ascend), [寒武纪](https://github.com/FlagTree/flagtree/tree/main/third_party/cambricon)、[摩尔线程](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads), [沐曦](https://github.com/FlagTree/flagtree/tree/main/third_party/metax). This means kernel written in Triton can compile and run on different hardware architectures. For more details, see [FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems).

Advantages: Easy-to-use Python-like syntax, with high optimization for GPU performance and memory management.

Applications: Enhancing the efficiency of the inference phase of LLMs by speeding up deep learning operators and enabling customization of operators.

This repository requires Triton 3.4.0 (released on July 31, 2025), which comes with torch == 2.8.0. Since Triton has excellent backward compatibility, other versions of PyTorch might work as well if Triton is manually upgraded.