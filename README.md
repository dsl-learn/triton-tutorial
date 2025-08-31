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

本教程面向没有 GPU 经验的的Triton初学者，带你从基础的向量加到RoPE、matmul_ogs、topk、Gluon Attention
等大模型算子进阶学习之路。如果没有Python基础，可以通过[Python编程入门教程(以在线评测平台为载体)](https://www.cnblogs.com/BobHuang/p/14341687.html)来学习 Python 语法，或者根据本教程内容与 ChatGPT 对话直接入门 Triton。

作者：[BobHuang](https://github.com/sBobHuang) - [OpenMLIR](https://mlir.top)

作者邮箱：tt@bobhuang.xyz

* 一、 [Triton 简介](#Triton-简介)

* 二、 [向量加算子实战](#向量加算子实战)

##  一、 <a name='Triton-简介'></a>Triton-简介

[OpenAI/Triton](https://github.com/openai/triton) 是一个让你用 Python 写高性能 GPU 算子的编程语言(DSL)。目前有[NVIDIA](https://github.com/triton-lang/triton/tree/main/third_party/nvidia)、[AMD](https://github.com/triton-lang/triton/tree/main/third_party/amd)、[华为昇腾](https://github.com/Ascend/triton-ascend)、[寒武纪](https://github.com/FlagTree/flagtree/tree/main/third_party/cambricon)、[摩尔线程](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads)、[沐曦](https://github.com/FlagTree/flagtree/tree/main/third_party/metax)等多个后端，一个kernel**多种硬件**均可以运行，具体见[FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems)。

优势：写法像 NumPy，轻松利用 GPU 并行和优化特性。

应用：加速深度学习算子和自定义算子，提升大模型训练和推理性能。

##  二、 <a name='向量加算子实战'></a>向量加算子实战

本教程使用 Triton 3.4.0(released on 2025, Jul 31)，只需安装 torch==2.8.0。若使用较低版本的 PyTorch，可自行升级 Triton版本。Triton具有很好的版本兼容，大部分算子对Triton版本**没有要求**。

入门先学 a + b，向量加法可以表示为 向量c = 向量a + 向量b，即把 a 和 b 中对应位置的每个数字相加。

我们先用Pytorch来实现下，我们可以用 torch.randn 来生成随机的向量a、b，在torch里直接相加就可以。

```Python
import torch

if __name__ == "__main__":
    N = 16
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = a + b
    print(a, b, c, sep="\n")
```

可以得到如下输出结果，第三个tensor的值是前两个tensor对应位置相加。由于是随机数据，所以以下输出结果会变化。

```
tensor([-0.3947,  0.1963,  0.4782, -0.0215,  1.5055,  0.1066, -0.8224,  0.0999,
        -0.1316,  0.3244, -1.6962, -0.1411,  0.5005,  0.0396,  0.4774,  0.9639],
       device='cuda:0')
tensor([-0.1621, -1.0437,  0.5023,  0.3897,  0.6714, -0.8212, -0.2596, -0.3467,
        -2.2264,  0.7489,  1.3961, -2.1076,  0.0119, -0.8835, -0.4079,  1.8599],
       device='cuda:0')
tensor([-0.5568, -0.8474,  0.9805,  0.3682,  2.1769, -0.7145, -1.0820, -0.2469,
        -2.3581,  1.0732, -0.3000, -2.2486,  0.5124, -0.8439,  0.0695,  2.8238],
       device='cuda:0')
```

