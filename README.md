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
* 二、 [向量加算子实战](ex1-vector_add/blog.md)
 * torch的向量加法
 * 单program 16个元素加法和验证
 * 通过mask控制元素访问
 * 多Block(program)运行
 * 使用参数化的BLOCK_SIZE
* 附录1、Triton kernel 原语覆盖情况
* 附录2、kernel 优化机制覆盖情况(未覆盖)
* 附录3、[其他项目及资料](other_repo.md)

## Triton-简介

[OpenAI/Triton](https://github.com/openai/triton) 是一个让你用 Python 写高性能 GPU 算子的编程语言(DSL)。目前有[NVIDIA](https://github.com/triton-lang/triton/tree/main/third_party/nvidia)、[AMD](https://github.com/triton-lang/triton/tree/main/third_party/amd)、[华为昇腾](https://github.com/Ascend/triton-ascend)、[寒武纪](https://github.com/FlagTree/flagtree/tree/main/third_party/cambricon)、[摩尔线程](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads)、[沐曦](https://github.com/FlagTree/flagtree/tree/main/third_party/metax)等多个后端，一个kernel**多种硬件**均可以运行，具体见[FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems)。

优势：写法像 NumPy，轻松利用 GPU 并行和优化特性。

应用：加速深度学习算子和自定义算子，提升大模型训练和推理性能。

本教程使用 Triton 3.4.0(released on 2025, Jul 31)，只需安装 torch==2.8.0。若使用较低版本的 PyTorch，可自行升级 Triton版本。Triton具有很好的版本兼容，大部分算子对Triton版本**没有要求**。

## [向量加算子实战](ex1-vector_add/blog.md)

## 附录1、[Triton kernel 原语](https://triton-lang.org/main/python-api/triton.language.html)覆盖情况

| 原语                              | 类型             | 功能说明                                        | 示例代码                                        | 首次出现上下文     |
|-----------------------------------|------------------|-------------------------------------------------|-------------------------------------------------|--------------------|
| `tl.arange`                       | 索引生成         | 生成连续索引向量，用于访问 GPU 数据              | `offsets = tl.arange(0, 16)`                   | 向量加算子实战     |
| `tl.program_id`                   | 线程/程序管理   | 获取当前 kernel program 的 ID（类似 CUDA 的 blockIdx） | `pid = tl.program_id(0)`                        | 向量加算子实战     |
| `tl.load`                         | 内存访问         | 从 GPU 全局或共享内存读取元素，可配 mask         | `a = tl.load(a_ptr + offsets, mask=mask)`       | 向量加算子实战     |
| `tl.store`                        | 内存访问         | 将数据写入 GPU 内存，可配 mask 控制写入位置      | `tl.store(c_ptr + offsets, c, mask=mask)`       | 向量加算子实战     |
| `tl.constexpr`                    | 编译时常量       | 用于 kernel 参数的编译时常量声明                 | `BLOCK_SIZE: tl.constexpr`                      |  待书写 |
| `tl.cat`                          | 创建操作         | 拼接多个 tensor                                  | `tl.cat([a, b], dim=0)`                         | 待书写 |
| `tl.full`                         | 创建操作         | 创建填充指定值的 tensor                           | `tl.full((2, 2), 1.0)`                          | 待书写 |
| `tl.zeros`                        | 创建操作         | 创建全零 tensor                                   | `tl.zeros((2, 2))`                              | 待书写 |
| `tl.zeros_like`                   | 创建操作         | 创建与给定 tensor 相同形状的全零 tensor           | `tl.zeros_like(a)`                              | 待书写 |
| `tl.cast`                         | 类型转换         | 将 tensor 转换为指定数据类型                      | `tl.cast(a, tl.float32)`                        | 待书写 |
| `tl.broadcast`                     | 形状操作         | 广播两个 tensor 至兼容形状                        | `tl.broadcast(a, b)`                            | 待书写 |
| `tl.broadcast_to`                 | 形状操作         | 将 tensor 广播到指定形状                         | `tl.broadcast_to(a, (2, 2))`                    | 待书写 |
| `tl.expand_dims`                  | 形状操作         | 在指定维度插入长度为 1 的新维度                   | `tl.expand_dims(a, 0)`                          | 待书写 |
| `tl.interleave`                   | 形状操作         | 在最后一个维度交错两个 tensor 的值                | `tl.interleave(a, b)`                           | 待书写 |
| `tl.join`                         | 形状操作         | 在新的维度上连接多个 tensor                      | `tl.join([a, b], dim=0)`                        | 待书写 |
| `tl.permute`                      | 形状操作         | 重排 tensor 的维度                               | `tl.permute(a, (1, 0))`                         | 待书写 |
| `tl.ravel`                        | 形状操作         | 返回 tensor 的扁平化视图                         | `tl.ravel(a)`                                   | 待书写 |
| `tl.reshape`                      | 形状操作         | 改变 tensor 的形状                               | `tl.reshape(a, (2, 2))`                         | 待书写 |
| `tl.split`                        | 形状操作         | 将 tensor 在最后一个维度上分割为两个 tensor        | `tl.split(a, 2)`                                | 待书写 |
| `tl.trans`                        | 形状操作         | 转置 tensor 的维度                               | `tl.trans(a)`                                   | 待书写 |
| `tl.view`                         | 形状操作         | 返回具有相同数据但不同形状的 tensor               | `tl.view(a, (2, 2))`                            | 待书写 |
| `tl.dot`                          | 线性代数操作     | 计算两个 tensor 的矩阵乘积                       | `tl.dot(a, b)`                                  | 待书写 |
| `tl.dot_scaled`                   | 线性代数操作     | 计算两个 tensor 的矩阵乘积，支持微缩格式           | `tl.dot_scaled(a, b)`                           | 待书写 |
| `tl.load`                         | 内存/指针操作    | 从内存中加载数据                                 | `tl.load(a_ptr + offsets, mask=mask)`           | 待书写 |
| `tl.store`                        | 内存/指针操作    | 将数据存储到内存中                               | `tl.store(c_ptr + offsets, c, mask=mask)`       | 待书写 |
| `tl.make_tensor_descriptor`       | 内存/指针操作    | 创建 tensor 描述符                                | `tl.make_tensor_descriptor(a)`                  | 待书写 |
| `tl.load_tensor_descriptor`       | 内存/指针操作    | 从 tensor 描述符中加载数据                        | `tl.load_tensor_descriptor(a_desc)`             | 待书写 |
| `tl.store_tensor_descriptor`      | 内存/指针操作    | 将数据存储到 tensor 描述符中                     | `tl.store_tensor_descriptor(c_desc, c)`         | 待书写 |
| `tl.make_block_ptr`               | 内存/指针操作    | 创建指向 tensor 块的指针                         | `tl.make_block_ptr(a, (0, 0))`                  | 待书写 |
| `tl.advance`                      | 内存/指针操作    | 将指针前进指定的偏移量                           | `tl.advance(a_ptr, 1)`                          | 待书写 |
| `tl.flip`                         | 索引操作         | 在指定维度上翻转 tensor                          | `tl.flip(a, 0)`                                 | 待书写 |
| `tl.where`                        | 索引操作         | 根据条件选择 tensor 中的元素                     | `tl.where(mask, a, b)`                          | 待书写 |
| `tl.swizzle2d`                    | 索引操作         | 将二维矩阵的行主索引转换为列主索引                 | `tl.swizzle2d(a)`                               | 待书写 |
| `tl.abs`                          | 数学操作         | 计算 tensor 中每个元素的绝对值                   | `tl.abs(a)`                                     | 待书写 |
| `tl.cdiv`                         | 数学操作         | 计算 tensor 中每个元素的上取整除法               | `tl.cdiv(a, b)`                                 | 待书写 |
| `tl.ceil`                         | 数学操作         | 计算 tensor 中每个元素的上取整                   | `tl.ceil(a)`                                    | 待书写 |
| `tl.clamp`                        | 数学操作         | 将 tensor 中每个元素限制在指定范围内             | `tl.clamp(a, 0, 1)`                             | 待书写 |
| `tl.cos`                          | 数学操作         | 计算 tensor 中每个元素的余弦值                   | `tl.cos(a)`                                     | 待书写 |
| `tl.div_rn`                       | 数学操作         | 计算 tensor 中每个元素的精确除法（四舍五入）      | `tl.div_rn(a, b)`                               | 待书写 |
| `tl.erf`                          | 数学操作         | 计算 tensor 中每个元素的误差函数                 | `tl.erf(a)`                                     | 待书写 |
| `tl.exp`                          | 数学操作         | 计算 tensor 中每个元素的指数值                   | `tl.exp(a)`                                     | 待书写 |
| `tl.exp2`                         | 数学操作         | 计算 tensor 中每个元素的以 2 为底的指数值         | `tl.exp2(a)`                                    | 待书写 |
| `tl.fdiv`                         | 数学操作         | 计算 tensor 中每个元素的快速除法                 | `tl.fdiv(a, b)`                                 | 待书写 |
| `tl.floor`                        | 数学操作         | 计算 tensor 中每个元素的下取整                   | `tl.floor(a)`                                   | 待书写 |
| `tl.fma`                          | 数学操作         | 计算 tensor 中每个元素的乘加（乘法加法）           | `tl.fma(a, b, c)`                               | 待书写 |
| `tl.log`                          | 数学操作         | 计算 tensor 中每个元素的自然对数                 | `tl.log(a)`                                     | 待书写 |

## 附录2、kernel 优化机制覆盖情况(未覆盖)

# Triton Kernel 优化机制

| 优化机制 | 类型 | 功能说明 | 示例 / 用法 |
|----------|------|----------|-------------|
| `autotune` | 自动调优 | 自动尝试多组 kernel 配置参数（如 BLOCK_SIZE、NUM_WARPS），选择最优性能组合 | `@triton.autotune configs=[{'BLOCK_SIZE': 64}, {'BLOCK_SIZE': 128}]` |
| `num_warps` | 并行度配置 | 设置每个 kernel block 使用的 warp 数量，提高 GPU 并行利用率 | `@triton.jit(num_warps=4)` |
| `BLOCK_SIZE` | tile / block 配置 | 设置每个 block 处理的数据量，影响并行度和内存访问效率 | `BLOCK_SIZE: tl.constexpr` |
