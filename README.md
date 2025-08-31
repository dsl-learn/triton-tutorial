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
* 附录3、[关联项目及学习资料](other_repo.md)

## Triton-简介

[OpenAI/Triton](https://github.com/openai/triton) 是一个让你用 Python 写高性能 GPU 算子的编程语言(DSL)。目前有[NVIDIA](https://github.com/triton-lang/triton/tree/main/third_party/nvidia)、[AMD](https://github.com/triton-lang/triton/tree/main/third_party/amd)、[华为昇腾](https://github.com/Ascend/triton-ascend)、[寒武纪](https://github.com/FlagTree/flagtree/tree/main/third_party/cambricon)、[摩尔线程](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads)、[沐曦](https://github.com/FlagTree/flagtree/tree/main/third_party/metax)等多个后端，一个kernel**多种硬件**均可以运行，具体见[FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems)。

优势：写法像 NumPy，轻松利用 GPU 并行和优化特性。

应用：加速深度学习算子和自定义算子，提升大模型训练和推理性能。

本教程使用 Triton 3.4.0(released on 2025, Jul 31)，只需安装 torch==2.8.0。若使用较低版本的 PyTorch，可自行升级 Triton版本。Triton具有很好的版本兼容，大部分算子对Triton版本**没有要求**。

## [向量加算子实战](ex1-vector_add/blog.md)

## 附录1、[Triton kernel 原语](https://triton-lang.org/main/python-api/triton.language.html)覆盖情况
| 原语                           | 类型    | 功能                     | 示例                                      | 首次出现  |
| ---------------------------- | ----- | ---------------------- | --------------------------------------- | ----- |
| `tl.arange`                  | 索引    | 生成连续索引向量               | `offsets = tl.arange(0,16)`             | 向量加算子 |
| `tl.program_id`              | 线程/程序 | 获取 kernel program ID   | `pid = tl.program_id(0)`                | 向量加算子 |
| `tl.load`                    | 内存    | 读取 GPU 全局/共享内存，可配 mask | `a = tl.load(a_ptr+offsets, mask=mask)` | 向量加算子 |
| `tl.store`                   | 内存    | 写入 GPU 内存，可配 mask      | `tl.store(c_ptr+offsets, c, mask=mask)` | 向量加算子 |
| `tl.constexpr`               | 编译常量  | 声明 kernel 参数常量         | `BLOCK_SIZE: tl.constexpr`              | 待书写   |
| `tl.cat`                     | 创建    | 拼接多个 tensor            | `tl.cat([a,b],dim=0)`                   | 待书写   |
| `tl.full`                    | 创建    | 创建指定值 tensor           | `tl.full((2,2),1.0)`                    | 待书写   |
| `tl.zeros`                   | 创建    | 全零 tensor              | `tl.zeros((2,2))`                       | 待书写   |
| `tl.zeros_like`              | 创建    | 与 tensor 同形全零          | `tl.zeros_like(a)`                      | 待书写   |
| `tl.cast`                    | 类型    | 转换 tensor 数据类型         | `tl.cast(a,tl.float32)`                 | 待书写   |
| `tl.broadcast`               | 形状    | 广播 tensor 至兼容形状        | `tl.broadcast(a,b)`                     | 待书写   |
| `tl.broadcast_to`            | 形状    | 广播到指定形状                | `tl.broadcast_to(a,(2,2))`              | 待书写   |
| `tl.expand_dims`             | 形状    | 指定维度插入长度1维度            | `tl.expand_dims(a,0)`                   | 待书写   |
| `tl.interleave`              | 形状    | 最后维度交错两个 tensor        | `tl.interleave(a,b)`                    | 待书写   |
| `tl.join`                    | 形状    | 新维度连接 tensor           | `tl.join([a,b],dim=0)`                  | 待书写   |
| `tl.permute`                 | 形状    | 重排维度                   | `tl.permute(a,(1,0))`                   | 待书写   |
| `tl.ravel`                   | 形状    | 扁平化 tensor             | `tl.ravel(a)`                           | 待书写   |
| `tl.reshape`                 | 形状    | 改变 tensor 形状           | `tl.reshape(a,(2,2))`                   | 待书写   |
| `tl.split`                   | 形状    | 最后维度分割                 | `tl.split(a,2)`                         | 待书写   |
| `tl.trans`                   | 形状    | 转置维度                   | `tl.trans(a)`                           | 待书写   |
| `tl.view`                    | 形状    | 返回不同形状视图               | `tl.view(a,(2,2))`                      | 待书写   |
| `tl.dot`                     | 线性代数  | 矩阵乘积                   | `tl.dot(a,b)`                           | 待书写   |
| `tl.dot_scaled`              | 线性代数  | 矩阵乘积，支持缩放              | `tl.dot_scaled(a,b)`                    | 待书写   |
| `tl.make_tensor_descriptor`  | 内存/指针 | 创建 tensor 描述符          | `tl.make_tensor_descriptor(a)`          | 待书写   |
| `tl.load_tensor_descriptor`  | 内存/指针 | 从描述符加载数据               | `tl.load_tensor_descriptor(a_desc)`     | 待书写   |
| `tl.store_tensor_descriptor` | 内存/指针 | 存储数据到描述符               | `tl.store_tensor_descriptor(c_desc,c)`  | 待书写   |
| `tl.make_block_ptr`          | 内存/指针 | 指向 tensor 块指针          | `tl.make_block_ptr(a,(0,0))`            | 待书写   |
| `tl.advance`                 | 内存/指针 | 指针偏移                   | `tl.advance(a_ptr,1)`                   | 待书写   |
| `tl.flip`                    | 索引    | 指定维度翻转                 | `tl.flip(a,0)`                          | 待书写   |
| `tl.where`                   | 索引    | 条件选择元素                 | `tl.where(mask,a,b)`                    | 待书写   |
| `tl.swizzle2d`               | 索引    | 2D 索引行列互换              | `tl.swizzle2d(a)`                       | 待书写   |
| `tl.abs`                     | 数学    | 元素绝对值                  | `tl.abs(a)`                             | 待书写   |
| `tl.cdiv`                    | 数学    | 元素上取整除法                | `tl.cdiv(a,b)`                          | 待书写   |
| `tl.ceil`                    | 数学    | 上取整                    | `tl.ceil(a)`                            | 待书写   |
| `tl.clamp`                   | 数学    | 限定元素范围                 | `tl.clamp(a,0,1)`                       | 待书写   |
| `tl.cos`                     | 数学    | 元素余弦                   | `tl.cos(a)`                             | 待书写   |
| `tl.div_rn`                  | 数学    | 精确除法四舍五入               | `tl.div_rn(a,b)`                        | 待书写   |
| `tl.erf`                     | 数学    | 误差函数                   | `tl.erf(a)`                             | 待书写   |
| `tl.exp`                     | 数学    | 指数                     | `tl.exp(a)`                             | 待书写   |
| `tl.exp2`                    | 数学    | 2 为底指数                 | `tl.exp2(a)`                            | 待书写   |
| `tl.fdiv`                    | 数学    | 快速除法                   | `tl.fdiv(a,b)`                          | 待书写   |
| `tl.floor`                   | 数学    | 下取整                    | `tl.floor(a)`                           | 待书写   |
| `tl.fma`                     | 数学    | 乘加运算                   | `tl.fma(a,b,c)`                         | 待书写   |
| `tl.log`                     | 数学    | 自然对数                   | `tl.log(a)`                             | 待书写   |

## 附录2、kernel 优化机制覆盖情况(未覆盖)

# Triton Kernel 优化机制

| 优化机制 | 类型 | 功能说明 | 示例 / 用法 |
|----------|------|----------|-------------|
| `autotune` | 自动调优 | 自动尝试多组 kernel 配置参数（如 BLOCK_SIZE、NUM_WARPS），选择最优性能组合 | `@triton.autotune configs=[{'BLOCK_SIZE': 64}, {'BLOCK_SIZE': 128}]` |
| `num_warps` | 并行度配置 | 设置每个 kernel block 使用的 warp 数量，提高 GPU 并行利用率 | `@triton.jit(num_warps=4)` |
| `BLOCK_SIZE` | tile / block 配置 | 设置每个 block 处理的数据量，影响并行度和内存访问效率 | `BLOCK_SIZE: tl.constexpr` |
