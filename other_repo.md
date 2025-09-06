## 关联项目及学习资料

### 1、[Triton tutorials](https://github.com/triton-lang/triton/blob/main/python/tutorials)

向量加就取于Triton的[python/tutorials/01-vector-add.py](https://github.com/triton-lang/triton/tree/main/python/tutorials/01-vector-add.py)，他还提供了包括[fused-attention](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)、[fused-softmax](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py)、[grouped-gemm](https://github.com/triton-lang/triton/blob/main/python/tutorials/08-grouped-gemm.py)在内的示例。

### 2、[Gluon tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)

Triton官方推出的可以控制内存、layout和调度等细粒度控制的新语言。提供了[Warp Specialization](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py)、[Persistent Kernels](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py)、[The 5th Generation TensorCore^TM](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/06-tcgen05.py)、[Gluon Attention](https://github.com/triton-lang/triton/blob/main/python/examples/gluon/01-attention-forward.py)在内的示例。

### 3、[triton_kernels](https://github.com/triton-lang/triton/tree/main/python/triton_kernels)

Triton官方推出的高性能kernel，有topk、matmul、swiglu、routing等高高性能算子，[gpt-oss](https://github.com/openai/gpt-oss) 就使用了此kernel集，目前也被各推理框架集成。

### 3、[LeetGPU 答案](https://github.com/OpenMLIR/leetgpu-challenges)

目前LeetGPU easy级别的全部Triton答案我已公开到此项目中，本教程将持续使用LeetGPU中的题目做为教程的例题，直接一步大模型算子不容易。

### 4、[FlagGems](https://github.com/FlagOpen/FlagGems)

FlagGems是清华智源高性能通用 AI 算子库，目前已加入 PyTorch 生态项目体系。通过提供一套内核函数，加速大语言模型的训练和推理过程。通过在 PyTorch 的 ATen 后端进行注册，FlagGems 让用户无需修改模型代码即可切换到 Triton 函数库。历时一年多的打造，FlagGems 已经成为全球支持芯片种类最多、数量最大的（超过 180 个）Triton 语言算子库。

### 5、[GPU MODE Lecture 14: Practitioners Guide to Triton](https://www.youtube.com/watch?v=DdTsX6DQk24)

GPU MODE 是一个专注于 GPU 编程的开源社区组织，旨在通过互动式学习和工具开发，提升开发者在高性能计算（HPC）、深度学习系统和 GPU 编程的能力。[Triton Kernel collection by cuda-mode](https://github.com/cuda-mode/triton-index) 是他们的Triton kernel集。

### 6、[linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)

高性能用于LLM 训练的Triton kernel。

### 7、[Puzzles by Sasha Rush](https://github.com/srush/Triton-Puzzles)

Triton-Puzzles 是由 Sasha Rush（srush）等人创建的一个开源项目，旨在通过一系列循序渐进的练习题，帮助开发者深入理解 Triton 编程语言的核心概念和实践应用。

### 8、[inccat/Awesome-Triton-Kernels](https://github.com/zinccat/Awesome-Triton-Kernels)

是一个汇总

### 9、本人关于Triton的博客

[浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)

[triton是否会冲击cuda生态？BobHuang的回答](https://www.zhihu.com/question/1919763006750975400/answer/1921121681612739823)

[LeetGPU入门教程 (CUDA guide最佳实践)](https://zhuanlan.zhihu.com/p/1899956367734867434)

[深度剖析 Triton编译器 MatMul优化（一）—— FMA](https://zhuanlan.zhihu.com/p/1922542705797465957)

[深度剖析 Triton编译器 MatMul优化（二）—— MMA](https://zhuanlan.zhihu.com/p/1922921325296615496)

[深度剖析 Triton编译器 MatMul优化（三）—— TMA](https://zhuanlan.zhihu.com/p/1924011555437155686)

[Triton Kernel 优先：全新 LLM 推理方式(47e9dcb)](https://zhuanlan.zhihu.com/p/1939592984820691987)

[Triton多层级runner v0.1.5：支持缓存机制，Benchmark更友好 (9c28df1)](https://zhuanlan.zhihu.com/p/1931261279072396108)

[Triton OpenCL 后端开发：矩阵乘实现验证(953bff6)](https://zhuanlan.zhihu.com/p/1925309765489230184)

[Triton 社区首贡献：Bug 修复实录](https://zhuanlan.zhihu.com/p/1917136776885174369)

[CUDA优化黑魔法：假装CUTLASS库(Triton PR7298)](https://zhuanlan.zhihu.com/p/1926902370920568120)
