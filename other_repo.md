<!-- Triton算子开发及编译器资源整理 -->

Agent时代很多知识都失效了，我搞Triton编译器已经一年了，把我的context共享给大家。本文后续不再更新，更多信息可直接通过 AI 检索获取。

### 一、Triton是什么

Triton 可以简单理解为：

- 一个用 Python 写 GPU kernel 的 DSL
- 一个非常容易上手，AI很会的算子DSL
- 一个以 MLIR 为基础设施的编译器
- 一个正在持续扩展的多后端编程模型
- 一个在 PyTorch、LLM 推理和新硬件适配里都很有价值的中间层

我们先看官方提供的资料：

- [triton-lang/triton](https://github.com/triton-lang/triton)<br>
  官方主仓库。
- [Official tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials)<br>
  官方教程入口，非常建议看一看。 最近在更新 [block-scaled-matmul](https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py)。
- [Gluon tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)<br>
  Triton 更低层、更接近硬件控制的新方向。
- [MAPL 2019 Triton 论文](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)<br>
  看设计动机和最初模型。

### 二、入门教程

这一部分适合刚开始写 Triton kernel，或者想系统补一遍基础原语的人。不过现在Agent改中学或者和llm对话。

- [[Hands-On] Triton 教程](https://www.zhihu.com/column/c_1948447902964901167)<br>
  我写的面向没有 GPU 和 PyTorch 经验者的 Triton 教程。 写得不多，已弃更。
- [SiriusNEO/Triton-Puzzles-Lite](https://github.com/SiriusNEO/Triton-Puzzles-Lite)<br>
  很精简很小的 Triton 教程，不过现在和 llm 对话学得更快，agent 也能直接写。
- [slowlyC大佬的Gluon实践](https://www.zhihu.com/column/c_1990516179064858333)<br>
  带你一步步学习 Gluon，这个AI还不太会，有用。 可配合 [Qwen3.5 GDN Prefill Kernel 优化](https://zhuanlan.zhihu.com/p/2007935329550766500) 一起看。

### 三、算子实践

- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)<br>
  使用 Triton 实现的线性注意力仓库，知乎上相关成员的分享很多。[Lightning Attention 是如何克服传统线性注意力机制需要累加求和的缺陷的](https://www.zhihu.com/question/9740764576/answer/80735153803), [线性注意力简史：从模仿、创新到反哺](https://zhuanlan.zhihu.com/p/1923344184754026149)，[从零开始学 KDA-1](https://zhuanlan.zhihu.com/p/1989809041849988324)
- [triton_kernels](https://github.com/triton-lang/triton/tree/main/python/triton_kernels)<br>
  官方高性能 kernel 集，有 topk、matmul、swiglu、routing 等高性能 MOE 算子。
- [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)<br>
  面向 LLM 训练优化的 Triton kernel 库，可直接 patch Hugging Face 模型，兼容 FlashAttention、FSDP、DeepSpeed。
- [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)<br>
  量化 attention 加速方向的代表项目。SageAttention 1 更偏 Triton 实现，后续版本同时提供 Triton/CUDA backend，做推理 attention 优化时很值得参考。
- [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn)<br>
  同一条技术线上的稀疏 attention 推理加速项目，和 SageAttention 可以对照着看，适合关注 block-sparse attention 的实现方式。
- [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)<br>
  zhuzilin 做的长上下文 / context parallel attention 工程实现，虽然不是纯 Triton 项目，但如果你在看 attention 并行化、长序列训练和推理，这个很值得和 Triton attention 生态一起看。
- [ModelTC/LightLLM](https://github.com/ModelTC/LightLLM)<br>
  轻量级 LLM 推理框架，纯 Python 架构，很多核心算子基于 Triton 实现，适合参考真实推理场景里的 kernel 工程写法。
- [FlagGems](https://github.com/FlagOpen/FlagGems)<br>
  FlagGems 是智源高性能通用 AI 算子库，目前已加入 PyTorch 生态项目体系。

### 四、Triton工具

- [triton_to_gluon_translater](https://github.com/triton-lang/triton/tree/main/python/triton/tools/triton_to_gluon_translater)<br>
  Triton 到 Gluon 的转换工具。 相关分享：[Triton conf 2025 Gluon](https://www.youtube.com/watch?v=KqeI23SpJx8)
- [官方profiler proton](https://github.com/triton-lang/triton/tree/main/third_party/proton)<br>
  轻量级的 Triton 性能分析器。 参考：[Triton conf 2025 Proton](https://www.youtube.com/watch?v=PGUw2P55ZYM)
- [Triton Runner](https://github.com/toyaix/triton-runner)<br>
  我开发的多层级 Runner 执行工具，用于把 Triton 的各层级产物跑起来。 文档：[用户文档专栏](https://www.zhihu.com/column/c_1959013459611059049)，[开发文档专栏](https://www.zhihu.com/column/c_1940119129400013405)
- [meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench)<br>
  Triton 和其他 kernel 的 bench 工具。 可参考 [PR：Enable TileIR for FA](https://github.com/meta-pytorch/tritonbench/pull/498)，比cuTile发布早了很多。
- [libtriton_jit](https://github.com/flagos-ai/libtriton_jit)<br>
  Triton JIT 优化 launch 时间。
- [pytorch-labs/tritonparse](https://github.com/pytorch-labs/tritonparse)<br>
  Triton 解析方向。

### 五、Triton 论文

- [Triton Linear Layouts](https://arxiv.org/abs/2505.23819)<br>
  把 layout 简化建模成 **F2​** 上的线性变换，用统一数学框架表示 tensor layout、layout conversion 和 swizzle
- [OSDI ‘25 KPerfIR](https://arxiv.org/abs/2505.21661)<br>
  通过profile深度集成到编译器中，识别出 Flash-Attention-3 中的性能瓶颈，通过优化将其性能提升 24.1%，比手工调优版本高出 7.6%，和proton相关
- [CGO ‘25 CuAsmRL](https://arxiv.org/abs/2501.08071)<br>
  集成到Triton的用RL(强化学习)在sass调度上优化，最高可以带来26%的性能提升
- [TritonBench](https://arxiv.org/abs/2502.14752)<br>
  Triton 和其他 kernel 的 bench 工具，和proton有联动，[TritonBench使用示例](https://github.com/meta-pytorch/tritonbench/blob/main/tritonbench/kernels/proton_fused_attention.py)
- [Triton-distributed](https://arxiv.org/abs/2504.19442)<br>
  seed的Triton分布式原文

### 六、Triton扩展及周边

- [Triton 扩展](https://github.com/triton-lang/triton-ext)<br>
  Triton 官方提供的扩展，目前还在开发中，算子编译器开发工程师可以尽情打洞。
- [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR)<br>
  cuTile 使用了 TileIR，那 Triton 也可以转过去借助 TileIR 获得性能。
- [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)<br>
  字节 Seed 推出的 Triton 分布式扩展。 相关分享：[Triton conf 2025](https://www.youtube.com/watch?v=ccMl2KLb-iY)
- [meta TLX扩展](https://github.com/facebookexperimental/triton/tree/tlx/third_party/tlx)<br>
  更 low-level 的 Triton 扩展，用来写高性能 kernel。 TLX [Triton Warp Spec](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap)的实现介绍，[NV的wasp论文](https://arxiv.org/pdf/2512.18134)
- [智源 TLE扩展](https://github.com/flagos-ai/FlagTree/tree/triton_v3.5.x/python/triton/experimental/tle)<br>
  Triton 用于共享内存与分布式同步语义的扩展。 作者实践：[写出比 SGLang 更快的 MoE Align Block Size](https://zhuanlan.zhihu.com/p/2014316594554226118)
- [pytorch-labs/helion](https://github.com/pytorch-labs/helion)<br>
  更高层 DSL 再往 Triton lower 的方向。 可配合 [Helion(TileTorch)的初步实践](https://zhuanlan.zhihu.com/p/1967505312484429901) 一起看。
- [InfiniTensor/ninetoothed](https://github.com/InfiniTensor/ninetoothed)<br>
  九齿（NineToothed）是一个基于 Triton 的更高层 DSL，引入 tensor-oriented metaprogramming，适合关注 Triton 之上更高抽象的人看。

### 七、Triton nightly

- [Triton nightly](https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/triton/)<br>
  需要和 [Github](https://github.com/triton-lang/triton/actions/workflows/wheels.yml?query=is%3Asuccess) 页面对应 commit。
- [Triton relase nightly](https://download.pytorch.org/whl/nightly/triton)

### 八、训练推理框架

- [unslothai/unsloth](https://github.com/unslothai/unsloth)<br>
  面向训练、微调和推理加速的框架，很多核心 kernel 是 Triton 手写的，RoPE、MLP 这类融合算子都值得参考。
- [ModelTC/LightLLM](https://github.com/ModelTC/LightLLM)<br>
  面向大模型推理与服务部署的框架，是 Triton 在真实推理系统中的典型落地案例。
- [OpenAI/gpt-oss](https://github.com/openai/gpt-oss)<br>
  Triton 在真实 LLM 推理里的第一个模型。
- [TritonLLM](https://github.com/toyaix/TritonLLM)<br>
  我做的 Triton 算子优先的大模型推理，5090 的 gpt-oss 单卡单请求超过 ollama 了，文档：[[开发文档] TritonLLM](https://www.zhihu.com/column/c_1936669246047393422)，已弃更。建议[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)。

### 九、Triton编译器开发

可以先看 [浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)来建立对Triton编译的认知

- [FlagTree多后端](https://github.com/flagos-ai/FlagTree)<br>
  国产卡 Triton 后端值得重点关注。
- [Triton OpenCL](https://github.com/toyaix/triton-ocl)<br>
  我开发的玩具 OpenCL 后端。 开发文档见 [Triton新后端接入](https://www.zhihu.com/column/c_1906884474676945862)，该项目已弃更。
- [microsoft/triton-shared](https://github.com/microsoft/triton-shared)<br>
  Triton -> Linalg 方向的重要参考。
- [Cambricon/triton-linalg](https://github.com/Cambricon/triton-linalg)<br>
  寒武纪开源的 Triton linalg。[Triton-Linalg](https://zhuanlan.zhihu.com/p/707274848) 很好的文章。
- [intel/intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)
- [triton-lang/triton-cpu](https://github.com/triton-lang/triton-cpu)
- [Ascend/triton-ascend](https://github.com/Ascend/triton-ascend)
- [Triton 社区首贡献：Bug 修复实录](https://zhuanlan.zhihu.com/p/1917136776885174369)<br>
  我在 Triton 修复一个 Bug 的记录，适合看如何从报错一路定位到源码和 Pass。
- [剖析 Triton编译器 MatMul优化三部曲](https://www.zhihu.com/column/c_1948449915928807129)<br>
  我分析了 [FMA](https://zhuanlan.zhihu.com/p/1922542705797465957)、[MMA](https://zhuanlan.zhihu.com/p/1922921325296615496)、[TMA](https://zhuanlan.zhihu.com/p/1924011555437155686) 三类矩阵乘法的 lower 流程，找出了每一个 IR 发生变化的 Pass。
- [MLIR学习可以参考的项目](https://zhuanlan.zhihu.com/p/1924384457349132481)<br>
  我整理的 MLIR 项目，有些项目有些过时。
- [MLIR IR 阅读指南：解构核心概念、Pass 模式与 Triton 编译实践](https://zhuanlan.zhihu.com/p/1974197504993150241)<br>
  平6层有非常多Triton有关的文章，可以阅读。
- [Triton Linear Layouts笔记](https://zhuanlan.zhihu.com/p/1915468456821788929)<br>
  思泽哥的解读
- [OpenAI Triton: Why layout is important](https://zhuanlan.zhihu.com/p/672720213)<br>
  二球带你一步步推layout

### 十、有趣的commit 或 PR

- [tl.make_block_ptr 废弃](https://github.com/triton-lang/triton/commit/3bed8f599608b113fc1c39a7434cfb42a587eb68)<br>
  RewriteTensorPointerPass 也删了。
- [slp-copyable-elements bug](https://github.com/triton-lang/triton/commit/9844da9)
- [Gluon fa SM120](https://github.com/triton-lang/triton/pull/9600)
- [MLIR加Triton变量名](https://github.com/triton-lang/triton/pull/7521)
- [B200 flex_attention_fwd 18% perf regression](https://github.com/triton-lang/triton/issues/8328)<br>
  一个有趣的性能 issue。
