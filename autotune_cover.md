## kernel 优化机制覆盖情况(未覆盖)

| 优化机制 | 类型 | 功能说明 | 示例 / 用法 |
|----------|------|----------|-------------|
| `autotune` | 自动调优 | 自动尝试多组 kernel 配置参数（如 BLOCK_SIZE、NUM_WARPS），选择最优性能组合 | `@triton.autotune configs=[{'BLOCK_SIZE': 64}, {'BLOCK_SIZE': 128}]` |
| `num_warps` | 并行度配置 | 设置每个 kernel block 使用的 warp 数量，提高 GPU 并行利用率 | `@triton.jit(num_warps=4)` |
| `BLOCK_SIZE` | tile / block 配置 | 设置每个 block 处理的数据量，影响并行度和内存访问效率 | `BLOCK_SIZE: tl.constexpr` |