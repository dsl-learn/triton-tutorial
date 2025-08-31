import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)
    
    # 计算当前 block 的起始元素索引
    block_start = pid * BLOCK_SIZE  # 每个 block 处理 BLOCK_SIZE 个元素
    
    # 生成当前 block 内的连续索引 [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # mask 标记哪些索引有效（不越界），即 offsets < N 的位置为 True
    mask = offsets < N
    
    # 根据 offsets 和 mask 从 a_ptr 加载数据
    a = tl.load(a_ptr + offsets, mask=mask)
    
    # 根据 offsets 和 mask 从 b_ptr 加载数据
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # 对应位置元素相加
    c = a + b
    
    # 将结果写回到 c_ptr 对应位置，只写 mask 为 True 的元素
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 16
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
