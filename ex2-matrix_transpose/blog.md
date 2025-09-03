## 上次练习解答

上次练习中的前两道题还是比较简单的，我们先动手做一下。

1、[Matrix Copy](https://leetgpu.com/challenges/matrix-copy)

在 GPU 上实现一个程序，将输入的 `N×N` 浮点矩阵 `A` 元素逐一复制到输出矩阵 `B`，即满足`A[i][j]=B[i][j]`。

其实就是`N×N`的元素拷贝到另外一个地方，需要你计算出`a+b`的总长度就可以了。以下是参考答案

```Python
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(
    input_ptr,               # 输入矩阵的指针
    output_ptr,              # 输出矩阵的指针
    n_elements,              # 总元素个数 (N * N)
    BLOCK_SIZE: tl.constexpr # 每个 block 处理的元素个数
):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)

    # 计算当前 block 的起始元素索引
    block_start = pid * BLOCK_SIZE

    # 生成当前 block 内的连续索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask 标记哪些索引有效（不越界）
    mask = offsets < n_elements

    # 根据 offsets 和 mask 从 input_ptr 加载数据
    val = tl.load(input_ptr + offsets, mask=mask)

    # 将结果按照 mask 写回到 output_ptr 对应位置
    tl.store(output_ptr + offsets, val, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    n_elements = N * N
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    copy_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    N = 2
    a = torch.randn((N, N), device='cuda')
    torch_output = a.clone()
    triton_output = torch.empty_like(a)
    solve(a, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

2、[Color Inversion](https://leetgpu.com/challenges/color-inversion)