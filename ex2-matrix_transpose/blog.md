摘要：本文通过矩阵转置算子的实现实践了`tl.trans` 的 Triton kernel 原语，并首次尝试了2D grid。

## 矩阵转置算子实践

[Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)矩阵转置将索引对应到`output[j][i]=input[i][j]`。torch的底层`CUDA`实现会利用共享内存，padding 解决 bank conflict等等优化，这是一个memory-bound 的算子, 因为其核心是完成矩阵内存排布的转换。具体可以参考[[CUDA 学习笔记] 矩阵转置算子优化](https://zhuanlan.zhihu.com/p/692010210)。

# 1、1D grid 进行转置

把program_id直接对应到数组的索引下标是在我们之前基础上可以想到的解决方法。我们使用`rows * cols`的 grid，然后去做对应的store是一个很朴素的想法。具体代码如下所示

```Python
import torch
import triton
import triton.language as tl

# 矩阵转置 即 output[j][i]=input[i][j]
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)

    # load旧值
    t = tl.load(input_ptr + pid)

    # 计算行坐标
    row_index = pid // cols

    # 计算列坐标
    col_index = pid % cols

    # 计算新位置
    new_index = col_index * rows + row_index

    # 按照新位置去存
    tl.store(output_ptr + new_index, t)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = (rows*cols, )
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols
    )


if __name__ == "__main__":
    rows, cols = 63, 72
    a = torch.randn((rows, cols), device='cuda')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='cuda')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

LeetGPU上本题的rows, cols数据范围比较小，`1 ≤ rows, cols ≤ 8192` 所以直接过了。

# 2、2D grid 进行转置

1D可以过，2D是同理的，我们可以将`grid`设置为`(rows, cols)`，那分别使用`row_index`和`col_index`就可以了。具体代码如下所示

```Python
import torch
import triton
import triton.language as tl

# 矩阵转置 即 output[j][i]=input[i][j]
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    # 获取当前 program 在 grid 中的索引（第 0 维）, 即 row_index
    row_index = tl.program_id(axis=0)
    # 获取当前 program 在 grid 中的索引（第 1 维）, 即 col_index
    col_index = tl.program_id(axis=1)

    # 计算旧位置
    old_index = row_index * cols + col_index
    # 从旧位置去取
    t = tl.load(input_ptr + old_index)

    # 计算新位置
    new_index = col_index * rows + row_index
    # 按照新位置去存
    tl.store(output_ptr + new_index, t)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols
    )

if __name__ == "__main__":
    rows, cols = 63, 72
    a = torch.randn((rows, cols), device='cuda')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='cuda')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

1D和2D的实现在Triton的时间是差不多的，均为21.6ms。与CUDA的native实现时间一样。

# 3、tl.trans 原语

Triton存在[tl.trans](https://github.com/triton-lang/triton/blob/c817b9b63d40ead1ed023b7663f5ea14f676f4bc/python/triton/language/core.py#L1740)，我们需要使用其加速按块转置加速我们的Kenrel。

```
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
    effectively transposing a 2D tensor.

    :param input: The input tensor.
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        trans(x, (2, 1, 0))
        trans(x, 2, 1, 0)

    :py:func:`permute` is equivalent to this function, except it doesn't
    have the special case when no permutation is specified.
```

这段话就是告诉你，`tl.trans`实际上是对张量进行重排，参数没指定的话，相当于对二维张量做转置。其支持通过元组传入`dims`参数，也可以直接传多个`dims`参数。
