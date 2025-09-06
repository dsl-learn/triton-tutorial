摘要：本文通过矩阵转置算子的实现实践了`tl.trans` 的 Triton kernel 原语，并首次尝试了2D grid。

# 矩阵转置算子实践

[Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)矩阵转置将索引对应到`output[j][i]=input[i][j]`。torch的底层`CUDA`实现会利用共享内存，padding 解决 bank conflict等等优化，这是一个memory-bound 的算子, 因为其核心是完成矩阵内存排布的转换。具体可以参考[[CUDA 学习笔记] 矩阵转置算子优化](https://zhuanlan.zhihu.com/p/692010210)。

## 1、1D grid 进行native转置

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

## 2、2D grid 进行native转置

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

## 3、tl.trans 原语

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

## 4、维度扩展的1x1 block实践

根据其原语，我们只要定义出`block`，然后直接使用`tl.trans(block)`就可以了。

在`a+b`中我们使用了`tl.arange`做了`1D block`，代码如下所示。

```Python
    # 生成连续索引 [0, 1, ..., 15]，用于访问 16 个元素
    offsets = tl.arange(0, 16)
    # 根据索引从 a_ptr 指向的地址加载 16 个元素
    a = tl.load(a_ptr + offsets)
```

我们先尝试`1x1`的`2D block`。这里需要引入一个新概念。PyTorch 的广播与维度扩展，其实就是通过**切片语法**增加新维度。比如存在一个`torch.arange(5)`的名为`offs`的`tensor`，他的shape为`torch.Size([5])`，值为`tensor([0, 1, 2, 3, 4])`。那么我使用`offs[:, None]`就是在 最后一维后面再加一个新维度，`:`是Python的切片语法，`,` 是维度分隔符，`None`表示一个新axis(轴)，`offs[:, None]`的shape为`torch.Size([5, 1])`的shape。同理使用`offs[None, :]`可以得到一个`torch.Size([1, 5])`的shape。代码如下所示，你可以运行下代码来理解清楚

```Python
if __name__ == "__main__":
    import torch
    offs = torch.arange(5)
    print('offs', offs.shape, offs)
    offs_col = offs[:, None]
    print('offs_col', offs_col.shape, offs_col)
    offs_row = offs[None, :]
    print('offs_row', offs_row.shape, offs_row)
```

有了`维度扩展`，我们就可以表达2D的block了。使用`offs_row[:, None] * cols + offs_col[None, :]`即可，具体代码如下所示。

```Python
    # 旧 block 内行偏移量
    offs_row = row_index + tl.arange(0, 1)
    # 旧 block 内列偏移量
    offs_col = col_index + tl.arange(0, 1)

    # 类似 row_index * cols + col_index
    # 行偏移量 * 列数 + 列偏移量 = 行偏移 + 列偏移 = 偏移
    old_offs = offs_row[:, None] * cols + offs_col[None, :]
```

那么load出来使用 `tl.trans(block)`最后再store就可以了，代码如下所示。

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
    # 旧 block 内行偏移量
    offs_row = row_index + tl.arange(0, 1)
    # 旧 block 内列偏移量
    offs_col = col_index + tl.arange(0, 1)

    # 类似 row_index * cols + col_index
    # 行偏移量 * 列数 + 列偏移量 = 行偏移 + 列偏移 = 偏移
    old_offs = offs_row[:, None] * cols + offs_col[None, :]

    # 取出旧Block
    block = tl.load(input_ptr + old_offs)

    # 使用tl.trans进行转置
    transposed_block = tl.trans(block)

    # 类似 col_index * rows + row_index
    new_block = offs_col[:, None] * rows + offs_row[None, :]

    # 存储转置后的 transposed_block
    tl.store(output_ptr + new_block, transposed_block)


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

目前程序还未得到加速，均是`21.6ms`，效果马上来。提升迅猛。

## 5、通过mask控制元素访问

我们把块增大到32，和向量加类似，不过这里是2个维度

```Python
    # 旧 block 内行偏移量
    offs_row = row_index * 32 + tl.arange(0, 32)
    # 旧 block 内列偏移量
    offs_col = col_index * 32 + tl.arange(0, 32)
```

mask和rows和cols比，和向量加算子类似，不过这里需要带上维度扩展。输出可以直接使用输入mask的转置。代码如下所示

```Python
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)
    # 取出旧Block
    block = tl.load(input_ptr + old_offs, mask=mask)
...
    # 存储转置后的 transposed_block，可以直接使用转置的mask
    tl.store(output_ptr + new_block, transposed_block, mask=mask.T)
```

完整代码如下所示

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
    # 旧 block 内行偏移量
    offs_row = row_index * 32 + tl.arange(0, 32)
    # 旧 block 内列偏移量
    offs_col = col_index * 32 + tl.arange(0, 32)

    # 类似 row_index * cols + col_index
    # 行偏移量 * 列数 + 列偏移量 = 行偏移 + 列偏移 = 偏移
    old_offs = offs_row[:, None] * cols + offs_col[None, :]

    # 进行mask
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)

    # 取出旧Block
    block = tl.load(input_ptr + old_offs, mask=mask)

    # 使用tl.trans进行转置
    transposed_block = tl.trans(block)

    # 类似 col_index * rows + row_index
    new_block = offs_col[:, None] * rows + offs_row[None, :]

    # 存储转置后的 transposed_block，可以直接使用转置的mask
    tl.store(output_ptr + new_block, transposed_block, mask=mask.T)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
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

## 6、性能对比

我们的代码来到了`0.087ms`，之前还是`21.66ms`，加速比为**249x**。`CUDA`中最快的为`0.0678ms`，这个解答是公开的，我现在运行是`0.0827ms`，Pytorch 原生实现是`0.21 ms`。
