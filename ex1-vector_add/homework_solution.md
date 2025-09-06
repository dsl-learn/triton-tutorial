## 练习解答

练习中的前3道题还是比较简单的，如果没搞定的话我们一起动手做一下。

### 1、[Matrix Copy](https://leetgpu.com/challenges/matrix-copy)

在 GPU 上实现一个程序，将输入的 `N×N` 浮点矩阵 `A` 元素逐一复制到输出矩阵 `B`，即满足`A[i][j]=B[i][j]`。

其实就是`N×N`的元素拷贝到另外一个地方，需要你计算出`a+b`的总长度就可以了。以下是参考答案

```Python
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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

### 2、[Color Inversion](https://leetgpu.com/challenges/color-inversion)

颜色反转操作：将每个颜色分量`(R, G, B)` 用 `255−分量值` 计算得到，`Alpha` 分量保持不变。

存储的时候用的是RGBA，也就是4个变量是一组的，最后一个不处理。可以转换为索引为3的不处理，`offsets % 4 != 3` 作为mask即可。另外`width * height`也可以直接合并起来。还有需要注意的是你每个`program`分配的任务数，1个像素`pixel`是要处理4个的。

```Python
import torch
import triton
import triton.language as tl

# 对图像进行颜色反转（R、G、B 取 255 - 分量值，A 保持不变）
@triton.jit
def invert_kernel(image_ptr, n_pixels, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)

    # 计算当前 block 的起始位置（像素按 RGBA 存储，每个像素 4 个通道）
    block_start = pid * BLOCK_SIZE * 4

    # 生成当前 block 内的所有元素索引（BLOCK_SIZE 个像素 * 4 通道）
    offsets = block_start + tl.arange(0, BLOCK_SIZE * 4)

    # mask 条件：
    # 1. offsets < n_pixels*4   → 保证不越界
    # 2. offsets % 4 != 3           → 跳过 Alpha 通道（不做反转）
    mask = (offsets < n_pixels * 4) & (offsets % 4 != 3)

    # 从输入图像中加载数据（只加载有效的 R、G、B 通道）
    input = tl.load(image_ptr + offsets, mask=mask)

    # 颜色反转：output = 255 - input
    output = 255 - input

    # 将反转后的结果写回图像（只写有效的 R、G、B 通道）
    tl.store(image_ptr + offsets, output, mask=mask)


def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE), )
    invert_kernel[grid](
        image,
        n_pixels,
        BLOCK_SIZE
    )

if __name__ == "__main__":
    width, height = 1024, 768
    img = torch.randint(0, 256, (width, height, 4), dtype=torch.uint8, device='cuda')
    # 拷贝一份
    torch_output = img.clone()
    # 对 R、G、B 通道执行 255 - 值
    torch_output[..., :3] = 255 - torch_output[..., :3]
    triton_output = img.clone()
    solve(triton_output, width, height)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

### 3、[Reverse Array](https://leetgpu.com/challenges/reverse-array)

反转数组是对`offsets`计算的巩固，实际上是`A[i] = A[N-1-i]`，在Triton我们操作的是`offsets`的连续索引。

不考虑mask的情况下可以简化`A[i+offsets] = B[N-i-offsets]`。在前后交换中，我们需要在1个program内拿到前和后两块，所以grid数目会砍半，如果数组长度为奇数，中间那个正好不用处理。以下是算子代码

```Python
import torch
import triton
import triton.language as tl

# 反转数组 即 input[i] ↔ input[N-1-i]
@triton.jit
def reverse_kernel(input_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的索引（第 0 维）
    pid = tl.program_id(axis=0)

    # 计算当前 block 内元素的索引
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 只需要处理数组前半部分，因为每次交换会处理前后两个位置
    mask = offsets < (N // 2)

    # 计算需要交换的前后位置索引
    front_pos = offsets
    back_pos = N - 1 - offsets

    # 从数组中加载这两个位置的值（只加载有效索引）
    front_val = tl.load(input_ptr + front_pos, mask=mask)
    back_val = tl.load(input_ptr + back_pos, mask=mask)

    # 交换前后两个值
    tl.store(input_ptr + front_pos, back_val, mask=mask)  # 前位置写回后值
    tl.store(input_ptr + back_pos, front_val, mask=mask)  # 后位置写回前值


def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE), )
    reverse_kernel[grid](input, N, BLOCK_SIZE)


if __name__ == "__main__":
    N = 21121
    a = torch.randn(N, device='cuda')
    torch_output = a.flip(0)
    triton_output = a.clone()
    solve(triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```