import torch
import triton
import triton.language as tl

# 矩阵转置 即 output[j][i]=input[i][j]
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 在 grid 中的索引（第 0 维）, 即 row_index
    row_index = tl.program_id(axis=0)
    # 获取当前 program 在 grid 中的索引（第 1 维）, 即 col_index
    col_index = tl.program_id(axis=1)
    # 旧 block 内行偏移量
    offs_row = row_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 旧 block 内列偏移量
    offs_col = col_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

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
        rows, cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
