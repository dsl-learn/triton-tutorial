import torch
from matrix_transpose_kernel import solve

if __name__ == "__main__":
    rows, cols = 1921, 3224
    a = torch.randn((rows, cols), device='cuda')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='cuda')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
