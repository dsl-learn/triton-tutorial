import torch
from vector_add_kernel import solve

if __name__ == "__main__":
    N = 12345
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b , triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
