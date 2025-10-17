**TL;DR**: The following code walks through a step-by-step implementation of vector addition and uses Triton primitives such as `tl.arange`、`tl.load`、`tl.store`、`tl.program_id` and `tl.constexpr`.

## Hands-on Triton kernel implementation: Vector Addition

Vector addition is the "hello world" program for Triton, can be written as:

$$
c = a + b
$$

$a$, $b$ and $c$ are all 1D tensors (vectors) of the same shape.

### 1. PyTorch's built-in vector addition

The built-in vector addition of PyTorch can be done simply using `+`, because it overrides the `+`operator.

We’ll start with a tensor of shape torch.Size([16]).
(Why not `15` or `17`? Is there a reason for choosing `16`? The short answer is yes—we’ll find out why soon.)

```Python
import torch
import random
random.seed(42)

N = 16 
a = torch.randn(N, device='cuda')
b = torch.randn(N, device='cuda')
c = a + b
print(a, b, c, sep="\n")
```

Notice that the $i$-th element in $c$ equals to the sum of the $i$-th element in $a$ and the $i$-th element in $b$

```c
tensor([-0.3947,  0.1963,  0.4782, -0.0215,  1.5055,  0.1066, -0.8224,  0.0999,
        -0.1316,  0.3244, -1.6962, -0.1411,  0.5005,  0.0396,  0.4774,  0.9639],
       device='cuda:0')
tensor([-0.1621, -1.0437,  0.5023,  0.3897,  0.6714, -0.8212, -0.2596, -0.3467,
        -2.2264,  0.7489,  1.3961, -2.1076,  0.0119, -0.8835, -0.4079,  1.8599],
       device='cuda:0')
tensor([-0.5568, -0.8474,  0.9805,  0.3682,  2.1769, -0.7145, -1.0820, -0.2469,
        -2.3581,  1.0732, -0.3000, -2.2486,  0.5124, -0.8439,  0.0695,  2.8238],
       device='cuda:0')
```

The built-in vector addition in Pytorch calls the `vectorized_elementwise_kernel` CUDA kernel defined here: [aten/src/ATen/native/cuda/CUDALoops.cuh:L334](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cuda/CUDALoops.cuh#L334).

### 2. Triton 1D tensor addition in one Thread Block (program instance)

**Thread Block and its max tensor numel( )**

Unlike PyTorch, which can handle inputs of various shapes and sizes automatically, working at the lower level is non-trivial—our hardware cannot process arbitrarily large inputs. In Triton, you need to look more closely at what happens under the hood and exercise fine-grained control over memory allocation.

That said, large input tensors must be handled differently and in a more complex way. It turns out that Triton prevents users from creating too many elements in a Thread Block, not because of a hardware limit, but as a (conservative) safeguard to avoid memory/complexity blowing up during compile-time. To start simply, we can use a smaller input size—`16` seems like a good choice within the capacity of one single Thread Block.

More details of [Grid Bloack](https://modal.com/gpu-glossary/device-software/thread-block-grid), [Thread Block](https://modal.com/gpu-glossary/device-software/thread-block), [Warp](https://modal.com/gpu-glossary/device-software/warp) and [Thread](https://modal.com/gpu-glossary/device-software/thread).

**Thread Block and Program Instance**

In Triton, a program instance represents the [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)-level “instruction stream” (the “I”), and its lanes are the data lanes. In the context of NVIDIA GPUs,  executes those lanes using [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) threads, making the CUDA thread block the physical carrier of the “T”.

Program instance $\approx$ Thread Block, they are the same thing but from different perspectives.

**Decorator**

Triton kernel code must be wrapped in the [decorator](https://www.geeksforgeeks.org/system-design/decorator-pattern/) `@triton.jit` at its declaration. This decorator labels the function as a Triton kernel, allowing it to be Just-In-Time (JIT) compiled and launched as a specialized GPU program.

Noted that we use pointers in Triton instead of the Pythonic reference when passing tensor parameters to a function. That's mainly because a Triton kernel is compiled to run on the GPU, and the GPU hardware operates on memory addresses (pointers) to access data efficiently in global memory. The pointers provide the necessary low-level control for the compiler to generate high-performance, hardware-aware code.

In case you want to ask: How does the reference become a pointer?

```Python
print(a.data_ptr())
```

gives

```c
140081249648640
```

Okay now lets get back to the implementation

```Python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    pass

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,) # `1` denotes the number of blocks we used, for a 16-element vector, 1 is more than enough
    vector_add_kernel[grid](a, b, c)
```

As aforementioned, we focus on `N=16`, therefore, we need to pair the values in the same position of two vectors and perform element-wise addition. We can first use `tl.arrange()` to generate indexes for elements [0, 1, 2, ..., 15]. The address of each element then can be written as `a_ptr + offset`. We then perform the addtion and save the result by using `tl.store( )`.

There we have written the vector addition kernel in Python:

```Python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 16) # generates indexes [0, 1, ..., 15] for 16 elements 
    
    a = tl.load(a_ptr + offsets) # load values from memory with their indexes, variable `a` is a Triton tensor, more specifically, the datatype of `a` is tl.tensor(float32, (16,))
    b = tl.load(b_ptr + offsets)

    c = a + b # element wise addition
    
    tl.store(c_ptr + offsets, c) 

"""
In case you might be curious about why the `+` operator works for `tl.tensor` here, Triton actually overloads `+` to perform element-wise addition. Unlike normal Python code where an element-wise vector addition would typically involve a hidden Python for-loop (or loops), the implementation of Triton's overloaded operator performs the operation in a truly parallel and vectorized manner on the GPU.
"""
```

The first and foremost thing for the kernel is always its correctness. We can use `torch.allclose()` to compare our results with the official PyTorch implementation.

```Python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 16)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = a + b
    tl.store(c_ptr + offsets, c)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)
    vector_add_kernel[grid](a, b, c)

if __name__ == "__main__":
    N = 16
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b , triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

running the code snippet above gives output:

```c
✅ Triton and Torch match
```

### 3. Bounded element load/store with `mask`

**Callback 1**: What if the input tensor has 15 elements? We can simply change `N=16` to `N=15` to see what happens.

```Python
""" WARNING: FOLLOWING CODE SAMPLE DEMONSTRATES A WRONG PATTERN"""
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 16)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = a + b
    tl.store(c_ptr + offsets, c)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)
    vector_add_kernel[grid](a, b, c)

if __name__ == "__main__":
    N = 15 # <- the only line we edit
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b , triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

Surprisingly (or not), it gives the output:

```c
✅ Triton and Torch match
```

Actually, it works for any value of `N` in the range of [1, 16]. But this is EXTREMELY DANGEROUS because you are writing a part of the memory you are NOT supposed to touch, please never do so.

To avoid out-of-bounds memory load/store, the size of the vectors and the `tl.arrange()` need to be consistent. Persumably we can change `tl.arrange(0, 16)` to `tl.arrange(0, 15)` when we have `N=15`. Let's give it a try:

```Python
""" WARNING: FOLLOWING CODE SAMPLE DEMONSTRATES A WRONG PATTERN"""
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 15) # <- the line we edit
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = a + b
    tl.store(c_ptr + offsets, c)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)
    vector_add_kernel[grid](a, b, c)

if __name__ == "__main__":
    N = 15 # <- the line we edit
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    torch_output = a + b
    triton_output = torch.empty_like(a)
    solve(a, b , triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

The output is:

```c
 ---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.12/dist-packages/triton/language/core.py in wrapper(*args, **kwargs)
     41                              "(`_semantic` argument must be provided outside of JIT functions.)")
---> 42         return fn(*args, **kwargs)
     43 

7 frames/usr/local/lib/python3.12/dist-packages/triton/language/core.py in arange(start, end, _semantic)
   1653     end = _unwrap_if_constexpr(end)
-> 1654     return _semantic.arange(start, end)
   1655 

/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py in arange(self, start, end, ret_ty)
    582         if (range & (range - 1)) != 0:
--> 583             raise ValueError("arange's range must be a power of 2")
    584         shape = [range]

ValueError: arange's range must be a power of 2

The above exception was the direct cause of the following exception:

...

CompilationError: at 2:14:
def vector_add_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 15)
              ^
```

This error message indicates that the range of `tl.arange()` cannot be an arbitrary number, but must be a power of 2. That's because many GPU architectures and memory units prefer power-of-two vector lengths, so Triton enforces it for better and easier alignment.

Luckily, Triton provides a workaround for this: the `mask`. A `mask` is a Triton tensor filled with boolean values, which indicates which elements of the input vector are out-of-bounds. Here is an example:

```Python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N):
    offsets = tl.arange(0, 16)

    mask = offsets < N # size of your input vector

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (1,)
    vector_add_kernel[grid](a, b, c, N)

if __name__ == "__main__":
    for N in range(1, 16+1):
        a = torch.randn(N, device='cuda')
        b = torch.randn(N, device='cuda')
        torch_output = a + b
        triton_output = torch.empty_like(a)
        solve(a, b , triton_output, N)
        if torch.allclose(triton_output, torch_output):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
```

**TODO**: In the code snippet above, the variable `N` is a runtime variable; however, Triton kernels require JIT compilation. Please explain: why doing `mask = offsets < N` works fine, but `tl.arange(0, N)` causes an error even when `N` is a power of two? Is there a way to write `tl.arange(0, N)` without errors?

The current Triton kernel only works properly when the input size is ≤ 16. We want our kernel to handle inputs of any size, but we don’t want to create a kernel that uses an excessively large compile-time tile size, which would increase register usage and reduce occupancy. Nor do we want to define multiple kernels with different fixed tile sizes, compile them JIT, and select one at runtime — both approaches would waste valuable GPU resources. Instead, we want a templated GPU kernel that can adaptively and efficiently handle arbitrary input sizes.

### 4. Multiple Thread Blocks for Large-size Inputs

**Callback 2**: The official documentation of `tl.arrange()` says this function:

>Returns contiguous values within the half-open interval [start, end). end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = 1048576

`1048576` (2^20) the max number of elements within one single Thread Block, so multiple Thread Blocks are needed. In Triton, the program instance (often parameterized by BLOCK) is the finest granularity of scheduling and optimization.

**TODO**: In CUDA, you think in terms of threads and warps:

```c
int tid = threadIdx.x + blockIdx.x * blockDim.x;
float a = A[tid];
float b = B[tid];
C[tid] = a + b;
```

however, Triton let programmers to express the math and data layout at the tile level, and let the compiler to handle threads and warps. Please explain why warps and threads are hidden abstractions in Triton, and what benefits (and trade-offs) that gives you.

In NVIDIA GPU, the hierarchy of logical execution (from high level to low level) is:
> Grid -- Thread Block -- Warp -- Thread

Just like what we do with the offset in previous code blocks, we don't need to define the behavior of each block -- remember we have SIMT inside of a Thread Block? Similarily, we have SPMD (Single Program Multiple Data) here.

A grid is a collection of blocks, and it can be 1D, 2D, or 3D, it can be defined as `grid = ()`, e.g. `grid=(B, C, H, W)`. For vector operations, it’s sufficient to arrange the blocks only along the x dimension. Inside the kernel, we can use `tl.program_id(axis=0)` to obtain the block index.

Therefore, multiple blocks can be managed by Grid. Here's how:

```Python
import time
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N):
    
    pid = tl.program_id(axis=0) # pid is a unique ID for each Thread Block
    
    block_start = pid * 16 # slicing data for each block

    offsets = block_start + tl.arange(0, 16)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = (triton.cdiv(N, 16), ) 
    vector_add_kernel[grid](a, b, c, N)
    

def time_op_gpu(fn, sync=True, warmup=5, iters=20):
    """
    Time a GPU operation using CUDA events for better accuracy (no CPU scheduling noise).
    - fn: a callable that launches GPU work
    - sync: whether to synchronize after each iteration (True recommended)
    - warmup: warm-up iterations to let JIT/caches settle
    - iters: timed iterations

    Returns: average time in milliseconds over 'iters' runs.
    """
    # warm-up does JIT and warms caches
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        # Wait for the events to be recorded & measure GPU time
        torch.cuda.synchronize()
        elapsed_ms += start.elapsed_time(end)
    return elapsed_ms / iters
    
if __name__ == "__main__":
    for power in range(1, 25 ,2):
        N = 2 ** power
        N = 1 << 24
        a = torch.randn(N, device='cuda')
        b = torch.randn(N, device='cuda')
        triton_output = torch.empty_like(a)
        
        def torch_op():
            return a + b
        
        def triton_op():
            triton_output = torch.empty_like(a)
            solve(a, b, triton_output, N)
            return triton_output
        
        torch_output = torch_op()  # warm-up
        torch_time_elapsed = time_op_gpu(torch_op)
        
        triton_output = triton_op()  # warm-up
        triton_time_elapsed = time_op_gpu(triton_op)

        if torch.allclose(triton_output, torch_output):
            print(f"✅ Triton and Torch match with input size 2^{power}")
            print(f"Torch  time: {torch_time_elapsed:.5f} ms, \nTriton time: {triton_time_elapsed:.5f} ms")
        else:
            print(f"❌ Triton and Torch differ with input size 2^{power}")
            
        print("grid size: ", triton.cdiv(N, 16), "\n")
```

Great, it passes all the tests! But looking at the printed output:

```c
✅ Triton and Torch match with input size 2^1
Torch  time: 1.33550 ms, 
Triton time: 4.31417 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^3
Torch  time: 1.61238 ms, 
Triton time: 3.68864 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^5
Torch  time: 1.14208 ms, 
Triton time: 3.15389 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^7
Torch  time: 1.25037 ms, 
Triton time: 2.98867 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^9
Torch  time: 1.57355 ms, 
Triton time: 2.96891 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^11
Torch  time: 1.12892 ms, 
Triton time: 2.86296 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^13
Torch  time: 1.80937 ms, 
Triton time: 3.51826 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^15
Torch  time: 1.25404 ms, 
Triton time: 2.88556 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^17
Torch  time: 1.35573 ms, 
Triton time: 2.77837 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^19
Torch  time: 1.52062 ms, 
Triton time: 2.86905 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^21
Torch  time: 1.27100 ms, 
Triton time: 2.63440 ms
grid size:  1048576 

✅ Triton and Torch match with input size 2^23
Torch  time: 1.25251 ms, 
Triton time: 3.44298 ms
grid size:  1048576 
```

Well we observe that: Triton kernel is SLOWER than the PyTorch Implementation. Why is that?

Short answer: the Triton kernel is **under-utilizing** the GPU and paying proportionally higher fixed overhead than PyTorch’s highly-tuned kernel.

Here is a detailed breakdown:

1. We launch `N` blocks, while each of them does only 16 additions, so the prologue/epilogue overhead per block dominates the time complexity.
2. PyTorch's vector addition, even as a fallback, it is not bad at all. It is actually heavily optimized ($O(1)$ launch complexity) especially for small input size with minimum launch overhead, while Triton is more time-consuming to launch and compile.
3. By default, 4 warps (4 * 32 = 128 threads) will be assigned to a block at launch, even if you don't use them, registers are still allocated and reserved for them, leads to a low occupancy.

An intuitive solution to this is: we want a number larger than `16` in the line:

```python
offsets = block_start + tl.arange(0, 16)
```

or, in a even better way, we want to write:

```python
offsets = block_start + tl.arrange(0, BLOCK_SIZE)
```

where `BLOCK_SIZE` is a runtime constant. There we have the final version of our vector addition kernel:

```python
import time
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    
    pid = tl.program_id(axis=0) # pid is a unique ID for each Thread Block
    
    # block_start = pid * 16 # slicing data for each block
    # offsets = block_start + tl.arange(0, 16)
    
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK), ) 
    vector_add_kernel[grid](a, b, c, N, BLOCK=BLOCK, num_warps=4)


def time_op_gpu(fn, sync=True, warmup=5, iters=20):
    """
    Time a GPU operation using CUDA events for better accuracy (no CPU scheduling noise).
    - fn: a callable that launches GPU work
    - sync: whether to synchronize after each iteration (True recommended)
    - warmup: warm-up iterations to let JIT/caches settle
    - iters: timed iterations

    Returns: average time in milliseconds over 'iters' runs.
    """
    # warm-up does JIT and warms caches
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        # Wait for the events to be recorded & measure GPU time
        torch.cuda.synchronize()
        elapsed_ms += start.elapsed_time(end)
    return elapsed_ms / iters
    
if __name__ == "__main__":
    for power in range(1, 20, 2):
        N = 2 ** power
        a = torch.randn(N, device='cuda')
        b = torch.randn(N, device='cuda')
        torch_output  = torch.empty_like(a)
        triton_output = torch.empty_like(a)
        
        def torch_op():
            return torch_output.copy_(a + b)
        
        def triton_op():
            triton_output = torch.empty_like(a)
            solve(a, b, triton_output, N)
            return triton_output
        
        torch_output = torch_op()  # warm-up
        torch_time_elapsed = time_op_gpu(torch_op)
        
        triton_output = triton_op()  # warm-up
        triton_time_elapsed = time_op_gpu(triton_op)

        if torch.allclose(triton_output, torch_output):
            print(f"✅ Triton and Torch match with input size 2^{power}")
            print(f"Torch  time: {torch_time_elapsed:.5f} ms, \nTriton time: {triton_time_elapsed:.5f} ms")
        else:
            print(f"❌ Triton and Torch differ with input size 2^{power}")
            
        print("grid size: ", triton.cdiv(N, 16), "\n")
```

with output:

```c
✅ Triton and Torch match with input size 2^1
Torch  time: 0.17645 ms, 
Triton time: 0.09037 ms
grid size:  1 

✅ Triton and Torch match with input size 2^3
Torch  time: 0.04901 ms, 
Triton time: 0.18940 ms
grid size:  1 

✅ Triton and Torch match with input size 2^5
Torch  time: 0.31917 ms, 
Triton time: 0.07534 ms
grid size:  2 

✅ Triton and Torch match with input size 2^7
Torch  time: 0.34054 ms, 
Triton time: 0.06917 ms
grid size:  8 

✅ Triton and Torch match with input size 2^9
Torch  time: 0.04664 ms, 
Triton time: 0.19412 ms
grid size:  32 

✅ Triton and Torch match with input size 2^11
Torch  time: 0.07051 ms, 
Triton time: 0.06189 ms
grid size:  128 

✅ Triton and Torch match with input size 2^13
Torch  time: 0.23926 ms, 
Triton time: 0.08479 ms
grid size:  512 

✅ Triton and Torch match with input size 2^15
Torch  time: 0.07361 ms, 
Triton time: 0.07344 ms
grid size:  2048 

✅ Triton and Torch match with input size 2^17
Torch  time: 0.05223 ms, 
Triton time: 0.06151 ms
grid size:  8192 

✅ Triton and Torch match with input size 2^19
Torch  time: 0.25605 ms, 
Triton time: 0.08298 ms
grid size:  32768 

✅ Triton and Torch match with input size 2^21
Torch  time: 0.76483 ms, 
Triton time: 0.46037 ms
grid size:  131072 

✅ Triton and Torch match with input size 2^23
Torch  time: 1.53014 ms, 
Triton time: 0.63391 ms
grid size:  524288 
```

Aha! Triton is FASTER, when we have large input sizes!

**TODO**:

1. Fixed `N`, Varying `BLOCK_SIZE`: With a fixed input size `N`, how does changing the `BLOCK_SIZE` affect the kernel's efficiency, and what is the optimal `BLOCK_SIZE` for the target hardware?
2. Fixed `BLOCK_SIZE`, Varying `N`: With a fixed `BLOCK_SIZE`, how does changing the input size `N` affect the kernel's efficiency, and what is the optimal `N` that maximizes efficiency for the target hardware?
3. Do your observations and conclusions from questions 1 and 2 vary across different GPU architectures or hardware?

[LeetGPU](https://leetgpu.com) is the leetcode for GPU kernel programming, it offers test cases and different hardwares for you to evaluate the code you write. Check this link out: [Vector Addition](https://leetgpu.com/challenges/vector-addition), and submit your code to see if it passes the test.

![submit code to LeetGPU - Vector Addition](https://img2024.cnblogs.com/blog/1154439/202508/1154439-20250831151106589-1307418925.png)

### 6. Source Code

Source code of this markdown can be found in `./ex1-vector_add-en/vector_add.ipynb`

## Beyond Vector Addition

More practice:
[Matrix Copy](https://leetgpu.com/challenges/matrix-copy) 
[Color Inversion](https://leetgpu.com/challenges/color-inversion) 
[Reverse Array](https://leetgpu.com/challenges/reverse-array) 
[Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) 
[ReLU](https://leetgpu.com/challenges/relu) 
[Leaky ReLU](https://leetgpu.com/challenges/leaky-relu)