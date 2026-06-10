# EVOLVE-BLOCK-START
"""
Float16 vector addition using an inline CUDA kernel with 128-bit vectorized
loads (uint4 / 8 float16 per transaction) to maximize HBM3 bandwidth on H100.
Compiled once at module load time to minimize per-call overhead.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// Each thread processes ITEMS_PER_THREAD consecutive uint4 chunks (8 fp16 each)
// Block size = 512 for high SM occupancy on H100
#define BLOCK_SIZE 512
#define ITEMS_PER_THREAD 4

__global__ void vecadd_fp16_vec8(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    uint4* __restrict__ c,
    int n_vec)
{
    int base = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ITEMS_PER_THREAD;
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        int idx = base + k;
        if (idx < n_vec) {
            uint4 va = a[idx];
            uint4 vb = b[idx];
            half* ha = (half*)&va;
            half* hb = (half*)&vb;
            uint4 vc;
            half* hc = (half*)&vc;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                hc[i] = __hadd(ha[i], hb[i]);
            }
            c[idx] = vc;
        }
    }
}

torch::Tensor vecadd_fp16(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n_elements = a.numel();
    int n_vec = n_elements / 8;
    int remainder = n_elements % 8;

    if (n_vec > 0) {
        const int block = BLOCK_SIZE;
        const int grid = (n_vec + block * ITEMS_PER_THREAD - 1) / (block * ITEMS_PER_THREAD);
        vecadd_fp16_vec8<<<grid, block>>>(
            reinterpret_cast<const uint4*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const uint4*>(b.data_ptr<at::Half>()),
            reinterpret_cast<uint4*>(c.data_ptr<at::Half>()),
            n_vec
        );
    }
    // Handle remainder elements (scalar fallback)
    if (remainder > 0) {
        auto a_tail = a.view(-1).slice(0, n_vec * 8);
        auto b_tail = b.view(-1).slice(0, n_vec * 8);
        auto c_tail = c.view(-1).slice(0, n_vec * 8);
        c_tail.copy_(a_tail + b_tail);
    }
    return c;
}
"""

_cpp_src = r"""
torch::Tensor vecadd_fp16(torch::Tensor a, torch::Tensor b);
"""

_ext = load_inline(
    name="vecadd_fp16_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vecadd_fp16"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def custom_kernel(data):
    a, b = data
    return _ext.vecadd_fp16(a.contiguous(), b.contiguous())
# EVOLVE-BLOCK-END
