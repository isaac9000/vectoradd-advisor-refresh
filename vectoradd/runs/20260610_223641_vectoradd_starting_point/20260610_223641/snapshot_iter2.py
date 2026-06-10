# EVOLVE-BLOCK-START
"""
Float16 vector addition with autotuned Triton kernel using large block sizes
to maximize HBM3 bandwidth utilization on H100.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
        triton.Config({'BLOCK_SIZE': 16384}),
        triton.Config({'BLOCK_SIZE': 32768}),
        triton.Config({'BLOCK_SIZE': 65536}),
    ],
    key=['n_elements'],
)
@triton.jit
def vecadd_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)


def custom_kernel(data):
    a, b = data
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty_like(a)
    n_elements = a.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vecadd_kernel[grid](a, b, c, n_elements)
    return c
# EVOLVE-BLOCK-END
