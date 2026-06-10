# EVOLVE-BLOCK-START
"""
Float16 vector addition using plain torch.add — backed by highly optimized
CUDA elementwise kernels with vectorized 128-bit loads on H100.
"""

import torch


def custom_kernel(data):
    a, b = data
    return torch.add(a, b)
# EVOLVE-BLOCK-END
