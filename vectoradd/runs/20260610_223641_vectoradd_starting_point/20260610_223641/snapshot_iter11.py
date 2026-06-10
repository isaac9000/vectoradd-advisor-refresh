# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add with a pre-allocated output buffer
cached per shape/dtype/device to eliminate per-call torch.empty_like overhead.
"""

import torch

_out_cache = {}


def custom_kernel(data):
    a, b = data
    key = (a.shape, a.dtype, a.device)
    if key not in _out_cache:
        _out_cache[key] = torch.empty_like(a)
    c = _out_cache[key]
    torch.add(a, b, out=c)
    return c
# EVOLVE-BLOCK-END
