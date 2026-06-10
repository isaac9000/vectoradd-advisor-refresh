# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add with a pre-allocated output buffer.
Cache key uses numel() (single int) instead of tuple to minimize per-call
Python overhead. Hot path: integer comparison + pre-resolved tensor reference.
"""

import torch

_cached_numel = -1
_cached_out = None
_torch_add = torch.add


def custom_kernel(data):
    global _cached_numel, _cached_out
    a, b = data
    n = a.numel()
    if n != _cached_numel:
        _cached_out = torch.empty_like(a)
        _cached_numel = n
    _torch_add(a, b, out=_cached_out)
    return _cached_out
# EVOLVE-BLOCK-END
