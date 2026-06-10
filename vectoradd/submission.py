# EVOLVE-BLOCK-START
"""
Float16 vector addition — minimal Python overhead baseline.
Pre-allocates output buffer per size. Hot path: single torch.add call with
pre-allocated out= and no conditional branches after warmup.
Uses a dispatch dict keyed by numel so the hot path is a single dict lookup
and a pre-bound function call with no attribute traversal.
"""

import torch

_cached_out = {}
_torch_add = torch.add

def custom_kernel(data):
    a, b = data
    n = a.numel()
    out = _cached_out.get(n)
    if out is None:
        out = torch.empty_like(a)
        _cached_out[n] = out
    _torch_add(a, b, out=out)
    return out
# EVOLVE-BLOCK-END
