# EVOLVE-BLOCK-START
"""
Float16 vector addition calling torch.ops.aten.add.out directly to bypass
the torch.add Python wrapper dispatch layer, combined with pre-allocated
output buffer cache to eliminate per-call allocation overhead.
"""

import torch

_out_cache = {}
_aten_add_out = torch.ops.aten.add.out


def custom_kernel(data):
    a, b = data
    key = (a.shape, a.dtype, a.device)
    if key not in _out_cache:
        _out_cache[key] = torch.empty_like(a)
    c = _out_cache[key]
    _aten_add_out(a, b, alpha=1, out=c)
    return c
# EVOLVE-BLOCK-END
