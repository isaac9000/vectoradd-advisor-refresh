# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.compile with mode='reduce-overhead' to
minimize Python dispatch overhead on repeated calls with the same shapes.
"""

import torch


def _add(a, b):
    return torch.add(a, b)


_compiled_add = torch.compile(_add, mode='reduce-overhead')


def custom_kernel(data):
    a, b = data
    return _compiled_add(a, b)
# EVOLVE-BLOCK-END
