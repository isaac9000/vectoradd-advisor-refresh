# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add with CUDA graph caching to eliminate
kernel launch overhead on repeated calls with the same tensor shape.
"""

import torch

_graph_cache = {}


def custom_kernel(data):
    a, b = data
    key = (a.shape, a.dtype, a.device)

    if key not in _graph_cache:
        # Allocate static input/output buffers for graph capture
        a_static = a.clone()
        b_static = b.clone()
        c_static = torch.empty_like(a)

        # Warm up
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                torch.add(a_static, b_static, out=c_static)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            torch.add(a_static, b_static, out=c_static)

        _graph_cache[key] = (g, a_static, b_static, c_static)

    g, a_static, b_static, c_static = _graph_cache[key]
    a_static.copy_(a)
    b_static.copy_(b)
    g.replay()
    return c_static.clone()
# EVOLVE-BLOCK-END
