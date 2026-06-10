# EVOLVE-BLOCK-START
"""
Float16 vector addition using CUDA Graphs to eliminate CPU dispatch overhead.
Lazily captures a torch.add graph per unique tensor size, then replays on hot path.
Static input/output buffers are pre-allocated per size; data is copied in before replay.
"""

import torch

# Per-size graph cache: size -> (graph, static_a, static_b, static_out)
_graph_cache = {}

def custom_kernel(data):
    a, b = data
    n = a.numel()
    shape = a.shape

    if n not in _graph_cache:
        # Allocate static buffers
        static_a = torch.empty_like(a)
        static_b = torch.empty_like(b)
        static_out = torch.empty_like(a)

        # Warm up before capture
        for _ in range(3):
            torch.add(static_a, static_b, out=static_out)
        torch.cuda.synchronize()

        # Capture the graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.add(static_a, static_b, out=static_out)

        _graph_cache[n] = (g, static_a, static_b, static_out)

    g, static_a, static_b, static_out = _graph_cache[n]
    static_a.copy_(a)
    static_b.copy_(b)
    g.replay()
    return static_out
# EVOLVE-BLOCK-END
