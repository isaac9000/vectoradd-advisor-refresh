# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-10 22:48:58 UTC ✅ KEEP

**Hypothesis:** Baseline 'previous_best' — initial benchmark

**Result:** 32.00 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add with a pre-allocated output buffer.
After first warmup call, skips all cache/numel checks — hot path is a single
pre-bound torch.add call with pre-allocated out tensor.
"""

import torch

_cached_out = None
_warmed_up = False
_torch_add = torch.add


def custom_kernel(data):
    global _cached_out, _warmed_up
    a, b = data
    if not _warmed_up:
        _cached_out = torch.empty_like(a)
        _warmed_up = True
    _torch_add(a, b, out=_cached_out)
    return _cached_out
# EVOLVE-BLOCK-END

```

---

## Experiment #2 — 2026-06-10 22:49:34 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed a Triton kernel with 128-bit vectorized (8×fp16) loads/stores, 1D flat grid, BLOCK_SIZE=1024, pre-allocated output buffer reused across calls. Implemented exactly as proposed: flatten 2D to 1D, Triton jit kernel with tl.load/store, cdiv grid sizing.

**Result:** 54.25 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using a Triton kernel with 128-bit vectorized loads/stores.
Flattens 2D tensor to 1D and processes in 8xfp16 (128-bit) chunks.
Pre-allocates output buffer and reuses across calls.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _vadd_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

_cached_out = None
_cached_n = None

def custom_kernel(data):
    global _cached_out, _cached_n
    a, b = data
    n = a.numel()
    if _cached_out is None or _cached_n != n:
        _cached_out = torch.empty_like(a)
        _cached_n = n
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _vadd_kernel[grid](a, b, _cached_out, n, BLOCK_SIZE=BLOCK_SIZE)
    return _cached_out
# EVOLVE-BLOCK-END

```

---

## Experiment #3 — 2026-06-10 22:50:19 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed: (1) explicit JIT warmup at module load time via dummy tensor call + torch.cuda.synchronize(), and (2) larger BLOCK_SIZE=4096 to reduce grid overhead and maximize memory transaction efficiency. Implemented both changes exactly as proposed. Result is essentially unchanged from experiment #2 (54.25→54.81 µs), confirming JIT warmup was not the bottleneck — the Triton kernel itself is slower than torch.add for this workload.

**Result:** 54.81 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using a Triton kernel with large block sizes and explicit warmup.
Flattens 2D tensor to 1D, uses BLOCK_SIZE=4096 to maximize memory transaction efficiency,
and pre-compiles the kernel via a dummy warmup call at module load time.
Pre-allocates output buffer and reuses across calls.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _vadd_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# Warmup: trigger JIT compilation at import time so it doesn't bleed into benchmarking
_BLOCK_SIZE = 4096
_dummy = torch.zeros(1, dtype=torch.float16, device='cuda')
_vadd_kernel[(1,)](_dummy, _dummy, _dummy, 1, BLOCK_SIZE=_BLOCK_SIZE)
torch.cuda.synchronize()
del _dummy

_cached_out = None
_cached_n = None

def custom_kernel(data):
    global _cached_out, _cached_n
    a, b = data
    n = a.numel()
    if _cached_out is None or _cached_n != n:
        _cached_out = torch.empty_like(a)
        _cached_n = n
    grid = (triton.cdiv(n, _BLOCK_SIZE),)
    _vadd_kernel[grid](a, b, _cached_out, n, BLOCK_SIZE=_BLOCK_SIZE)
    return _cached_out
# EVOLVE-BLOCK-END

```

---

## Experiment #4 — 2026-06-10 22:51:06 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed CUDA Graphs: lazily capture torch.add per unique tensor size, replay on hot path with static input/output buffers. Implemented exactly as proposed with per-size graph cache, 3-call warmup before capture, and copy_() to populate static buffers before each replay. Result is much worse (73.3 µs vs 32.0 µs baseline) because the copy_() calls for large tensors (N=4096: 99 µs, N=8192: 329 µs) dominate — copying data into static buffers costs as much as or more than the original torch.add call itself.

**Result:** 73.30 μs

**Kernel code:**
```python
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

```

---

## Experiment #5 — 2026-06-10 22:52:15 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed inline CUDA kernel with half2 vectorized loads (__ldg), strided grid pattern, compiled at module load time. Implemented with half2 kernel using __ldg + __hadd2, float4 not used (half2 stride loop instead), 256 threads/block, 528 blocks fixed (132 SMs * 4). Small sizes improved dramatically (N=1024: 11.9µs, N=2048: 20.1µs) but large sizes are much worse (N=4096: 56.5µs, N=8192: 189.8µs) — the fixed 528-block grid caps throughput for large tensors that need more blocks to cover all elements in the strided loop. Geomean 40.0µs, worse than 32µs baseline.

**Result:** 39.99 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using an inline CUDA kernel with half2 vectorized loads/stores.
Uses float4 casts for 128-bit transactions, __ldg() read-only cache for inputs,
strided grid pattern. Compiled once at module load time. Pre-allocated output per size.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void vadd_half2_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int n_half2)
{
    // Process 2 fp16 values (1 half2) per thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* b2 = reinterpret_cast<const __half2*>(b);
    __half2* c2 = reinterpret_cast<__half2*>(c);
    for (int i = idx; i < n_half2; i += stride) {
        __half2 va = __ldg(&a2[i]);
        __half2 vb = __ldg(&b2[i]);
        c2[i] = __hadd2(va, vb);
    }
}

void vadd_half2(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c)
{
    int n = a.numel();
    int n_half2 = n / 2;
    const int threads = 256;
    // 132 SMs * 4 blocks per SM = 528 blocks to saturate H100
    const int blocks = min(528, (n_half2 + threads - 1) / threads);
    vadd_half2_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        n_half2
    );
}
"""

_cpp_src = r"""
void vadd_half2(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);
"""

_ext = load_inline(
    name="vadd_half2_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vadd_half2"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

_cached_out = {}

def custom_kernel(data):
    a, b = data
    n = a.numel()
    if n not in _cached_out:
        _cached_out[n] = torch.empty_like(a)
    out = _cached_out[n]
    _ext.vadd_half2(a, b, out)
    return out
# EVOLVE-BLOCK-END

```

---

## Experiment #6 — 2026-06-10 22:53:11 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed removing the artificial 528-block cap and scaling blocks with tensor size using cdiv(n_half2, threads). Made exactly that one change: removed min(528, ...) and replaced with a direct cdiv. Large-size performance recovered substantially (N=4096: 56.5→47.8µs, N=8192: 189.8→156.2µs) and small sizes remain excellent (N=1024: 12.0µs, N=2048: 17.6µs). Geomean improved from 40.0µs to 35.5µs but still above the 32.0µs baseline.

**Result:** 35.46 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using an inline CUDA kernel with half2 vectorized loads/stores.
Uses float4 casts for 128-bit transactions, __ldg() read-only cache for inputs,
strided grid pattern. Compiled once at module load time. Pre-allocated output per size.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void vadd_half2_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int n_half2)
{
    // Process 2 fp16 values (1 half2) per thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* b2 = reinterpret_cast<const __half2*>(b);
    __half2* c2 = reinterpret_cast<__half2*>(c);
    for (int i = idx; i < n_half2; i += stride) {
        __half2 va = __ldg(&a2[i]);
        __half2 vb = __ldg(&b2[i]);
        c2[i] = __hadd2(va, vb);
    }
}

void vadd_half2(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c)
{
    int n = a.numel();
    int n_half2 = n / 2;
    const int threads = 256;
    // Scale blocks with tensor size — no artificial cap
    const int blocks = (n_half2 + threads - 1) / threads;
    vadd_half2_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        n_half2
    );
}
"""

_cpp_src = r"""
void vadd_half2(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);
"""

_ext = load_inline(
    name="vadd_half2_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vadd_half2"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

_cached_out = {}

def custom_kernel(data):
    a, b = data
    n = a.numel()
    if n not in _cached_out:
        _cached_out[n] = torch.empty_like(a)
    out = _cached_out[n]
    _ext.vadd_half2(a, b, out)
    return out
# EVOLVE-BLOCK-END

```

---

## Experiment #7 — 2026-06-10 22:54:23 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed upgrading to float4 (128-bit) loads per thread processing 8 fp16 values using 4x __hadd2 operations, with grid scaled by n/8. Implemented exactly: float4 kernel loading 128-bit chunks via __ldg, reinterpreting as 4x half2, adding with __hadd2, storing float4 result. Geomean improved from 35.46µs to 33.09µs. Large sizes recovered (N=4096: 47.8→42.1µs, N=8192: 156.2→140.1µs) while small sizes remain excellent (N=1024: 12.0µs, N=2048: 17.0µs). Still 1µs above the 32.0µs baseline.

**Result:** 33.09 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using an inline CUDA kernel with half2 vectorized loads/stores.
Uses float4 casts for 128-bit transactions, __ldg() read-only cache for inputs,
strided grid pattern. Compiled once at module load time. Pre-allocated output per size.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Process 8 fp16 values (4 half2 = 1 float4 = 128 bits) per thread
__global__ void vadd_f4_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n_f4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n_f4; i += stride) {
        float4 va = __ldg(&a[i]);
        float4 vb = __ldg(&b[i]);
        // Reinterpret as half2 x 4 and add
        __half2* ha = reinterpret_cast<__half2*>(&va);
        __half2* hb = reinterpret_cast<__half2*>(&vb);
        float4 vc;
        __half2* hc = reinterpret_cast<__half2*>(&vc);
        hc[0] = __hadd2(ha[0], hb[0]);
        hc[1] = __hadd2(ha[1], hb[1]);
        hc[2] = __hadd2(ha[2], hb[2]);
        hc[3] = __hadd2(ha[3], hb[3]);
        c[i] = vc;
    }
}

void vadd_half2(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c)
{
    int n = a.numel();
    int n_f4 = n / 8;  // 8 fp16 per float4
    const int threads = 256;
    const int blocks = (n_f4 + threads - 1) / threads;
    vadd_f4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const float4*>(b.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(c.data_ptr<at::Half>()),
        n_f4
    );
}
"""

_cpp_src = r"""
void vadd_half2(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);
"""

_ext = load_inline(
    name="vadd_f4_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vadd_half2"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

_cached_out = {}

def custom_kernel(data):
    a, b = data
    n = a.numel()
    if n not in _cached_out:
        _cached_out[n] = torch.empty_like(a)
    out = _cached_out[n]
    _ext.vadd_half2(a, b, out)
    return out
# EVOLVE-BLOCK-END

```

---

## Experiment #8 — 2026-06-10 22:55:24 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed two changes: (1) replace strided loop with single-pass direct mapping (one thread per float4, no loop body, grid = cdiv(n_f4, threads)), and (2) replace plain store with __stcs() streaming store to bypass L1/L2 cache. Implemented both exactly. Result essentially unchanged from #7 (33.09→33.08µs) — single-pass and streaming stores made no meaningful difference. The bottleneck for large sizes (N=4096: 42µs, N=8192: 140µs) is not loop overhead or write caching, but something else (likely kernel launch overhead or insufficient occupancy for small sizes, and memory bandwidth ceiling for large sizes).

**Result:** 33.08 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using an inline CUDA kernel with half2 vectorized loads/stores.
Uses float4 casts for 128-bit transactions, __ldg() read-only cache for inputs,
strided grid pattern. Compiled once at module load time. Pre-allocated output per size.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Process 8 fp16 values (4 half2 = 1 float4 = 128 bits) per thread
// Single-pass: one thread per float4 element, no loop
// __stcs streaming store bypasses L1/L2 for write-only streaming output
__global__ void vadd_f4_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n_f4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_f4) return;
    float4 va = __ldg(&a[idx]);
    float4 vb = __ldg(&b[idx]);
    // Reinterpret as half2 x 4 and add
    __half2* ha = reinterpret_cast<__half2*>(&va);
    __half2* hb = reinterpret_cast<__half2*>(&vb);
    float4 vc;
    __half2* hc = reinterpret_cast<__half2*>(&vc);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    __stcs(&c[idx], vc);
}

void vadd_half2(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c)
{
    int n = a.numel();
    int n_f4 = n / 8;  // 8 fp16 per float4
    const int threads = 256;
    const int blocks = (n_f4 + threads - 1) / threads;
    vadd_f4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const float4*>(b.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(c.data_ptr<at::Half>()),
        n_f4
    );
}
"""

_cpp_src = r"""
void vadd_half2(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);
"""

_ext = load_inline(
    name="vadd_f4_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vadd_half2"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

_cached_out = {}

def custom_kernel(data):
    a, b = data
    n = a.numel()
    if n not in _cached_out:
        _cached_out[n] = torch.empty_like(a)
    out = _cached_out[n]
    _ext.vadd_half2(a, b, out)
    return out
# EVOLVE-BLOCK-END

```

---

## Experiment #9 — 2026-06-10 22:56:27 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed per-size specialized dispatch with different thread counts per size class to tune occupancy. Implemented: 128 threads/block for small tensors (N<=2M elements), 1024 threads/block for large tensors. This reduces block count for large sizes (N=8192: from ~32K blocks to ~8K blocks with 1024 threads). Result slightly worse than #8 (33.56µs vs 33.08µs) — thread block size tuning did not help. The kernel has plateaued around 33µs and per-size dispatch adds no benefit.

**Result:** 33.56 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Float16 vector addition using an inline CUDA kernel with half2 vectorized loads/stores.
Uses float4 casts for 128-bit transactions, __ldg() read-only cache for inputs,
strided grid pattern. Compiled once at module load time. Pre-allocated output per size.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Process 8 fp16 values (4 half2 = 1 float4 = 128 bits) per thread
// Single-pass: one thread per float4 element, no loop
// __stcs streaming store bypasses L1/L2 for write-only streaming output
__global__ void vadd_f4_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n_f4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_f4) return;
    float4 va = __ldg(&a[idx]);
    float4 vb = __ldg(&b[idx]);
    // Reinterpret as half2 x 4 and add
    __half2* ha = reinterpret_cast<__half2*>(&va);
    __half2* hb = reinterpret_cast<__half2*>(&vb);
    float4 vc;
    __half2* hc = reinterpret_cast<__half2*>(&vc);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    __stcs(&c[idx], vc);
}

void vadd_half2(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c)
{
    int n = a.numel();
    int n_f4 = n / 8;  // 8 fp16 per float4
    // Per-size tuned thread count:
    // Small tensors (<=2M elems): 128 threads/block -> fewer, larger blocks, lower launch overhead
    // Large tensors: 1024 threads/block -> fewer blocks, better SM utilization
    int threads;
    if (n <= 2*1024*1024) {
        threads = 128;
    } else {
        threads = 1024;
    }
    const int blocks = (n_f4 + threads - 1) / threads;
    vadd_f4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const float4*>(b.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(c.data_ptr<at::Half>()),
        n_f4
    );
}
"""

_cpp_src = r"""
void vadd_half2(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);
"""

_ext = load_inline(
    name="vadd_f4_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vadd_half2"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

_cached_out = {}

def custom_kernel(data):
    a, b = data
    n = a.numel()
    if n not in _cached_out:
        _cached_out[n] = torch.empty_like(a)
    out = _cached_out[n]
    _ext.vadd_half2(a, b, out)
    return out
# EVOLVE-BLOCK-END

```

---

## Experiment #10 — 2026-06-10 22:57:41 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed hybrid dispatch: float4 CUDA kernel for small tensors (N*N<=4M elements, i.e. N<=2048) and torch.add with pre-allocated output for large tensors (N>2048). Crossover at 4M elements based on observed per-size data. Implemented cleanly with reverted __stcs back to plain store (cleanest version from #7). Result: 33.1µs — essentially identical to pure CUDA kernel results (#7-#9). The hybrid didn't help because the large-size torch.add is also slow in this benchmark context (N=4096: 41.9µs, N=8192: 138.4µs) — the 32µs baseline geomean is itself a geometric mean across all four sizes, and the large-size torch.add performance here matches what we're seeing with the CUDA kernel.

**Result:** 33.11 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Hybrid float16 vector addition:
- Small tensors (N*N <= 4M elements): inline CUDA float4 kernel with __ldg loads
  (2-3x faster than torch.add at small sizes due to lower dispatch overhead)
- Large tensors (N*N > 4M elements): torch.add with pre-allocated output buffer
  (ATen's native kernel wins at large bandwidth-dominated sizes)
Crossover at N=2048 (4M elements): CUDA kernel at ~17µs, torch.add wins at N=4096+.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Process 8 fp16 values (4 half2 = 1 float4 = 128 bits) per thread
__global__ void vadd_f4_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n_f4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_f4) return;
    float4 va = __ldg(&a[idx]);
    float4 vb = __ldg(&b[idx]);
    __half2* ha = reinterpret_cast<__half2*>(&va);
    __half2* hb = reinterpret_cast<__half2*>(&vb);
    float4 vc;
    __half2* hc = reinterpret_cast<__half2*>(&vc);
    hc[0] = __hadd2(ha[0], hb[0]);
    hc[1] = __hadd2(ha[1], hb[1]);
    hc[2] = __hadd2(ha[2], hb[2]);
    hc[3] = __hadd2(ha[3], hb[3]);
    c[idx] = vc;
}

void vadd_small(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c)
{
    int n = a.numel();
    int n_f4 = n / 8;
    const int threads = 256;
    const int blocks = (n_f4 + threads - 1) / threads;
    vadd_f4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const float4*>(b.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(c.data_ptr<at::Half>()),
        n_f4
    );
}
"""

_cpp_src = r"""
void vadd_small(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);
"""

_ext = load_inline(
    name="vadd_hybrid_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vadd_small"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

# Crossover: CUDA kernel wins for N<=2048 (<=4M elements), torch.add wins for N>=4096
_CROSSOVER_NUMEL = 4 * 1024 * 1024  # 4M elements = N=2048 boundary

_cached_out = {}
_torch_add = torch.add

def custom_kernel(data):
    a, b = data
    n = a.numel()
    if n not in _cached_out:
        _cached_out[n] = torch.empty_like(a)
    out = _cached_out[n]
    if n <= _CROSSOVER_NUMEL:
        _ext.vadd_small(a, b, out)
    else:
        _torch_add(a, b, out=out)
    return out
# EVOLVE-BLOCK-END

```

---

## Experiment #11 — 2026-06-10 22:58:34 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed minimizing Python dispatch overhead with pre-bound method objects and pre-allocated output buffers. Implemented: stripped back to pure torch.add with dict.get() lookup (avoids 'in' + index double-lookup), pre-allocated output per size, no conditionals in hot path after first call. Dropped all CUDA extension overhead. Result: 32.44µs — closer to baseline (32.0µs) but still doesn't beat it. The dict.get() is marginally faster than the original _warmed_up boolean check approach. Large sizes still dominate geomean (N=4096: 42.2µs, N=8192: 138.4µs).

**Result:** 32.44 μs

**Kernel code:**
```python
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

```

