# Advisor Proposals

---

## Iteration 1

## STATE
Single baseline on the board at 32.00 µs geomean. The current approach is pure PyTorch `torch.add` with a pre-allocated output buffer. No custom kernels have been attempted yet. SOL estimates put the four sizes at ~1.9/7.5/30/120 µs — the 32 µs geomean suggests we're running well above SOL for smaller sizes (kernel launch overhead) and possibly at or near SOL for the larger ones. The noise floor for this problem is roughly 1–3 µs small / 5–15 µs large.

## RATIONALE
`torch.add` goes through Python dispatch, ATen, and CUDA graph overhead on every call. For a pure memory-bandwidth-bound kernel this thin, the bottleneck at small sizes is launch latency and dispatch overhead, not compute or bandwidth. A hand-written Triton kernel with tuned grid sizing and 128-bit vectorized loads/stores can eliminate the ATen dispatch path, reduce launch overhead, and saturate HBM3 bandwidth more directly. The key axis to exploit is maximum memory throughput: wide vector loads (128-bit / 8 fp16 per transaction), minimal occupancy overhead, and correct block tiling for each size class.

## PROPOSAL
Write a Triton kernel that performs vectorized fp16 addition using 128-bit (8×fp16) loads and stores. The kernel should flatten the 2D tensor to a 1D view and process elements in wide chunks. Use a single-level 1D grid — no 2D tiling needed since this is a flat memory operation. The worker should experiment with block sizes that keep the grid large enough to saturate all 132 SMs on the H100 while maximizing memory transaction efficiency. The output buffer should be pre-allocated and reused across calls. Focus first on getting the bandwidth utilization close to the SOL floor before tuning launch configuration.

---

## Iteration 2

## STATE
Two experiments on the board. The Triton kernel (experiment #2) came in at 54.25 µs — worse than the 32.00 µs baseline. This is a first-attempt Triton result and is untuned; the Triton approach is still maturing. The regression almost certainly stems from: (1) Triton JIT compilation overhead bleeding into the first measured call, (2) suboptimal block size (1024 fp16 elements = 2 KB, far below the 128-byte transaction granularity sweet spot per warp), and/or (3) insufficient warps per SM to hide memory latency. The baseline is a well-warmed PyTorch call that benefits from cuBLAS/ATen kernel caching.

## RATIONALE
The Triton approach hasn't been allowed to mature. The specific failure mode here is likely insufficient parallelism and block size mismatch: with BLOCK_SIZE=1024, the grid for N=1024 (1M elements) is only ~1024 blocks — that's manageable — but for N=4096 (16M elements) we get ~16K blocks. The real issue is likely JIT warmup on first call and per-block overhead exceeding memory bandwidth benefit. The fix is to dramatically increase the number of elements per block to amortize grid launch overhead and maximize memory transaction width, while ensuring the kernel is fully pre-compiled before measurement begins by adding an explicit warmup step. Additionally, using `tl.load` with explicit `cache_modifier=".cg"` (L2 cache bypass) for streaming access patterns can improve bandwidth for the large sizes.

## PROPOSAL
Revise the Triton kernel with two key changes: (1) add an explicit kernel warmup call at initialization time (with a small dummy tensor) so JIT compilation cost is never paid during benchmarking, and (2) significantly increase the elements-per-block to reduce grid overhead and maximize each block's memory transaction efficiency — the worker should experiment with much larger block sizes that keep each thread block doing substantial work per launch. Also explore whether pinning the number of programs to a fixed multiple of SM count (rather than cdiv) helps load balance. Keep the pre-allocated output buffer approach.

---

## Iteration 3

## STATE
Three experiments in. The baseline torch.add sits at 32.00 µs geomean. Both Triton attempts landed at ~54–55 µs — worse than baseline and not improving across iterations. The worker correctly identified that JIT warmup was not the bottleneck. The Triton kernel overhead itself is the problem: a custom Triton loop is actually slower than torch.add for this specific task, likely because cuBLAS/CUDA's built-in elementwise kernels are already highly optimized and the Python→Triton dispatch overhead exceeds any bandwidth advantage. The Triton direction is now showing two flat attempts with no improvement — it's maturing into a dead end.

## RATIONALE
The baseline `torch.add` with pre-allocated output is already a well-optimized path. At 32 µs geomean vs SOL ~30 µs for N=4096, the large sizes are near-SOL already. The gap to close is at *small sizes* where kernel launch latency dominates. The right next move is to explore CUDA Graphs: by capturing a `torch.add` (or even a simple elementwise) into a CUDA Graph, kernel replay eliminates all CPU-side dispatch overhead for repeated same-size calls. For N=1024 and N=2048, the SOL is ~1.9 µs and ~7.5 µs — the current 32 µs geomean suggests small sizes are taking 5–15 µs, meaning launch overhead is a large fraction. CUDA Graph replay can bring that closer to the true HBM floor with near-zero CPU overhead.

## PROPOSAL
Abandon the Triton direction entirely — it has matured into a dead end with two flat results. Pivot to **CUDA Graphs**: use `torch.cuda.CUDAGraph` to capture a `torch.add(..., out=output)` call for each distinct tensor size encountered, then replay the graph on subsequent calls of the same size. This eliminates Python dispatch and CUDA kernel launch overhead from the hot path. The worker should build a dict keyed by tensor size, lazily capturing graphs on first encounter and replaying on all subsequent calls. Static input buffers (required by CUDA Graph semantics) should be pre-allocated per size, with a `.copy_()` to populate them before each graph replay. Focus on whether the replay latency for small sizes drops significantly below the current 32 µs baseline.

---

## Iteration 4

## STATE
Four experiments, all worse than the 32 µs baseline. The run has now invalidated three distinct strategies: Triton kernels (~54 µs, launch overhead dominates), CUDA Graphs with copy_ (~73 µs, memcpy cost exceeds savings), and block-size tuning within Triton (flat). The baseline `torch.add` with pre-allocated output is proving very hard to beat — it's already a thin, optimized path. The large-size SOL is ~120 µs for N=8192 and ~30 µs for N=4096, and the 32 µs geomean suggests large sizes are running near SOL while small sizes inflate the average.

## RATIONALE
Every custom approach so far adds overhead rather than removing it. The real insight from CUDA Graphs failure is that `copy_()` costs as much as the addition itself — meaning the problem is purely memory-bandwidth bound with no room for staging. The baseline is already close to optimal for large sizes. The only remaining lever that hasn't been tried is **inline CUDA via `torch.utils.cpp_extension.load_inline`** — a hand-written CUDA kernel that uses `__half2` vectorized instructions (processes two fp16 values per 32-bit register), explicit loop unrolling, and zero Python dispatch overhead. Unlike Triton, this runs true PTX with full control over the instruction schedule. Crucially, the kernel should be pre-compiled at module load time so zero compilation cost bleeds into benchmarking, and the output buffer should be pre-allocated per size. The target is to match or beat `torch.add`'s native CUDA path for all four sizes.

## PROPOSAL
Try an **inline CUDA kernel** using `torch.utils.cpp_extension.load_inline`. Write a CUDA kernel that uses `half2` vectorized loads and stores (2 fp16 per 32-bit transaction, paired with 128-bit `float4` casts for maximum throughput), processes the flat tensor with a strided grid pattern to cover all elements, and is compiled once at module load time into a cached extension. The hot path in `custom_kernel` should only do: populate the pre-allocated output pointer, launch the kernel with a fixed grid (pre-computed per size), and return. The worker should explore whether using `__ldg()` (read-only data cache) for the input loads and `__stcs()` streaming stores for the output improves bandwidth utilization beyond what `torch.add` achieves.

---

## Iteration 5

## STATE
Five experiments in. The inline CUDA kernel (experiment #5) is the most informative yet: small sizes improved dramatically (N=1024: ~12 µs, N=2048: ~20 µs) but large sizes regressed badly (N=4096: ~57 µs, N=8192: ~190 µs) because of the fixed 528-block grid cap. The geomean is 40 µs vs 32 µs baseline. This approach is clearly still maturing — the ceiling is visible (small sizes are beating baseline by 2-3×) but large sizes are bottlenecked by the artificial grid size limit causing all elements to be processed in a strided serial loop rather than in parallel. The fix is straightforward: remove the 528-block cap and let the grid scale with tensor size.

## RATIONALE
The inline CUDA kernel approach is the right direction — it showed the best small-size performance we've seen. The regression for large sizes is a single implementation bug: `min(528, ...)` caps the grid at 528 blocks regardless of tensor size. For N=8192 (67M elements, 33M half2 pairs), 528 blocks of 256 threads = 135K threads handling 33M elements each in a ~245-iteration loop — far too serial. With a properly scaled grid (enough blocks to cover all elements directly with minimal looping), each thread does O(1) work and memory bandwidth saturates naturally. The worker just needs to remove the block cap and use the full `cdiv(n_half2, threads)` grid, optionally clamped at a high limit like 65535 to stay within CUDA bounds while still being HBM-saturating.

## PROPOSAL
Continue with the inline CUDA `half2` kernel but fix the grid sizing: scale the number of blocks with the tensor size using `ceil(n_half2 / threads)`, removing the artificial 528-block ceiling. The worker may optionally clamp at a very large limit (e.g., the CUDA max grid dim) but should NOT artificially cap at SM-count × 4. Also consider switching from a strided loop to a direct one-shot mapping (one thread per half2 element, no loop body) for the largest sizes — this is the standard approach for memory-bandwidth-bound kernels. Keep the pre-allocated output cache and the `__ldg` read-only loads. This single change should recover large-size performance to near-SOL while preserving the small-size gains already demonstrated.

---

## Iteration 6

## STATE
The inline CUDA `half2` approach is clearly the right direction and still maturing. Experiment #6 improved from 40.0 µs to 35.46 µs by fixing the block cap — now only 3.5 µs above baseline. Small sizes are beating baseline (N=1024: ~12 µs vs SOL ~1.9 µs — still launch-overhead-bound; N=2048: ~18 µs). Large sizes remain the weak point: N=4096 at ~48 µs (SOL ~30 µs), N=8192 at ~156 µs (SOL ~120 µs) — both still ~30-40% above SOL. The kernel is improving iteration-over-iteration; this approach has not yet flattened.

## RATIONALE
The large-size gap (~30-40% above SOL) is almost certainly due to insufficient memory transaction width. The current kernel processes 2 fp16 values (32 bits) per thread per iteration — but H100 memory transactions are 128 bytes wide, and each warp (32 threads) issues one transaction. At 32 bits/thread × 32 threads = 128 bytes/transaction, this is correct in theory. However, the bottleneck is likely **insufficient in-flight memory requests**: to fully saturate HBM3, you need many outstanding transactions per SM. The standard fix is to have each thread process **multiple elements per memory instruction** using `float4` (8 fp16 values = 128 bits per thread), which increases arithmetic intensity per thread and gives the memory subsystem more latitude to coalesce and pipeline. Additionally, the current kernel uses `__ldg` (L1/L2 read-only path), but for streaming access patterns that exceed L2 (N≥4096, 100+ MB), bypassing L1 entirely with `.cs` (cache streaming) or `.cg` loads may improve throughput.

## PROPOSAL
Upgrade the inline CUDA kernel to process **8 fp16 values per thread** using `float4` casts: load 128 bits (one `float4`) per thread from each input, reinterpret as `half2 × 4`, perform four `__hadd2` operations, then store 128 bits to output. This increases the work-per-thread and memory-request granularity, potentially improving memory-level parallelism for large tensors. The grid size should then be `cdiv(n / 8, threads)`. For the large streaming sizes, also try adding `__builtin_assume_aligned` hints or explicit `.cs` cache modifiers (via PTX `asm volatile`) on the stores to hint the hardware not to cache output. Keep the pre-allocated output dict. This single vectorization change is the key remaining gap between the current kernel and SOL for large sizes.

---

## Iteration 7

## STATE
The inline CUDA approach is steadily converging: #5 (40.0 µs) → #6 (35.5 µs) → #7 (33.1 µs), now only ~1 µs above the 32.0 µs baseline. The float4 kernel is working well for small sizes (N=1024: ~12 µs, N=2048: ~17 µs) but large sizes are still above SOL: N=4096 at ~42 µs (SOL 30 µs), N=8192 at ~140 µs (SOL 120 µs). The geomean is dominated by the large sizes. The kernel is still maturing with a clear positive slope.

## RATIONALE
The remaining ~17% gap at large sizes (42 µs vs 30 µs SOL for N=4096) points to two likely causes: (1) the strided loop pattern means threads revisit elements in non-ideal order, creating serialization in the memory pipeline even with 128-bit loads; (2) the store path (`c[i] = vc`) goes through L1/L2 cache unnecessarily for a write-only streaming operation. For a pure streaming write-once workload, bypassing the L1 cache on stores with explicit non-temporal/streaming stores (`.cs` in PTX, or `__stcs`) can materially improve write bandwidth. Additionally, the strided loop (even with float4) means each warp issues multiple rounds of memory requests sequentially — a single-pass "one thread per float4 element" grid (no loop at all) with `gridDim.x * blockDim.x >= n_f4` eliminates the loop overhead entirely and may improve memory-level parallelism by letting the hardware issue all loads at once.

## PROPOSAL
Make two targeted changes to the float4 inline CUDA kernel: (1) replace the strided loop with a **direct single-pass mapping** — one thread per float4 element, no loop body — by ensuring the grid covers all elements; and (2) replace the plain store `c[i] = vc` with a **streaming/non-temporal store** using `__stcs()` (or the equivalent `stcs` PTX instruction) to bypass L1/L2 cache on the write path. For write-once streaming outputs this avoids cache pollution and can push write bandwidth closer to the theoretical HBM limit. Keep `__ldg` on the input reads. This combination should close the remaining gap to SOL for large sizes.

---

## Iteration 8

## STATE
The inline CUDA float4 kernel has plateaued at ~33.08–33.09 µs across the last two experiments — experiments #7 and #8 are essentially identical results despite different implementation changes (single-pass vs strided, plain store vs __stcs). This is the noise floor of the approach. The kernel sits 1 µs above the 32 µs baseline, which is within run-to-run variance (~1–3 µs for small sizes). Large sizes (N=4096: ~42 µs vs SOL 30 µs; N=8192: ~140 µs vs SOL 120 µs) are ~15-17% above SOL and not improving further with these micro-optimizations.

## RATIONALE
The float4 inline CUDA kernel has matured — two consecutive attempts show flat results (33.09 → 33.08 µs). The 1 µs gap to baseline is within noise. The approach ceiling appears to be around 33 µs. The baseline `torch.add` at 32 µs is already using a highly optimized ATen CUDA elementwise kernel that likely uses similar or identical vectorization internally. The remaining large-size gap (42 µs vs 30 µs SOL for N=4096) is structural — likely due to kernel launch overhead being amortized over fewer waves for mid-size tensors, not addressable by further kernel tweaks. The most promising unexplored angle is **occupancy and wave efficiency**: at N=4096 (16M elements, 2M float4s), dividing by 256 threads/block gives ~8K blocks. On 132 SMs that's ~60 blocks/SM — potentially too many serialized waves. Experimenting with larger thread counts per block (to reduce block count and increase register reuse) or explicitly tuning the thread-block size for each size class could help.

## PROPOSAL
Rather than further micro-optimizing the CUDA kernel, pivot to a **thread-block size sweep** for the float4 kernel: the current 256 threads/block may not be optimal for all sizes. Try significantly larger thread counts (saturating each SM's warp slots more aggressively) to reduce the total block count and potentially improve the ratio of computation to kernel launch overhead. Alternatively, explore a **per-size specialized dispatch**: for small sizes (N≤2048, where kernel launch overhead dominates), use a very compact kernel with minimal blocks; for large sizes (N≥4096), use the full vectorized float4 path. The key insight is that a single fixed thread configuration cannot be simultaneously optimal for both 1M-element and 67M-element tensors. The worker should measure per-size latency and tune the configuration for each size class independently.

---

## Iteration 9

## STATE
The inline CUDA float4 kernel has thoroughly plateaued at ~33 µs across experiments #7, #8, and #9 (33.09 → 33.08 → 33.56 µs). Three consecutive attempts with meaningfully different changes all land within noise of each other. The approach ceiling is ~33 µs, still ~1 µs above the 32 µs baseline. This is the last iteration (9/10), so this is the final proposal. The large-size gap (N=4096: ~42 µs vs 30 µs SOL; N=8192: ~140 µs vs 120 µs SOL) is structural — the custom CUDA kernel cannot match ATen's tuned elementwise kernel for the large bandwidth-dominated sizes.

## RATIONALE
With one iteration left, the high-value move is not to keep tuning the CUDA kernel (which has flatlined) but to reconsider whether a **hybrid approach** — using the custom CUDA kernel only where it shows a clear advantage (small sizes, where it's ~2× faster than baseline) and falling back to `torch.add` for large sizes where ATen wins — can produce a geomean lower than both. The small-size data is compelling: N=1024 at ~12 µs vs baseline's implied ~12-15 µs (hard to know exactly without per-size breakdown from #1, but the geomean arithmetic suggests large sizes are pulling the baseline down). Actually, more precisely: if the baseline is 32 µs geomean and the CUDA kernel gives 33 µs geomean, and small sizes are better with CUDA but large sizes are worse, the hybrid captures the best of both worlds.

## PROPOSAL
For the final iteration, implement a **hybrid dispatch**: use the float4 inline CUDA kernel (from experiment #7/#8, without the __stcs and without per-size thread tuning — the cleanest version) for small tensors where it demonstrably beats the baseline, and use `torch.add` with the pre-allocated output buffer for large tensors where ATen's native kernel is faster. The crossover point should be determined by the per-size data from the history: the CUDA kernel wins at N=1024 and N=2048, while `torch.add` wins at N=4096 and N=8192. This hybrid directly combines the best per-size performance observed and should produce a geomean below 32 µs.

---

## Iteration 10

## STATE
This is the final iteration (10/10). The entire epoch has been unable to beat the 32.00 µs baseline. The inline CUDA float4 kernel plateaued at ~33 µs across five consecutive experiments (#7–#10), all within 0.5 µs of each other — solidly in the noise floor. The hybrid dispatch (#10) revealed that `torch.add` in this benchmark context also produces ~42 µs at N=4096 and ~138 µs at N=8192, meaning the 32 µs baseline geomean must come from prior warmup or measurement conditions that our custom kernels aren't replicating. The baseline is essentially unbeatable with the approaches explored in this epoch.

## RATIONALE
This is the last iteration of the last epoch. The key learning from this entire run: the baseline `torch.add` is already highly optimized and the 32 µs baseline number reflects a stable, well-warmed PyTorch ATen path. Every custom kernel approach — Triton, CUDA Graphs, inline CUDA — has either matched it at best or fallen short. The inline CUDA float4 kernel came the closest (~33 µs, 3% above baseline) but the remaining gap is structural noise. No further improvement is achievable with the remaining iteration budget. The most honest final proposal is to attempt a `torch.add` + output caching variant that eliminates the conditional branch in the hot path — stripping the current baseline down to the absolute minimum Python overhead.

## PROPOSAL
For the final attempt, try eliminating even the `_warmed_up` conditional check from the baseline hot path by using a **pre-bound method object** approach: at module load time, pre-allocate the output buffer for all four expected benchmark sizes, then bind the `torch.add` call parameters into a closure or `functools.partial` so the hot path is a single Python function call with zero attribute lookups. The worker should also try using `a.add_(b)` (in-place on a pre-copied buffer) or `torch.add(a, b, alpha=1, out=out)` with the output pre-allocated per size — whichever removes the most Python overhead from repeated calls at the same size. The goal is to squeeze the baseline from 32 µs down into the 28–30 µs range by eliminating the last remnants of Python dispatch cost.

