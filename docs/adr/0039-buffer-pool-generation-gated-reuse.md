# Buffer pool: generation-gated reuse + soft memory-budget valve

## Decision

M32 recycles `MTLBuffer`s through an exact-byte-size free-list pool inside `MetalContext`, gated by the M31 commit-generation machinery: every encode site stamps a **last-use generation** on every buffer it touches (inputs and output), and a pooled buffer is reusable only once that generation has completed. `tensor_release`-at-zero returns the buffer to the pool instead of dropping it; `tensor_alloc_*` pops a completed same-size entry before allocating fresh. Reshape aliases share an `Arc<PoolState>` — the `MTLBuffer` may enter the pool only when the last `TensorBuffer` referencing it dies (`Arc::strong_count == 1`).

Three companions ride the same milestone:

- **Zeros blit-fill**: `tensor_alloc_zeros_gpu` pools too; a pool hit is dirty, so it encodes an `MTLBlitCommandEncoder` `fillBuffer(value: 0)` into the shared command buffer and stamps the write generation (host reads auto-flush per ADR-0035). A miss stays a fresh OS-zeroed buffer with no fill.
- **Soft memory-budget valve**: on a pool miss that would push live+pooled bytes past a budget (`MALUS_MEM_BUDGET_MB`, default 8 GiB), and only when the missed size's bucket holds pending entries a flush would unlock, the allocator fires one `gpu_barrier()` and retries the pool. A retry miss still allocates fresh — the budget is a recycling trigger, not a cap.
- **Transient-data `setBytes` + MPS kernel cache**: the per-dispatch uniforms blob and per-tensor 68-byte `TensorMeta` records use `setBytes` instead of freshly allocated `MTLBuffer`s, and `MPSMatrixMultiplication` kernels are cached by `(result_rows, result_cols, interior_cols)` — the only init parameters that vary.

## Why this is surprising

**Pooling gates on last-*use* generation, not M31's last-*write* generation.** Write-gen answers "may the host read this?"; it says nothing about a buffer that is an in-flight *input* to uncommitted work. Recycling such a buffer would hand it to a CPU memcpy that races the not-yet-executed GPU read. So the pool needed its own stamp, and the invariant threads through every encode site: **any new encode path must `stamp_use` every buffer it touches** or the pool silently corrupts data.

**A null-data `tensor_alloc_gpu` no longer returns a zeroed buffer.** Fresh `StorageModeShared` buffers arrive OS-zeroed; pooled ones are dirty. Every caller either passes data, fully overwrites on GPU (audited at M32: no stdlib kernel reads or partially writes its output), or must use `tensor_alloc_zeros_gpu`.

**The pool only cycles when something flushes.** A training loop that never reads the GPU never advances the completed generation, so every pooled entry stays pending forever — 0% hit rate and device memory growing by one step's temporaries per step. This isn't a pool defect (Metal command buffers retain encoded resources, so pre-M32 builds grew the same way); the valve exists to bound it. Realistic loops (per-step loss read, `--bench` flush) cycle the pool without ever hitting the valve.

**Allocation must never happen while `current_command_buffer` is held.** The valve calls `gpu_barrier()` from inside the allocator, which takes that lock. Every encode site allocates its output *before* taking the guard — true today and now load-bearing.

## Considered alternatives

**Metal completion handlers** (per-command-buffer callbacks moving buffers to the pool on completion). Push-based and precise, but handlers fire on a Metal-owned thread, which violates the `MetalContext` single-consumer contract (ADR-0033) and would force real locking discipline onto every pool touch. The generation compare is a two-atomic read on the allocating thread. Rejected.

**Size-class rounding (next power of two above 4 KB).** The spec left the bucket policy to an allocation histogram, which settled it: the toy nanoGPT run allocates exactly **10 distinct sizes** (4 B–32 KB, all already powers of two), and transformer training reuses identical shapes every step. Rounding would change nothing today and waste up to 2x per buffer at capstone scale, where the budget matters. Misses are gen-gated (pending entries between flush points), not size scatter — no bucket policy can remove them; that is V6 commit-planner territory. Exact-size buckets, revisit only if a future histogram shows near-miss scatter.

**Pool eviction / hard cap.** With the zeros fix the pool provably stabilizes across steady-state steps (asserted by the extended M29 leak test), so eviction would be untested complexity serving no measured need. The pool is unbounded by design; `malus_pool_reset()` drains it for tests. `MTLHeap` suballocation stays deferred to V6 per the spec.

**CPU memset for pooled zeros.** Simpler than the blit, but puts a large memset on the CPU hot path — the exact overhead class V5 exists to remove. Rejected.

## Consequences

- Toy-config warm median dropped **6.065 → ~2.6 ms/step** (runs range 2.4–3.7 ms), ≈ **parity with PyTorch-MPS f32** (2.729 ms) at the toy config — before M33–M35 land.
- Observability follows the ADR-0031 counter pattern: `malus_pool_stats()` (hits/misses/pooled/peak), `malus_pool_buckets()`, `malus_pool_reset()`, and a `MALUS_ALLOC_HISTOGRAM=1` size histogram; the CLI prints pool stats under `--bench`.
- The M29 leak test now also asserts the end-of-run pool is byte- and bucket-identical across step counts — the assertion that catches any alloc path that bypasses the pool on the way in but feeds it on the way out.
- `stamp_use` is a standing obligation on every future encode site (kernel dispatch, MPS ops, blit fills).
- Buffers enter the pool carrying possibly-pending generations; releases can arrive out of gen order, so the pop scans its bucket for the first completed entry rather than trusting the front.
