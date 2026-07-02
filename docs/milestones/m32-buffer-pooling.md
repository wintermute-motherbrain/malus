# M32 — Buffer Pooling + Memory Budget

**Crates:** `malus-runtime`
**Track:** perf
**Depends on:** M31 (pooling a buffer requires knowing its GPU work completed — the pending tracking from M31 answers that)
**Status:** done (2026-07-01) — toy warm median 6.065 → ~2.6 ms/step ≈ **parity (0.95x) with f32 PyTorch-MPS** at the toy config; capstone-dims proxy peak 12.9 GB (≈16 GB with head-folding correction) on the 48 GB target; exact-size buckets confirmed by the allocation histogram (10 distinct sizes, all powers of two). Companions shipped: zeros blit-fill, `setBytes` for uniforms/TensorMeta, `MPSMatrixMultiplication` cache, soft memory-budget valve (`MALUS_MEM_BUDGET_MB`, default 8 GiB). See ADR-0039 and the M32 addendum in `m29-benchmark-results.md`.

Every op currently allocates a fresh `MTLBuffer` (`StorageModeShared`) and frees it at its CTMM drop point. At capstone scale that is thousands of ~1–100 MB device allocations per step. Recycle them.

## Done-When

1. A size-class free-list pool inside `MetalContext`: `tensor_alloc_*` draws from the pool on size-class hit; `tensor_release`-at-zero returns the buffer to the pool instead of dropping the `MTLBuffer`. Pool respects the M31 pending rules — a buffer is reusable only once the command-buffer generation that last wrote it has completed.
2. Pool hit-rate and peak device memory are measurable (`malus_pool_stats()` or equivalent, following the CPU-compute-counter pattern from ADR-0031).
3. Capstone-config memory budget verified: 6 layers retain ~100 MB of attention probabilities each on the tape (~0.6 GB), plus params/grads/Adam moments (~10–25M params ×4 tensors) — peak must fit comfortably on the 48 GB target machine with clear headroom; document actual peak.
4. Toy + (if M33/M34 have landed) capstone step times re-measured against baseline; published, not gated.
5. No leak: the M29 per-iteration RC-leak check (`test_v4_m29_rc_leak_assertion`) extended to assert pool-aware steady-state (pool size stabilizes across steps).
6. `cargo test --workspace` passes.

## Scope

- Size classes: round allocation sizes up (e.g. next power of two above 4 KB, exact-size buckets for the handful of hot shapes — decide empirically from an allocation histogram of one capstone step, which is itself a deliverable).
- The pool is per-`MetalContext` and inherits its single-consumer contract (ADR-0033); no locking beyond what `MetalContext` already does.
- `tensor_free`/`tensor_release` ABI is unchanged — pooling is invisible below the C ABI.

## Out of Scope

- Cross-step activation memory planning / gradient checkpointing (only if the M35 budget fails).
- `MTLHeap`-based suballocation — revisit in V6 if pool fragmentation shows up in profiles.
- In-place op rewriting (`inout` for stdlib ops) — V6, pairs with fusion.
