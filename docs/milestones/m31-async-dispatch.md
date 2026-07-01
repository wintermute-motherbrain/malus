# M31 — Async Dispatch Substrate

**Crates:** `malus-runtime` (primary), `malus-sema` (barrier demotion), `malus-codegen-cpu` (host-read sites)
**Track:** perf
**Depends on:** M30
**Status:** planned

Kill the sync-per-matmul architecture. All GPU work — custom kernels *and* MPS matmul — encodes into the shared command buffer; commits happen only when the host actually needs data. Read safety moves from per-call-site workarounds to a runtime guarantee. This is phase 1 of the execution-model ladder (ADR-0035).

## The problem being fixed

`tensor_matmul` (`metal.rs:495-531`) currently calls `gpu_barrier()` (flushing all pending work and blocking), then encodes the MPS op, then commits and waits *again*. A transformer train step contains ~20+ matmuls (forward + backward), so the step is punctuated by dozens of full CPU↔GPU round-trips. Separately, `gpu_barrier()` is barrier-before-*drop*, not barrier-before-*read* (ADR-0032 known gap): RC-managed reads can see stale GPU state, worked around per-call-site with `__flush()`.

## Done-When

1. `tensor_matmul` encodes via `encodeToCommandBuffer:` into the current shared command buffer and returns a **pending** tensor. Zero `commit`/`waitUntilCompleted` inside any op function.
2. Every `TensorBuffer` carries a pending flag (or generation counter vs the last-committed generation). Every host-side read path — `tensor_print`, `.data[i]` buffer reads, scalar extraction, any `cpu_fallback` op — checks the flag and flushes first if pending. A full audit of host-read sites is part of the milestone deliverable.
3. All `__flush()` call sites (e.g. `examples/gradient_check.ml`) are deleted; grep for `__flush` returns nothing. The builtin may remain as a no-op or be removed — decide at implementation; user code must not need it.
4. CTMM's static barrier insertion (`insert_barriers`) is demoted from correctness mechanism to optimization: it may remain (a well-placed static flush can batch better than a lazy one) but the test suite must pass with it disabled, proving the runtime guarantee stands alone.
5. Full workspace test suite + all gradient checks pass under async dispatch. The M26 full-step `cpu_compute_count()==0` gate and M29 RC-ratio gate remain green.
6. Toy-config warm-median step time re-measured and published against the M30 baseline (expect roughly an order-of-magnitude improvement; not a gate — the V5 gate is M35's).
7. ADR-0035 written: execution-model ladder — async substrate (V5) → compile-time scheduling/fusion (V6) → optional static backward (V7); runtime lazy graph capture rejected.
8. `cargo test --workspace` passes.

## Scope

### 1. Async MPS matmul

Replace the commit+wait in `tensor_matmul` with `MPSMatrixMultiplication.encodeToCommandBuffer:` on the shared `current_command_buffer`. MPS encoders and custom-kernel compute encoders serialize correctly within one command buffer; the existing per-op `computeCommandEncoder`/`endEncoding` pattern already composes. The result tensor is pending, not ready — this changes `tensor_matmul`'s documented contract (CONTEXT.md "Ready tensor" lists MPS ops as ready; amend it).

### 2. Pending tracking

Simplest sound scheme: a global monotonically-increasing commit generation; each `TensorBuffer` written by GPU work stamps the generation of the command buffer it was encoded into; `flush_if_pending(handle)` commits+waits iff the buffer's generation is newer than the last completed one. One branch per host read; zero cost on the GPU path.

### 3. Host-read audit

Enumerate every runtime function that dereferences tensor contents on the CPU. Known: `tensor_print`, `.data[i]` index reads, `tensor_len`-adjacent shape uses are metadata-only (no flush needed), Philox `randn` writes into *fresh* buffers (no flush needed), batch-building `Buffer<i32>` writes are host-owned until `freeze`. The deliverable includes a table in the milestone notes: function → reads-device-data? → flush inserted?

### 4. Command-buffer lifecycle

With commits now rare, one command buffer can accumulate an entire forward+backward+optimizer step (loss printing forces one flush per step — same cadence as PyTorch's `.item()`). Verify Metal's limits (encoder count per buffer) are not hit at capstone scale; if they are, chunked commits without waits (commit-and-continue) are acceptable as long as reads still wait correctly.

## Out of Scope

- Buffer pooling (M32).
- Static commit-point planning / barrier coalescing in sema (V6, ADR-0035 phase 2).
- Kernel fusion (V6).
- Multi-buffer double-buffering across steps (only if M35 profiling demands it).
