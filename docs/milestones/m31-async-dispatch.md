# M31 — Async Dispatch Substrate

**Crates:** `malus-runtime` (primary), `malus-sema` (barrier demotion), `malus-codegen-cpu` (host-read sites)
**Track:** perf
**Depends on:** M30
**Status:** done (2026-07-01) — warm median **6.065 ms/step** vs M30's 26.187 (4.3x; ≈2.2x vs f32 PyTorch-MPS at the toy config). See implementation notes at the bottom and the M31 addendum in `m29-benchmark-results.md`.

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

## Implementation notes (2026-07-01)

**Decisions grilled before implementation** (all user-approved): static barrier
insertion default-OFF behind `CheckOptions::insert_static_barriers` + hidden
`--static-barriers` CLI flag (A/B lever, deleted when V6's commit-planner
lands); pending tracking via commit-generation counters; per-handle
`flush_if_pending` rather than ambient `gpu_barrier` at read sites; chunked
commit-without-wait NOT built (measure-first — one buffer per step, flushed by
loss print / `bench_step_end`, hits no Metal limits at toy scale; re-verify at
M35 capstone scale); GPU errors panic at the flush with the Metal error
description; `MALUS_SYNC_DISPATCH=1` restores per-op fault attribution.

**Mechanism**: `ENCODE_GEN` bumps when the shared command buffer is (lazily)
opened; GPU-written outputs stamp `TensorBuffer::last_write_gen`;
`gpu_barrier()` advances `COMPLETED_GEN` after commit+wait; a buffer is
pending iff `last_write_gen > COMPLETED_GEN`. `reshape_to` aliases share the
underlying `MTLBuffer` and **inherit** the stamp (regression-tested). Drops of
pending tensors need no barrier: Metal command buffers retain referenced
resources.

**Results**: 6.065 ms/step warm median (min 5.291 / max 7.436, 297 warm
steps); A/B with `--static-barriers` = 24.015 ms/step (≈ M30 baseline,
empirically confirming the default-off call); loss curve bit-identical to
pre-M31 build. Full suite green with barriers off (done-when #4 proof runs by
default); M26 `cpu_compute_count()==0` and M29 RC-ratio gates green.

### Host-read audit (done-when #2 deliverable)

| Function | Reads device data? | Guard |
|---|---|---|
| `tensor_print` | yes (all dtypes) | `flush_if_pending` inserted (was: relied on CTMM barrier) |
| `malus_tensor_get_f32` (`t[i]`, `.data[i]`) | yes | `flush_if_pending` inserted (docstring previously claimed a barrier that didn't exist) |
| `permute_by_perm` (general N-D permute, ungated) | yes | self-flushing now; `tensor_permute`'s ambient `gpu_barrier()` removed |
| `tensor_matmul_cpu` (zero-dim fallback / test ground truth) | yes | per-input `flush_if_pending` (was: `gpu_barrier`) |
| `reshape_to` | no (clones buffer handle) | inherits `last_write_gen` from source |
| `tensor_len` / `tensor_ndim` / `tensor_dim` / shape uses | no (host-side metadata) | none needed |
| `tensor_alloc_gpu` / `randn` / `rand_uniform` / `rand_int` / `causal_mask` / `Buffer<i32>` `freeze` | no (write fresh buffers) | none needed |
| production `tape.rs` (`backward`, VJP dispatch) | no (seed grad uses loss shape only) | none needed |
| `cpu_fallback`-gated fwd ops (transpose, sum, broadcast, reduce, softmax, layernorm, gelu, cross_entropy, embedding) | yes (test-only builds) | keep their own `gpu_barrier()` (correct under async; precision irrelevant off the canonical path) |
| `cpu_fallback` helpers (`broadcast_to_shape`, `sum_to_shape`, `unsqueeze_at`, `tensor_scatter_add`, tape `read()`, `sum_bwd`, `cross_entropy_bwd` target read) | yes | `flush_if_pending` inserted (previously relied on caller discipline) |
