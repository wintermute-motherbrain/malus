# MPS migration for matmul

Amends ADR-0012 (eager CPU loops for matmul/transpose/sum).

## Decision

In M21, migrate `tensor_matmul` (2-D, batched 3-D, and 3-D⊗2-D broadcast) from eager
CPU triple-loops to `MPSMatrixMultiplication` via `objc2-metal-performance-shaders 0.3`.
Axis reductions (`tensor_reduce_sum_axis`, `tensor_reduce_mean_axis`) and `tensor_transpose`
stay as CPU loops — they are not on the transformer's critical matmul path and MPS overhead
for small tensors is non-trivial.

`tensor_matmul` is **eager**: it calls `gpu_barrier()` first to flush any pending
element-wise kernels, then encodes all MPS ops (one per batch slice) into a fresh command
buffer, commits, and waits. It returns a ready tensor. Pending-tensor matmul is deferred
post-V3. The 10× speedup rationale comes from AMX compute, not command-buffer batching, so
the eager design loses nothing material.

## Why now

ADR-0012 deferred MPS migration on the grounds that V1 "proves expressiveness, not
throughput." That argument holds for the MLP capstone but fails for nanoGPT: a transformer
forward+backward at any useful scale runs dozens of matmuls per step over thousands of steps.
A 512×512 matmul takes ~100ms on the CPU loop; a single transformer step is seconds. The V3
capstone would take days to show visible loss decrease, making it non-demonstrable.

## Implementation note

The port required migrating `malus-runtime` off the deprecated `metal-rs 0.29` / `objc 0.2`
stack onto `objc2 0.6` + `objc2-metal 0.3` (commit 1). The MPS bindings
(`objc2-metal-performance-shaders 0.3`) were not available in metal-rs and are the
direct enabler for this migration (commit 2).

## Consequences

- `tensor_matmul_cpu` is kept as a private reference implementation for differential
  correctness testing (max-abs-diff < 1e-3 vs MPS, tested in `tests.rs`).
- `tensor_transpose`, `tensor_sum`, and all axis reductions remain CPU-eager; CTMM is
  unchanged.
- VJP closures for matmul in `tape.rs` still call `gpu_barrier()` before reading gradients;
  these calls are now redundant (MPS matmul already waits) but harmless.
