# MPS migration for matmul and axis reductions

Amends ADR-0012 (eager CPU loops for matmul/transpose/sum).

## Decision

In M21, migrate `tensor_matmul` (2-D and batched), `tensor_reduce_sum_axis`, and `tensor_reduce_mean_axis` from eager CPU loops to `MPSMatrixMultiplication` / custom Metal kernels. The migrated ops return **pending tensors** (no internal `gpu_barrier` call) so CTMM can batch command buffers across chained ops.

## Why now

ADR-0012 deferred MPS migration on the grounds that V1 "proves expressiveness, not throughput." That argument holds for the MLP capstone but fails for nanoGPT: a transformer forward+backward at any useful scale runs dozens of matmuls per step over thousands of steps. On the CPU triple-loop implementation (`tensor_matmul:248–282`), a 512×512 matmul takes ~100ms; a single transformer step is seconds. The V3 capstone would take days to show visible loss decrease, making it non-demonstrable.

The prerequisite work is in place by M21: the pending-tensor model exists (CTMM already inserts barriers), the command buffer and pipeline infrastructure is in `MetalContext`, and VJP rules for matmul are implemented in Rust closures that can issue their own `gpu_barrier` before reading back gradients.

## Consequences

- `tensor_matmul` no longer calls `gpu_barrier()` internally — it encodes an MPS op and returns a pending tensor. Callers that need a ready tensor (e.g. VJP backward closures that read back gradients to CPU) must issue their own barrier.
- `tensor_transpose` is optionally migrated; the CPU loop path is retained as a fallback. `tensor_sum` (whole-tensor, V1) stays as a CPU loop — it is rarely on the critical path and MPS overhead for small tensors is non-trivial.
- The existing eager-CPU test suite (`malus-runtime/src/tests.rs`) gains a tolerance-based correctness comparison between MPS and CPU results (1e-3 for f32).
- VJP closures for matmul (M14) and batched matmul (M17) are updated to call `gpu_barrier()` before leaf-gradient accumulation.
