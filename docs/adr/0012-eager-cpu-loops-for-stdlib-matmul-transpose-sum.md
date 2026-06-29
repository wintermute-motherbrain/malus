# Eager CPU loops for V1 stdlib matmul / transpose / sum

In V1, `tensor_matmul`, `tensor_transpose`, and `tensor_sum` are implemented as eager
C-ABI functions that commit any pending GPU work (via `gpu_barrier`), read the
`StorageModeShared` buffers directly on the CPU, compute in plain Rust, and return a
ready (non-pending) output tensor.

This is a deliberate V1 stopgap. Both CPU loops and a naïve MSL kernel would be
throwaway placeholders relative to the real performance target: MPS
(MetalPerformanceShaders). MPS `MPSMatrixMultiplication` reaches the AMX coprocessor
on M-series chips, which is not accessible from custom MSL. ADR-0005 documents the
intent to use MPS for stdlib ops; this ADR records that migration is deferred
post-V1 because (a) the CPU-loop path is correct, (b) V1 is proving expressiveness
not throughput, and (c) adding raw `objc` MPS plumbing in M8 would add risk with no
proportionate benefit.

The cost is a second execution model: CTMM marks these eager results as pending (their
`return_placement` is `Some(Gpu)`) and may insert redundant `gpu_barrier` calls before
CPU reads. Those barriers are no-ops when there is no pending command buffer — a perf
and purity cost, not a correctness one.

## Migration path (completed in M21)

`tensor_matmul` was replaced with an MPS-backed implementation using
`MPSMatrixMultiplication` from `objc2-metal-performance-shaders 0.3`. The C-ABI
signature and call sites in codegen-cpu are unchanged; only the runtime body changed.
The runtime was simultaneously ported from the deprecated `metal-rs 0.29` / `objc 0.2`
stack to `objc2 0.6` + `objc2-metal 0.3`. `tensor_transpose` and `tensor_sum` remain as
eager CPU loops (not on the critical path). See ADR-0017 for the scope decision.
