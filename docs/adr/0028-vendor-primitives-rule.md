# ADR-0028 — Vendor Primitives Rule: MPS Builtins vs. Kernel Language

**Status:** Accepted (V4)  
**Amends:** ADR-0005 (MPS for stdlib — partially; inverts the "stdlib → MPS" rule)

## Context

ADR-0005 established: "stdlib tensor operations use MPS, user-written kernels compile to custom MSL." This drew the line at user/stdlib, not at hardware capability. The result: V3 mixed MPS matmul with CPU-loop softmax/layernorm — inconsistent and half-computed on GPU.

ADR-0027 inverts ADR-0005 by rewriting all stdlib ops as malus kernels. But matmul must remain MPS: `MPSMatrixMultiplication` reaches the AMX matrix coprocessor on M-series chips, which is not addressable from custom MSL compute kernels. A malus-kernel matmul would be a generic SIMD implementation, losing 5-10× to MPS — defeating the PyTorch-MPS benchmark target.

## Decision

The line is drawn at **hardware capability**, not user vs. stdlib:

**Vendor primitive** (blessed builtin): an operation whose optimal implementation requires hardware resources that are not accessible from custom MSL compute kernels. For Apple Silicon, this currently means: `MPSMatrixMultiplication` (AMX coprocessor). Vendor primitives remain hard-coded Rust functions in `malus-runtime` with C-ABI names callable from JIT'd code. They are not expressed in the kernel language.

**Kernel language** (everything else): if optimal performance is achievable from a standard MSL compute kernel (shared memory, barriers, SIMD groups, device memory), the op must be expressed as a malus kernel. This includes softmax, layernorm, gelu, activation functions, embedding, axis reductions, cross-entropy, broadcasting, and all backward-pass ops.

**Corollary:** the stdlib is dogfooded in the kernel language. If a user can write it optimally in the kernel language, it belongs in the kernel language — in the stdlib `.ml`. The compiler and users see the same facility.

**Known vendor primitives in V4:**
- `matmul` (2-D, batched 3-D, 3-D⊗2-D broadcast) → `MPSMatrixMultiplication`

**Reserved for post-V4:**
- Fused flash-attention (tiled, online softmax, in-kernel matmul via tensor cores/AMX — a deliberate exception to the kernel-language rule, post-V4 alongside mixed precision).
- Any future `MPSGraph`-backed ops, if they reach hardware not addressable from MSL.

## Why this is correct

This is not special-casing. It is universal practice: Triton calls cuBLAS/cuDNN for GEMM; JAX calls cuDNN for convolutions; PyTorch calls cuBLAS. "Write everything in your own kernel language" is ideologically pure and practically slower. The rule is: use the vendor library where it reaches dedicated silicon you cannot otherwise touch; write everything else yourself.

## Consequences

- ADR-0005's "stdlib → MPS" rule is inverted: "stdlib → kernel language, except AMX-gated ops → MPS."
- `tensor_matmul` (MPS) stays in `malus-runtime/src/metal.rs` as a C-ABI Rust function.
- `tensor_softmax_axis_cpu`, `tensor_layernorm_axis`, `tensor_gelu`, etc. are retired (behind `#[cfg(feature = "cpu_fallback")]`).
- New `stdlib/` directory of `.ml` kernel files ships with the compiler.
