# MPS for stdlib ops, custom MSL for user-written kernels

**⚠ Status: Amended by ADR-0028 (V4).**  
The user/stdlib split is replaced by a hardware-capability split: ops that require vendor silicon not reachable from custom MSL (currently: matmul → MPS/AMX) remain builtins; everything else is written in the malus kernel language and dogfooded in the stdlib. The stdlib is no longer implemented in Rust CPU loops or MPS — it is implemented as malus kernels (ADR-0027). `matmul` remains MPS-backed per the vendor-primitives rule (ADR-0028).


Stdlib tensor operations (matmul, reductions, element-wise ops) use Apple's Metal Performance Shaders (MPS) for optimized implementations. User-written `kernel` functions always compile to custom MSL through malus's codegen pipeline. This gives users Apple's hand-tuned performance for standard operations while preserving the ability to write custom GPU code — which is malus's primary value proposition.

## Considered Options

- **MPSGraph for both**: MPSGraph is too high-level and assumes an eager or graph-execution model that conflicts with malus's dual-pipeline compilation approach.
- **Custom MSL for everything**: No reason to rewrite Apple's heavily optimized matmul implementation. MPS matmul on M-series chips uses AMX coprocessor instructions that are not accessible from custom MSL.
