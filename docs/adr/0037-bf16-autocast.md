# Mixed precision: bf16-first with autocast-style semantics; f16 + loss scaling deferred

## Decision

V5's M36 (sequenced strictly after the f32 ≤2x gate passes at M35) makes `bf16` malus's first non-f32 float compute dtype, with **autocast-style** mixed-precision training:

- Parameters, gradients, and optimizer state stay f32 (full-precision master path).
- Matmuls and forward elementwise kernels compute in MSL `bfloat` (Metal 3.1+/M2+; the target machine is M4 Max).
- Reductions — softmax, layernorm, cross-entropy accumulations — accumulate in f32.
- f16 is deferred: bf16 shares f32's exponent range, so no loss scaling is needed; f16 would require a loss-scaling mechanism for zero additional capstone benefit.

The op-level cast policy (exactly which kernels take/produce `bfloat`, where f32↔bf16 conversion kernels sit) and the user-facing surface are finalized when M36 begins, amending this ADR. The recommendation on record: a scoped `with autocast:` block mirroring `no_grad`'s existing scope machinery, rather than explicit per-tensor casts (boilerplate) or a global flag (no mixed-eval story). Success bar: convergence parity with the f32 capstone run within noise; speedup published (soft ≥1.3x over malus f32; informational, not a gate).

## Why this is surprising

Two orderings look wrong from the outside. First, v4-plan.md named f16/bf16 "the first post-V4 perf milestone," yet V5 spends five milestones on dispatch architecture before touching precision. That note predates the measured benchmark: the 60x gap is synchronization stalls, not arithmetic width — halving element size while every matmul still blocks the CPU would be tuning the engine of a parked car. Second, the V5 gate (M35) is f32-vs-f32 even though PyTorch users train bf16 by default on Apple Silicon; that is deliberate, so the gate measures malus's architecture against PyTorch's rather than mixing a precision migration into the comparison. M36 then publishes the bf16-vs-bf16 number separately.

## Considered alternatives

**f16 first.** Better hardware ubiquity historically, but requires loss scaling (narrow exponent range under/overflows gradients), which is a training-loop mechanism malus would have to design and users would have to reason about. bf16 needs none. Rejected.

**Pure-bf16 (params in bf16 too).** Smallest memory footprint, no master copy. Rejected: optimizer math (small `lr*m_hat/...` updates) loses precision in bf16 accumulation, and PyTorch's own MPS autocast keeps f32 masters — matching that contract keeps the M36 benchmark apples-to-apples (ADR-0022).

**Skip mixed precision until V6 flash attention needs it.** Rejected by user decision (2026-07-01): the v4-plan promise is honored inside V5, and flash attention (V6) then builds on an already-proven bf16 substrate rather than landing both at once.

## Consequences

- The dtype-tag plumbing (Bf16=2 already flows through `ScalarTy`/`Dtype`) stops panicking only on the autocast-covered paths; all other non-f32 dtypes keep the ADR-0006 panic behavior.
- Kernel MSL emission gains element-type parameterization (`float`/`bfloat`), which is also the prerequisite infrastructure for V6's `simdgroup_matrix` flash attention.
- Conversion kernels (f32↔bf16) join the stdlib and count as GPU work under the CPU-compute gate.
- The M35 f32 benchmark result remains the canonical "architecture" number; M36's bf16 number is reported alongside, never replacing it.
