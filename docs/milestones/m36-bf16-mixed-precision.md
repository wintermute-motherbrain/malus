# M36 — Mixed Precision (bf16-first)

**Crates:** `malus-codegen-gpu`, `malus-runtime`, `malus-sema`, `malus-stdlib`
**Track:** perf (post-gate)
**Depends on:** M35 (the f32 gate must already be passed — this milestone is additive speed, not gate rescue)
**Status:** planned — scope to be finalized in ADR-0037 at implementation start

The v4-plan named f16/bf16 "the first post-V4 perf milestone"; V5 honors that after earning the f32 claim. bf16 first because it needs no loss scaling (same exponent range as f32) and MSL has native `bfloat` on Metal 3.1+/M2+ (target machine is M4 Max).

## Done-When

1. `bf16` is a real compute dtype: `Tensor<bf16>` tensors allocate, dispatch through malus kernels emitting MSL `bfloat` element types, and multiply via MPS bf16 matmul. The dtype tag (Bf16=2) already exists end-to-end; what changes is that it stops panicking.
2. Autocast-style mixed-precision training on the capstone: parameters and optimizer state stay f32; matmuls and forward elementwise kernels compute in bf16; reductions (softmax/layernorm/cross-entropy accumulations) accumulate in f32. The precise op-level policy is ADR-0037's to fix.
3. Convergence parity: capstone loss curve in bf16 mode matches the f32 run within run-to-run noise; samples equally Shakespeare-ish.
4. Speedup published vs malus-f32 and vs PyTorch-MPS bf16-autocast at the same config. Soft target ≥1.3x over malus f32 — informational, not a release gate.
5. All standing gates green; `cargo test --workspace` passes.

## Scope (outline — ADR-0037 finalizes)

- Kernel codegen: parameterize element type in MSL emission (`float` → `bfloat`) for the stdlib kernels on the autocast list; cast at kernel boundaries where accumulation must stay f32.
- Runtime: bf16 MPS matrix descriptors; conversion kernels f32↔bf16 for cast points; `Dtype::Bf16` un-panicked on the covered paths only (uncovered dtype paths keep panicking per ADR-0006).
- Sema: how autocast is expressed — compiler-inserted (a `with autocast:` scope mirroring `no_grad`) vs explicit user casts. Recommendation going into ADR-0037: scoped `autocast` block for PyTorch familiarity; explicit casts rejected as boilerplate. Decide there.

## Out of Scope

- f16 + loss scaling (deferred; bf16 makes it unnecessary for the capstone).
- Flash attention / `simdgroup_matrix` (V6 — natural companion, consciously sequenced after).
- bf16 storage for optimizer state or gradients (full-f32 master path only).
- General non-f32 dtype completeness (i8/u8/etc. still panic).
