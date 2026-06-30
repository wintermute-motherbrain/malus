# ADR-0031 — Demo-Gated CI Asserts as V4 Milestone Gates

**Status:** Accepted (V4)

## Context

V1–V3 used spec-checkbox milestones: "implement X, add a test that X is syntactically expressible." This allowed the hard part (GPU execution, real kernel algorithms, ergonomic abstractions) to be deferred while still "completing" the milestone. The V3 capstone runs almost entirely on the CPU despite claiming GPU-native compute.

## Decision

V4 milestones are gated by **CI asserts against the running nanoGPT demo**. A milestone is complete when a hard assertion passes, not when an implementation exists.

**The canonical gate: `malus_cpu_compute_count() == 0` over a full nanoGPT train step.**

This is structurally impossible to stub. If CPU arithmetic is invoked during the hot path, the assert fails. There is no way to "defer" it.

**Implementation of the CPU-compute counter:**
- `AtomicI64 CPU_COMPUTE_CALLS` in `malus-runtime`.
- `inc()` at the entry of each CPU arithmetic function: `softmax_axis_cpu`, `tensor_layernorm_axis`, `tensor_gelu`, `tensor_cross_entropy`, `tensor_embedding`, `reduce_*_axis`, backward `elem_*`.
- **Excludes** (by definition): orchestration — tape-walk, dispatch encoding, alloc/free, retain/release, the driver loop in `main`. These run on the CPU and are correct to do so.
- Exported as `malus_cpu_compute_count() -> i64` / `malus_cpu_compute_reset()`.
- Hot-path test via the existing `run_metal_src` harness in `crates/malus-codegen-cpu/tests/metal_integration.rs`.

**Belt-and-suspenders:** CPU arithmetic fns gated behind `#[cfg(feature = "cpu_fallback")]`. The hot-path CI build omits the feature → stray CPU call = link error, not a silent pass.

**Incremental gates per milestone:**
- M2 (forward kernels): `count() == 0` over a nanoGPT forward pass only.
- M3 (backward kernels): `count() == 0` over a full train step (canonical gate).
- M4 (kill Variable): gate from M3 still holds; zero `ResolvedTy::Variable` in typed IR.
- M5 (Module + generic optimizer): no-hand-unrolled-optimizer lint passes (AST check: one generic `fn adamw<M: Module>`, AdamW state appears ≤1×, no `.grad` arithmetic in `main`'s train loop outside the optimizer).
- M6 (borrow-inference): RC-op-count ≤ ~5% of allocations; all prior gates green; nanoGPT within Nx of PyTorch-MPS.

**Real-kernels assert:** after `compile_kernels`, assert `name_to_id` contains `softmax`, `layernorm`, `gelu`, `cross_entropy`, `embedding` — the registry proves these ops dispatch GPU kernels, not CPU builtins.

## Why this prevents the V1–V3 failure mode

The V1–V3 failure mode: a milestone is defined as "implement feature X," completed by "X is expressible in the language," with "X actually runs on GPU" deferred. The counter-assert makes this logically impossible: if the op runs on the CPU, the bit increments, the CI test fails, the milestone is not met. You cannot defer GPU execution while claiming the milestone.

## Consequences

- CI test infrastructure grows to include the hot-path counter test.
- Migration: existing exact-equality goldens that compare CPU sequential-sum results must convert to tolerance asserts (GPU reduction order differs from sequential CPU).
- The counter `inc()` calls are behind a `cfg` feature in release; zero overhead in production builds.
