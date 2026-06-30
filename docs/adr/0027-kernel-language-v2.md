# ADR-0027 — Kernel Language v2: Real GPU Programming Model

**Status:** Accepted (V4-M1)  
**Amends:** ADR-0005 (MPS for stdlib — inverted), ADR-0012/0017 (eager CPU loops — retired for stdlib ops)

## Context

The V1/V2/V3 `kernel` language expresses only `out[tid] = f(inputs[tid])` — a per-thread elementwise map. The `lower_kernel_body`/`lower_expr` pass in `malus-codegen-gpu/src/lib.rs` rejects all `TypedStmt` variants except `LetBind`/`Return` and all indexing that isn't implicit `a[tid]`. This means:

- No loops, conditionals, or reductions inside kernels.
- No shared memory, no threadgroup synchronization, no barrier instructions.
- No arbitrary tensor indexing (`a[i,j]`, `a[row*stride+col]`).
- Cannot express softmax, layernorm, attention, or any real GPU algorithm.

All transformer ops are consequently implemented as Rust CPU loops in `malus-runtime/src/metal.rs`, defeating the GPU-first design.

## Decision

V4-M1 rewrites the `kernel` codegen to a real GPU programming model. The kernel language gains:

**Thread hierarchy intrinsics:** `thread_id()`, `threadgroup_id()`, `threads_per_threadgroup()`, `threads_per_grid()` → MSL `thread_position_in_grid`, etc.

**Arbitrary tensor indexing:** `a[i]` and `a[i,j]` (and higher rank) → MSL buffer pointer arithmetic with stride/shape metadata passed as uniforms. The dispatch ABI is extended to pass per-tensor `{shape, strides, ndim}` as uniform buffers alongside the data handle (see CI-assert mechanism in the plan).

**Control flow inside kernels:** `for`/`while`/`if` — the `TypedStmt` variants already exist in the IR; the GPU codegen simply needs to lower them instead of rejecting them.

**Shared memory:** `let shared x: SharedArray<f32, N>` → MSL `threadgroup float x[N]`. Size `N` must be a compile-time literal (static shared-mem sizing).

**Barrier:** `barrier()` → MSL `threadgroup_barrier(mem_flags::mem_threadgroup)`.

**Reductions:** threadgroup/tree reductions expressed as explicit loops using the above primitives.

**Extended dispatch ABI:** `kernel_dispatch` is extended (starting in M0) to accept:
- Grid/threadgroup configuration (not inferred from input[0] shape alone).
- Per-tensor uniforms: `{data_ptr, shape_ptr, ndim, strides_ptr}`.
- Scalar uniforms (f32, i32) for parameters like `axis`, `eps`, `T`.

## Why this is the M1 rewrite

This is not additive — `lower_kernel_body` and `lower_expr` must be rewritten, not patched. The dispatch ABI change is backward-compatible (old elementwise kernels can use the new ABI with simplified uniforms), so existing user kernels continue to work.

## Consequences

- `malus-syntax` / `malus-sema`: kernel grammar is extended to accept the new intrinsics, `SharedArray`, and `barrier()`. Sema validates shared-mem size literals.
- `malus-codegen-gpu/src/lib.rs`: `lower_kernel_body` rewrite; `lower_expr` extended for index expressions and intrinsics.
- `malus-runtime/src/metal.rs`: `kernel_dispatch` ABI extended (shape/stride/scalar uniforms). The `RuntimeSymbols` struct gains new fields for the extended dispatch.
- All existing elementwise kernels (built-in `malus_add`, etc.) continue to work unchanged.
- M0 de-risk spike: one hand-written MSL softmax kernel is dispatched through the extended ABI to validate the design before the full M1 rewrite.
