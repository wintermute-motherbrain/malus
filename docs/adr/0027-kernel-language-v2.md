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

**Extended dispatch ABI:** `kernel_dispatch_v2` (new symbol added in M23, distinct from the
existing `kernel_dispatch`) has the following signature:

```c
i64 kernel_dispatch_v2(
    u64 kernel_id,
    const i64*   handles,        // input tensor handles [0..handle_count-1]
    usize        handle_count,
    const usize* grid_dims,      // [gx,gy,gz] = threadgroup counts → dispatchThreadgroups
    const usize* tg_dims,        // [tx,ty,tz] = threads per threadgroup
    const usize* out_shape,      // output shape (not inferred from inputs[0])
    usize        out_ndim,
    i32          out_dtype_tag,
    const void*  uniforms,       // opaque scalar blob; bound at buffer index handle_count+1
    usize        uniforms_bytes
) -> i64;                        // output tensor handle
```

Buffer binding convention: inputs at `0..handle_count-1`, output at `handle_count`, uniforms
blob at `handle_count+1`. Uses `dispatchThreadgroups_threadsPerThreadgroup` so `grid_dims`
are threadgroup counts (not total thread counts). This is required for shared-memory
reductions where one threadgroup must own exactly one row.

Compared to `kernel_dispatch` (which infers output shape from `inputs[0]` and uses
`dispatchThreads`), `kernel_dispatch_v2` is fully explicit. Old elementwise kernels continue
to use `kernel_dispatch` unchanged — no backward-compat break.

**Static shared memory:** ADR-0027 mandates `threadgroup float scratch[N]` with `N` a
compile-time literal in the MSL source. The host never calls `setThreadgroupMemoryLength`.
No `tg_mem_bytes` in the ABI.

**Per-tensor shape/stride uniform descriptors:** `{data_ptr, shape_ptr, ndim, strides_ptr}`
for arbitrary `a[i,j]` indexing are deferred to M24. M23's softmax kernel needs only a
scalar `cols` uniform, so building stride infra now would ship un-exercised ABI.

**Scalar uniforms (f32, i32):** Passed as an opaque `const void* uniforms` blob bound as
one buffer at `buffer(handle_count+1)`. Each kernel MSL function declares the relevant
fields as a `constant uint&`, `constant float&`, or similar at that index. M24 will formalize
a struct layout for multi-field uniform blobs.

## Why this is the M1 rewrite

This is not additive — `lower_kernel_body` and `lower_expr` must be rewritten, not patched. The dispatch ABI change is backward-compatible (old elementwise kernels can use the new ABI with simplified uniforms), so existing user kernels continue to work.

## Consequences

- `malus-syntax` / `malus-sema`: kernel grammar is extended to accept the new intrinsics, `SharedArray`, and `barrier()`. Sema validates shared-mem size literals.
- `malus-codegen-gpu/src/lib.rs`: `lower_kernel_body` rewrite; `lower_expr` extended for index expressions and intrinsics.
- `malus-runtime/src/metal.rs`: `kernel_dispatch` ABI extended (shape/stride/scalar uniforms). The `RuntimeSymbols` struct gains new fields for the extended dispatch.
- All existing elementwise kernels (built-in `malus_add`, etc.) continue to work unchanged.
- M23 de-risk spike: one hand-written MSL `softmax_row` kernel is dispatched through
  `kernel_dispatch_v2` to validate the ABI and CPU-compute counter before the M24 rewrite.
  The kernel is registered under `M23_SOFTMAX_ROW_KERNEL_ID = 0x8000_0000_0000_0001`
  (high-bit reserved; cannot collide with sequential IDs from `compile_kernels`). Retired
  in M24 when the malus compiler generates the equivalent MSL from a `.ml` kernel source.
  `kernel_dispatch_v2` is NOT added to `RuntimeSymbols` in M23 — that happens in M24
  when codegen first emits a call to it.
