# ADR-0027 — Kernel Language v2: Real GPU Programming Model

**Status:** Accepted (V4-M1); M24 implemented  
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

**Arbitrary tensor indexing:** `a[expr]` (flat 1-D) in M24; `a[i,j]` multi-dim with stride/shape metadata deferred to M25 (requires both launch-config and per-tensor `TensorMeta` strides — same runtime-shape concern, designed together). M24 kernels use flat manual index arithmetic (`a[row*cols+col]`) plus scalar uniform params (`cols: i32`).

**Control flow inside kernels:** `for`/`while`/`if` — the `TypedStmt` variants already exist in the IR; the GPU codegen simply needs to lower them instead of rejecting them.

**Shared memory:** `let shared x: Array<f32, N>` (reusing the existing `Array<T,N>` type with the `shared` storage qualifier) → MSL `threadgroup float x[N]`. Size `N` must be a compile-time literal (static shared-mem sizing). `shared` is a contextual keyword, not a reserved word. `SharedArray` was considered but rejected — a separate type per address space conflicts with the Mojo-inspired model where address space is a parameter on the same array abstraction.

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

**Per-tensor shape/stride uniform descriptors:** `{shape_ptr, ndim, strides_ptr}` for
arbitrary `a[i,j]` indexing are deferred to M25 (alongside launch-config syntax, which
requires the same per-shape runtime information).

**Scalar uniforms (f32, i32):** codegen-gpu emits a `struct Uniforms_N { … }` whose fields
match kernel scalar params in declaration order.  Bound at `buffer(handle_count+1)` and
accessed as `u.field_name` in the MSL kernel.  The host packs an equivalent `#[repr(C)]`
struct and passes it as the `uniforms` pointer to `kernel_dispatch_v2`.

## Why this is the M1 rewrite

This is not additive — `lower_kernel_body` and `lower_expr` must be rewritten, not patched. The dispatch ABI change is backward-compatible (old elementwise kernels can use the new ABI with simplified uniforms), so existing user kernels continue to work.

## Consequences

- `malus-syntax`: `StmtKind::LetShared { name, elem_ty, size }` added; `shared` made
  a contextual keyword (not reserved, only special after `let`).
- `malus-sema`: new `KernelOnly` builtin kind for thread intrinsics, `barrier()`, and
  scalar-math helpers (`fmax`, `fmin`, `rsqrt`). Scalar-math pass-through for `Fixed`
  builtins in kernel bodies.  `is_implicit_map_kernel` predicate drives param binding.
- `malus-codegen-gpu/src/lib.rs`: `lower_kernel_explicit` / `lower_kernel_body_explicit` /
  `lower_expr_kernel` added alongside the preserved implicit-map path.  `collect_used_intrinsics`
  injects only the thread-position attributes actually referenced by the body.
- `malus-runtime/src/metal.rs`: M23 spike (`SOFTMAX_ROW_MSL`, `register_m23_softmax_row_kernel`,
  `M23_SOFTMAX_ROW_KERNEL_ID`) retired in M24.  `kernel_dispatch_v2` ABI unchanged.
- `RuntimeSymbols` **not** extended in M24 — the codegen-cpu call-site emission for
  `kernel_dispatch_v2` is deferred to M25 along with launch-config.
- All existing elementwise kernels (built-in `malus_add`, etc.) continue to work unchanged.
- M23 de-risk spike (hand-written MSL `softmax_row` registered under
  `M23_SOFTMAX_ROW_KERNEL_ID = 0x8000_0000_0000_0001`) retired in M24 as intended.
