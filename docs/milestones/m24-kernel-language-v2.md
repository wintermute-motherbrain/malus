# M24 — Kernel Language v2

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-gpu`, `malus-runtime`  
**Track:** GPU  
**Depends on:** M23  
**Status:** ✅ Done

Rewrite the kernel language from "elementwise map only" to a real GPU programming model:
thread/threadgroup/grid hierarchy intrinsics, flat 1-D tensor indexing, shared memory,
`barrier()`, and control flow (`if`/`for`/`while`).  Validated by authoring
`softmax`, `layernorm`, and `gelu` as standalone `.ml` kernel files in
`examples/v4-kernels/`.  See ADR-0027.

## Done-When

1. `examples/v4-kernels/softmax.ml`, `layernorm.ml`, and `gelu.ml` exist as
   kernel-language source files that compile and produce output matching the CPU
   references within 1e-5. ✅
2. `malus_cpu_compute_count() == 0` over a dispatch of each of those three kernels
   (integration test `test_v4_m24_{softmax,layernorm,gelu}_gpu_counter_zero`). ✅
3. A codegen-gpu unit test (`test_m24_explicit_kernel_emits_msl`) exercises a kernel
   body with `let shared`, `barrier()`, a `for` loop, `threadgroup_id()`,
   `thread_in_threadgroup()`, flat indexing, and scalar uniforms — and asserts the
   emitted MSL contains all expected constructs. ✅
4. `cargo test --workspace` passes. ✅

## What was built

### Explicit vs implicit-map kernels

M24 introduces the **explicit / implicit-map** distinction.  The predicate
`is_implicit_map_kernel(body)` returns true if and only if the body contains only
`let`/`let mut` bindings and a final `return scalar_expr` — the legacy elementwise
sugar form.  Any other statement (`if`, `for`, `while`, `let shared`, `out[i]=…`,
or a call to a thread intrinsic) makes the body **explicit**.

- **Implicit-map**: tensor params rebound to scalar element type (old behaviour);
  `return expr` → `out[tid] = expr`.  Fully backward compatible.
- **Explicit**: tensor params stay as `Tensor<dtype>` (indexable pointers in MSL);
  output written via implicit `out` binding (`out[expr] = val`); bare `return`
  is an early exit.

### Syntax

`StmtKind::LetShared { name, elem_ty, size }` — new AST node; no initializer.
`shared` is a **contextual keyword** (not a reserved word; parses as `Ident("shared")`
everywhere except after `let`), so existing identifiers named `shared` still parse.

### Sema additions

- `KernelOnly` builtin kind: thread intrinsics (`thread_id`, `threadgroup_id`,
  `thread_in_threadgroup`, `threads_per_threadgroup`, `threads_per_grid` → `i32`;
  `barrier()` → `Unit`); `fmax`, `fmin` (→ `Scalar(F32)`); `rsqrt` (→ `Scalar(F32)`).
  All raise `KernelIntrinsicOutsideKernel` if called from a `fn` body.
- Scalar-math pass-through: `Fixed` builtins (`exp`, `log`, `sqrt`, `tanh`, `abs`, …)
  invoked with all-scalar args inside a kernel body return `Scalar(dtype)` instead of
  `Tensor<dtype>` (return-type override, no arg type-check change).
- In explicit kernels, the `for` loop variable is typed `Scalar(I32)` (not the default
  `I64`).  Comparisons in kernel bodies return the operand's scalar type (not `Bool`).
- `LetShared` binds `Array<Scalar(T), N>` as mutable so `scratch[i] = …` works.
- `out` bound as mutable `Tensor<return_ty>` in explicit kernels.

### Codegen-gpu additions

- `lower_kernel_explicit` / `lower_kernel_body_explicit` / `lower_expr_kernel`:
  the new MSL generation path for explicit kernels.
- Scalar params → `struct Uniforms_N { … }` at `buffer(handle_count+1)`, accessed
  as `u.field`.
- Thread intrinsics → `uint _var [[attribute]]` params injected only if actually
  used by the body (`collect_used_intrinsics` scan).
- `barrier()` → `threadgroup_barrier(mem_flags::mem_threadgroup)`.
- `let shared` → `threadgroup T name[N]`.
- `For` loop → `for(long var = start; var < end; var++)` (integer literals default to
  `I64` in sema; MSL handles the implicit promotion against `int` uniforms safely).
- `if`/`while`/`Return` → standard MSL.

### Runtime

M23 spike (`SOFTMAX_ROW_MSL`, `register_m23_softmax_row_kernel`,
`M23_SOFTMAX_ROW_KERNEL_ID`) retired; replaced by a comment noting the retirement.
No ABI change — `kernel_dispatch_v2` from M23 is the dispatch path for M24 kernels.

## Design amendments vs pre-implementation spec

| Spec item | Implementation decision |
|---|---|
| Multi-dim `a[i,j]` indexing + `TensorMeta` strides | **Deferred to M25** with launch-config (both require runtime shape info).  M24 uses flat 1-D indexing `a[row*cols+col]` + scalar uniforms. |
| `SharedArray<T, N>` as a new type | Reused existing `Array<T, N>` with `shared` storage qualifier (`let shared`).  No new type node; same indexing semantics. |
| Done-when #3: multi-dim indexing + `xcrun metal` compile | Changed to flat-indexing assertion test (no `xcrun` in CI). |
| Kernel files in `stdlib/` | Located in `examples/v4-kernels/` for M24 (validation artifacts); move to `stdlib/` at M25 when they replace CPU ops. |

## Out of Scope (deferred)

- Backward kernels (M26).
- Replacing stdlib CPU fns with these kernels (M25).
- 2-D/3-D threadgroup intrinsics (`.xy`, `.xyz`) — x-axis only in M24.
- Multi-dim `a[i,j]` indexing + per-tensor `TensorMeta` strides (M25).
- Launch-config syntax / `RuntimeSymbols` wiring for `kernel_dispatch_v2` (M25).
- `inout` kernel parameters.
- SIMD-group operations.
