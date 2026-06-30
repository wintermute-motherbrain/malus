# M23 — De-Risk Spike

**Crates:** `malus-runtime`  
**Track:** GPU (first)  
**Depends on:** V3 complete (M22 done)  
**Status:** ✅ done

Validate the two V4-critical infrastructure pieces before committing to the M24 kernel-language rewrite:
(1) the extended `kernel_dispatch_v2` ABI that can carry explicit grid/threadgroup configuration,
an independent output shape/dtype, and scalar uniforms; and (2) the CPU-compute counter CI gate
(ADR-0031) that all subsequent V4 milestone gates rely on.

Proof vehicle: one hand-written MSL `softmax_row` kernel — a genuine threadgroup-per-row reduction
— that exercises shared memory, barriers, and `dispatchThreadgroups`, showing the architecture is
sound. The kernel and all glue are throwaway scaffolding, retired in M24 when the malus compiler
starts generating MSL from `.ml` kernel source.

## Done-When (as built)

1. A Rust integration test dispatches `softmax_row` via `kernel_dispatch_v2` on a `[4, 8]`
   tensor and asserts the output matches a pure-Rust softmax reference within `1e-5`.
2. `malus_cpu_compute_count() == 0` over that dispatch (verified in the same test).
3. `cargo test --workspace` passes.

## Locked Decisions

| # | Decision | Choice |
|---|---|---|
| D1 | Kernel architecture | Real threadgroup reduction — one threadgroup per row, static `threadgroup float scratch[1024]`, `threadgroup_barrier` for max + sum reductions. Serial per-thread loop (from original spec) discarded. |
| D2 | ABI scope | Minimal-but-real: `kernel_dispatch_v2` carries grid/tg config, output shape/dtype, and a scalar uniforms blob. Per-tensor `{shape, strides, ndim}` descriptors deferred to M24. No `tg_mem_bytes` (ADR-0027 mandates static shared-mem). |
| D3 | Validation approach | Option B (runtime-only): Rust integration test calls runtime functions directly. No `.ml` example, no `builtins.rs` entry, no codegen changes, no `RuntimeSymbols` change. |
| D4 | Dispatch semantics | `dispatchThreadgroups_threadsPerThreadgroup`; `grid_dims` = threadgroup counts. Required so one row maps to exactly one threadgroup. Old `kernel_dispatch` keeps `dispatchThreads` (backward-compat). |
| D5 | Benchmark | Deferred to M25. No PyTorch-MPS number exists to race until the forward pass runs entirely on GPU. |
| D6 | Counter completeness | Complete principled set — every Rust function that loops over tensor element values. Spec's subset omitted `tensor_transpose`, `tensor_sum`, `permute_by_perm`, `broadcast_cpu_loop`, `tensor_randn`, `tensor_causal_mask`, `tensor_scatter_add`, `broadcast_to_shape`/`sum_to_shape` (non-identity), `scalar_mul`. Incomplete counter produces false `count()==0` — the exact failure ADR-0031 prevents. |

## Scope (as implemented)

### CPU-compute counter — `crates/malus-runtime/src/lib.rs`

```rust
static CPU_COMPUTE_CALLS: AtomicI64 = AtomicI64::new(0);
pub fn cpu_compute_inc() { CPU_COMPUTE_CALLS.fetch_add(1, Ordering::Relaxed); }
#[no_mangle] pub extern "C" fn malus_cpu_compute_count() -> i64 { ... }
#[no_mangle] pub extern "C" fn malus_cpu_compute_reset() { ... }
```

`cpu_compute_inc()` called at entry of the complete instrumented set in `metal.rs`:
`tensor_transpose`, `tensor_sum`, `broadcast_cpu_loop`, `tensor_reduce_sum/mean/max/var_axis`,
`permute_by_perm`, `softmax_axis_cpu`, `tensor_layernorm_axis`, `tensor_gelu`,
`tensor_cross_entropy`, `tensor_causal_mask`, `tensor_embedding`, `tensor_scatter_add`,
`tensor_randn`, `broadcast_to_shape` (non-identity), `sum_to_shape` (non-identity).

`cpu_compute_inc()` called at entry of all VJP element-loop helpers in `tape.rs`:
`elem_add`, `elem_sub`, `elem_cmp_eq`, `elem_mul`, `elem_div`, `scalar_mul`, `elem_apply`.

**Excluded (orchestration / zero-copy):** `tensor_alloc*`, `tensor_free`, `tensor_retain/release`,
`gpu_barrier`, `kernel_dispatch`, `kernel_dispatch_v2`, `tensor_print`, `tensor_matmul` (MPS),
`tensor_reshape` (zero-copy, ADR-0023), identity-path `broadcast_to_shape`/`sum_to_shape`.

### `kernel_dispatch_v2` — `crates/malus-runtime/src/metal.rs`

New `#[no_mangle] pub extern "C"` symbol after `kernel_dispatch`. Uses
`dispatchThreadgroups_threadsPerThreadgroup`. Allocates output via the existing
`tensor_alloc_gpu(out_dtype_tag, out_shape, out_ndim, null)`. Copies the uniforms blob into
a freshly allocated `MTLBuffer` and binds it at index `handle_count+1`; the buffer drops at
end of the function (Metal encoder retains it until command-buffer completion). Follows the
deferred-commit invariant (encodes but does not commit; `gpu_barrier` flushes).

Full signature: see ADR-0027.

### Hand-written `softmax_row` MSL — `crates/malus-runtime/src/metal.rs`

- `pub const M23_SOFTMAX_ROW_KERNEL_ID: u64 = 0x8000_0000_0000_0001` (high-bit reserved).
- `const SOFTMAX_ROW_MSL: &str` — inline MSL with static `threadgroup float scratch[1024]`.
- `pub fn register_m23_softmax_row_kernel()` — compiles the MSL string, looks up
  `"softmax_row"` by name, creates a `MTLComputePipelineState`, inserts under the reserved id.
  Bypasses `runtime_init`'s `malus_kernel_{id}` naming convention — registered separately.

### CI test — `crates/malus-codegen-cpu/tests/metal_integration.rs`

`test_v4_m23_softmax_row_gpu_counter_zero`:
1. Compute reference via pure-Rust `softmax_ref` (no counter increment).
2. `register_m23_softmax_row_kernel()` + `malus_cpu_compute_reset()`.
3. Allocate `[4,8]` input tensor, dispatch with `grid=[4,1,1]`, `tg=[8,1,1]`, `uniforms={cols=8u32}`.
4. `gpu_barrier()`, snapshot `malus_cpu_compute_count()`.
5. Per-element assertion within `1e-5`; `assert_eq!(cpu_count, 0)`.

## Out of Scope

- Any malus syntax / sema / codegen change.
- `builtins.rs` entry, `RuntimeSymbols` change (those come in M24).
- Per-tensor `{shape, strides, ndim}` uniform descriptors (M24).
- `cpu_fallback` feature gate (M25).
- PyTorch-MPS benchmark baseline (M25).
- Any kernel other than `softmax_row`.
