# M25 — Stdlib Forward Kernels

**Crates:** `malus-runtime`, `malus-sema`, `malus-codegen-cpu`, `malus-codegen-gpu`, `malus-cli`  
**Track:** GPU  
**Depends on:** M24

Replace all Rust CPU-loop forward-pass operations with malus `.ml` kernels in the stdlib. The nanoGPT forward pass must run end-to-end with zero CPU arithmetic. See ADR-0028.

## Done-When

1. `examples/nanogpt.ml` forward pass (logits computed from a batch of tokens, cross-entropy loss computed) completes with `malus_cpu_compute_count() == 0`.
2. Registry assert passes: after `compile_kernels`, `name_to_id` contains entries for `softmax`, `layernorm`, `gelu`, `cross_entropy`, `embedding`, `reduce_sum_axis`, `reduce_mean_axis`, `reduce_var_axis`, `broadcast_add`, `broadcast_mul` (and their variants). These must be kernel IDs, not sentinel CPU-builtin tags.
3. Existing golden tests that depend on exact-equality CPU output are migrated to tolerance asserts (GPU reduction order differs from sequential CPU).
4. `cargo test --workspace` passes.

## Scope

### 1. Stdlib `.ml` kernel files

Author the following in `stdlib/` using the M24 kernel language:

| File | Replaces |
|---|---|
| `stdlib/softmax.ml` | `tensor_softmax_axis_cpu` in metal.rs |
| `stdlib/layernorm.ml` | `tensor_layernorm_axis` in metal.rs |
| `stdlib/gelu.ml` | `tensor_gelu` in metal.rs |
| `stdlib/cross_entropy.ml` | `tensor_cross_entropy` in metal.rs (forward only) |
| `stdlib/embedding.ml` | `tensor_embedding` in metal.rs (forward only) |
| `stdlib/reduce_sum.ml` | `tensor_reduce_sum_axis` in metal.rs |
| `stdlib/reduce_mean.ml` | `tensor_reduce_mean_axis` in metal.rs |
| `stdlib/reduce_max.ml` | `tensor_reduce_max_axis` in metal.rs |
| `stdlib/reduce_var.ml` | `tensor_reduce_var_axis` in metal.rs |
| `stdlib/broadcast_binop.ml` | broadcasting path in binary ops (unequal-shape BinOp) |

`matmul` stays MPS. `causal_mask` is a simple alloc + fill; keep it as a cheap CPU alloc (no arithmetic, no counter increment).

### 2. Retire CPU fns (`malus-runtime/src/metal.rs`)

Gate the above CPU functions behind `#[cfg(feature = "cpu_fallback")]`. The default build (no feature) will fail to link if any of them are called — a link-error backstop that makes silent regressions impossible.

Keep the functions available under the feature for correctness testing: the M25 CI build includes a `cpu_fallback` test that compares GPU kernel output vs. retained CPU output within 1e-5.

### 3. Stdlib loader (`malus-cli/src/main.rs` + `malus-loader`)

The CLI must load and compile `stdlib/*.ml` before compiling user code. Stdlib kernels are prepended to the `KernelRegistry` before user kernels. A new `--stdlib-path` flag (defaulting to the installed `stdlib/` directory) controls where the CLI looks.

For tests: `run_metal_src` in `metal_integration.rs` must include the stdlib kernels. Pass them as a pre-compiled `KernelRegistry` or as source strings prepended to the test program.

### 4. Codegen-cpu: replace CPU-builtin calls with kernel dispatch

When sema encounters a builtin call that now has a kernel implementation (e.g. `softmax(t, axis=2)`), codegen-cpu emits `kernel_dispatch_v2(kernel_id, ...)` instead of the old C-ABI CPU fn call. The `kernel_ids` map (already passed to `compile_and_run`) provides the ID.

This requires a new field in the `RuntimeSymbols` / codegen table: a `builtin_to_kernel_id: HashMap<BuiltinOp, u64>` populated by `compile_kernels` from the stdlib registry.

### 5. Migrate exact-equality tests to tolerance asserts

All tests that assert exact output values from softmax/layernorm/gelu/reduction ops must be converted to `assert_within_tol(actual, expected, 1e-5)`. GPU floating-point reduction order is non-deterministic across threadgroups; sequential CPU order is not a valid reference for GPU output.

The CPU reference values (computed by the `cpu_fallback` functions) ARE still used as the tolerance reference in the `cpu_fallback`-gated correctness tests.

### 6. CI test: forward CPU-counter==0

```rust
#[test]
fn test_nanogpt_forward_zero_cpu_compute() {
    malus_cpu_compute_reset();
    // run one forward pass of the nanogpt model (no backward)
    run_metal_src_forward_only(include_str!("../../../examples/nanogpt.ml"));
    assert_eq!(malus_cpu_compute_count(), 0,
        "CPU arithmetic invoked in nanoGPT forward pass");
}
```

## Out of Scope

- Backward kernels (M26).
- Replacing VJPs in tape.rs (M26).
- The `Variable` type / grad-inference changes (M27).
- `randn` CPU-side Philox — this is not tensor arithmetic (it generates data, does not compute gradients); it stays CPU-side in V4 (marked with a `#[cfg(not(feature="cpu_fallback"))]` exemption and no counter increment, since GPU RNG is post-V4).
