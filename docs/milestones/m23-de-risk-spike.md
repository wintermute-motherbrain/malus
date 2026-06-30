# M23 — De-Risk Spike

**Crates:** `malus-runtime`  
**Track:** GPU (first)  
**Depends on:** V3 complete (M22 done)

Validate the two V4-critical infrastructure pieces before committing to the M24 full rewrite: (1) the extended `kernel_dispatch` ABI that can pass shape/stride/scalar uniforms, and (2) the CPU-compute counter CI gate. Do this with one hand-written MSL kernel — a row-softmax — to prove the target architecture is sound.

## Done-When

1. `examples/m23-softmax.ml` calls a kernel named `softmax_row` on a `[4, 8]` tensor; the output matches `softmax_axis_cpu` within 1e-5.
2. `malus_cpu_compute_count() == 0` over that dispatch (verified in a test).
3. `cargo test --workspace` passes.

## Scope

### 1. CPU-compute counter (`malus-runtime/src/lib.rs`)

Add a process-global counter:

```rust
static CPU_COMPUTE_CALLS: AtomicI64 = AtomicI64::new(0);

pub fn cpu_compute_inc() { CPU_COMPUTE_CALLS.fetch_add(1, Ordering::Relaxed); }

#[no_mangle] pub extern "C" fn malus_cpu_compute_count() -> i64 { CPU_COMPUTE_CALLS.load(Ordering::SeqCst) }
#[no_mangle] pub extern "C" fn malus_cpu_compute_reset() { CPU_COMPUTE_CALLS.store(0, Ordering::SeqCst) }
```

Call `cpu_compute_inc()` at the entry of each CPU arithmetic function: `softmax_axis_cpu`, `tensor_layernorm_axis`, `tensor_gelu`, `tensor_cross_entropy`, `tensor_embedding`, `reduce_sum_axis`, `reduce_mean_axis`, `reduce_max_axis`, `reduce_var_axis`, and all backward `elem_*` functions in `tape.rs`. **Does not increment for:** tensor_alloc*, tensor_free, tensor_retain, tensor_release, gpu_barrier, kernel_dispatch, tensor_print, matmul (MPS), read_file. Those are orchestration, not compute.

Add to `RuntimeSymbols`: function pointers for `malus_cpu_compute_count` and `malus_cpu_compute_reset` so the JIT can call them from malus source (useful for test assertions).

### 2. Extended `kernel_dispatch` ABI (`malus-runtime/src/metal.rs`)

Current signature (effectively):
```c
i64 kernel_dispatch(u64 kernel_id, const i64* handles, usize count)
```
Output buffer shape is inferred from `inputs[0]` shape. No way to pass grid/threadgroup config, multi-dimensional shapes, strides, or scalar parameters. This blocks any real GPU algorithm.

New signature:
```c
i64 kernel_dispatch_v2(
    u64 kernel_id,
    const i64* handles,           // tensor handles (input[0..n-1], output slot = n)
    usize handle_count,
    const usize* grid_dims,       // [grid_x, grid_y, grid_z]
    const usize* tg_dims,         // [tg_x, tg_y, tg_z]
    const usize* out_shape,       // shape of the output tensor to allocate
    usize out_ndim,
    i32 out_dtype_tag,
    const void* uniforms,         // opaque blob of scalar uniforms (f32/i32 values)
    usize uniforms_bytes
)
```

Keep the old `kernel_dispatch` working for existing elementwise kernels (backward compat). Add `kernel_dispatch_v2` as a new C-ABI symbol in `RuntimeSymbols`.

The implementation allocates the output tensor from `out_shape`/`out_dtype_tag`, encodes a compute pass with the extended uniform buffer, and dispatches with the provided grid/threadgroup config.

### 3. Hand-written MSL softmax kernel (inline in `malus-runtime/src/metal.rs`)

Write one hard-coded MSL source string for `softmax_row`:

```metal
kernel void softmax_row(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      cols   [[buffer(2)]],   // uniform: number of columns
    uint row [[thread_position_in_grid]]
) {
    uint base = row * cols;
    float m = input[base];
    for (uint j = 1; j < cols; j++) m = max(m, input[base + j]);
    float s = 0.0;
    for (uint j = 0; j < cols; j++) s += exp(input[base + j] - m);
    for (uint j = 0; j < cols; j++) output[base + j] = exp(input[base + j] - m) / s;
}
```

Register this kernel in `runtime_init` under name `"softmax_row"`. Expose its `kernel_id` in the `name_to_id` map returned by `compile_kernels`.

### 4. `examples/m23-softmax.ml`

```malus
fn main():
    let x = Tensor.gpu<f32>([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                               [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]])
    let y = softmax_row_kernel(x)
    println(y)
```

This calls `kernel_dispatch_v2` under the hood (grid = [4,1,1], tg = [1,1,1], uniforms = cols=8).

### 5. CI test (`crates/malus-codegen-cpu/tests/metal_integration.rs`)

```rust
#[test]
fn test_v4_m0_cpu_counter_zero() {
    // run the m0 softmax example, assert CPU compute counter is 0
    malus_cpu_compute_reset();
    run_metal_src(include_str!("../../../examples/m23-softmax.ml"));
    assert_eq!(malus_cpu_compute_count(), 0, "CPU compute was invoked on the hot path");
}
```

## Out of Scope

- Malus syntax changes (the softmax kernel is registered by hand, not compiled from `.ml` kernel source).
- Shape/stride uniform structs in codegen-gpu (that's M24).
- Any other kernel.
- The `cpu_fallback` feature gate (add that at M25 when retiring the CPU fns).
