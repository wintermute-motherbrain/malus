# M5 — GPU Codegen

**Crate:** `malus-codegen-gpu` (generates MSL); `malus-runtime` (compiles and dispatches it)
**Done when:** A malus `kernel add` is compiled to MSL, loaded into a `MTLComputePipelineState`, dispatched over four elements, and the output buffer contains the correct sum.

## Scope

### MSL code generation (`malus-codegen-gpu`)

Walk the `TypedProgram`'s `kernel` items and emit MSL source as a `String`.

**Lowering rules for element-wise kernels:**

| malus | MSL |
|---|---|
| `kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>` | `kernel void malus_add(device float* a, device float* b, device float* out, uint tid [[thread_position_in_grid]])` |
| `return a + b` (element-wise, detected by tensor type) | `out[tid] = a[tid] + b[tid];` |
| `Dtype::F32` | `float` |
| `Dtype::F16` | `half` |
| `Dtype::BF16` | `bfloat` (Metal 3+, M-series) |
| Integer dtypes | `int`, `uint`, `short`, `ushort`, etc. |

**Element-wise detection rule:** A binary op on two tensors of the same shape with no explicit thread indexing is lowered as element-wise. The output buffer is the same size as the inputs. Thread ID is implicit (`thread_position_in_grid`).

**Generated MSL for the MVP demo:**

```metal
#include <metal_stdlib>
using namespace metal;

kernel void malus_add(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = a[tid] + b[tid];
}
```

### Kernel registry

`malus-codegen-gpu` produces a `KernelRegistry`: a map from `kernel_id: u64` → `msl_source: String`. This is passed to `malus-runtime` at startup so it can compile all kernels before execution begins.

### MSL compilation and dispatch (`malus-runtime`)

Extend `malus-runtime` to:

1. **Compile MSL at startup** — for each entry in the `KernelRegistry`:
   - `[device newLibraryWithSource:options:error:]` → `MTLLibrary`
   - `[library newFunctionWithName:]` → `MTLFunction`
   - `[device newComputePipelineStateWithFunction:error:]` → `MTLComputePipelineState`
   - Cache `MTLComputePipelineState` by `kernel_id`

2. **Implement `kernel_dispatch`** (replaces the M4 stub wholesale):
   - **ABI migration:** M4 preserved the M3 ABI (`name: *const u8, handles: *const i64, n: i32`). M5 migrates to `kernel_id: u64, handles: *const i64, count: usize` now that the `KernelRegistry` makes a `u64` id meaningful. This requires updating `malus-codegen-cpu`'s `RuntimeSymbols` struct and the `KernelCall` IR emission.
   - Allocate output buffer: `tensor_alloc_gpu(dtype, len, null)`
   - Encode a compute pass:
     - `[commandBuffer computeCommandEncoder]`
     - `setComputePipelineState:` for the kernel
     - `setBuffer:offset:atIndex:` for each input and the output
     - `dispatchThreads:threadsPerThreadgroup:` — use `MTLSizeMake(len, 1, 1)` and threadgroup size `MTLSizeMake(min(len, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)`
   - `endEncoding`, do NOT commit yet (commit happens in `gpu_barrier`)
   - Return the output tensor handle

## Tests

- Generate MSL for `kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> { return a + b }` → matches expected string
- Compile the generated MSL via Metal → no compilation errors
- Dispatch over `[1.0, 2.0, 3.0, 4.0]` + `[5.0, 6.0, 7.0, 8.0]` → output buffer contains `[6.0, 8.0, 10.0, 12.0]`
- Dispatch followed by `gpu_barrier` then `tensor_print` → correct output printed

## Out of scope for M5

- Kernels with explicit thread indexing (intrinsics) — deferred to v1
- `@threadgroup_size` annotations — deferred to v1
- `inout` parameters — deferred to v1
- Non-element-wise kernels (reductions, matmul) — deferred to v1 stdlib
- Multi-kernel pipelines / kernel fusion — future optimization
