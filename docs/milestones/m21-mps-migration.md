# M21 — MPS Migration

**Crates:** `malus-runtime`.

Migrate `tensor_matmul`, `tensor_matmul_batched`, and the axis reductions needed by the transformer (`tensor_reduce_sum_axis`, `tensor_reduce_mean_axis`) from eager CPU loops to Metal Performance Shaders / custom Metal kernels. The results become **pending tensors** so CTMM can batch command buffers across chained ops. Amends ADR-0012 and ADR-0017. Everything outside `malus-runtime` is unchanged.

## Done-When

`examples/mps_bench.ml` compiles, produces correct results, and demonstrates speedup:

```malus
fn main():
    let a = randn(512, 512)
    let b = randn(512, 512)
    let c = a @ b
    tensor_print(c)

    let a3 = randn(8, 512, 512)
    let b3 = randn(8, 512, 512)
    let c3 = a3 @ b3
    tensor_print(c3)

    println("MPS matmul: OK")
```

MPS results match the old CPU loop results within 1e-3 (float rounding allowed). On an M-series Mac with 512×512 matrices, MPS matmul runs at least 10× faster than the CPU loop (measured via the CLI timing flag or a Rust benchmark).

Existing VJP unit tests (from M14/M17) still pass — the VJPs call `tensor_matmul` and must produce correct gradients through the MPS path.

## Scope

### 1. `MPSMatrixMultiplication` for 2-D and Batched Matmul

**Runtime (`malus-runtime/src/metal.rs`):** Replace the triple-nested CPU loop in `tensor_matmul` (currently at `:248–282`) with an `MPSMatrixMultiplication` call:

- Create `MPSMatrix` descriptors from `TensorBuffer.shape` and `MTLBuffer`.
- Encode `MPSMatrixMultiplication.encode(commandBuffer:...)` into the current `command_buffer` (do not commit — let `gpu_barrier` commit it).
- Return a new `TensorBuffer` of shape `[M, N]` whose `MTLBuffer` is the MPS output — a **pending tensor** (no `gpu_barrier` called here).

For 3-D batched matmul: use `MPSMatrixMultiplication` in a loop over the batch dimension, encoding one MPS op per batch into the same command buffer. Return a `[B, M, N]` pending tensor.

The removal of the internal `gpu_barrier()` call from `tensor_matmul` (currently at `:249`) is intentional — MPS encodes into the command buffer rather than flushing it. Callers that need a ready tensor must still call `gpu_barrier()` explicitly or rely on CTMM's barrier insertion.

### 2. MPS Axis Reductions

**Runtime (`malus-runtime/src/metal.rs`):** Replace `tensor_reduce_sum_axis` and `tensor_reduce_mean_axis` (added in M16 as CPU loops) with `MPSMatrixVectorMultiplication` or a custom Metal kernel for the common transformer reduction patterns (sum/mean over the last axis of a 2-D or 3-D tensor). Return pending tensors.

A custom Metal kernel is acceptable for reductions if the MPS reduction API is cumbersome for arbitrary axes. The kernel should handle `[N, D]` → `[N, 1]` (axis=1) and `[B, N, D]` → `[B, N, 1]` (axis=2) patterns, which cover the transformer's layernorm and softmax normalization.

### 3. `tensor_transpose` — Optional MPS Path

Optionally migrate 2-D `tensor_transpose` to a Metal kernel (trivially parallelizable). The performance gain is smaller than matmul; only migrate if the implementation is straightforward. The CPU loop path is kept as a fallback.

### 4. VJP Compatibility

The VJPs for `matmul` (M14) and batched matmul (M17) call `tensor_matmul` / `tensor_transpose` internally in their Rust backward closures. Since these now return pending tensors, the backward closures must call `gpu_barrier()` before reading back gradients to CPU for leaf `.grad` accumulation. Update each affected VJP closure to insert the barrier.

Alternatively: leaf `.grad` accumulation can also go through Metal if the accumulate-into-leaf op is itself encoded as a GPU op. This is cleaner but requires more work; either approach is acceptable as long as VJP tests pass.

## Out of Scope

- MPS for softmax, layernorm, GELU, cross-entropy (these are CPU loops in M18/M19; post-V3)
- MPS for `var` / `max` reductions (post-V3)
- Metal Performance Shaders Graph (MPSGraph) for kernel fusion (post-V3)
- Non-f32 MPS dispatch (post-V3)
