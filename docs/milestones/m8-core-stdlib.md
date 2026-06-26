# M8 — Core Stdlib

**Crates:** `malus-runtime`, `malus-sema`, `malus-codegen-gpu`, `malus-codegen-cpu`

matmul, activation functions, and shape ops. Turn malus from "a tensor calculator" into something that can do linear algebra.

## Done-When

```malus
fn forward(x: Tensor<f32>, w1: Tensor<f32>, w2: Tensor<f32>) -> Tensor<f32>:
    let h = relu(x @ w1)
    return h @ w2

fn main():
    let x = ones(2, 3)
    let w1 = ones(3, 4)
    let w2 = ones(4, 2)
    let out = forward(x, w1, w2)
    println("forward output: {}", out)
    let s = sum(out)
    println("sum: {}", s)
    let wt = transpose(w1)
    println("transpose done")
    println("exp result: {}", exp(x))
```

## Scope

### 1. TensorBuffer Shape Metadata (`malus-runtime/src/metal.rs`)

`TensorBuffer` currently holds `{ buffer, dtype, len }`. Add `shape: Vec<usize>` (the full n-dimensional shape; `len` equals `shape.iter().product()`).

Change `tensor_alloc_gpu` to accept shape information:
```c
i64 tensor_alloc_gpu(i32 dtype_tag, const usize* shape_ptr, usize ndims, const float* data)
```

All existing call sites (codegen-cpu's `TensorLiteral` lowering) must pass a 1D shape matching the element count. Update `RuntimeSymbols` accordingly. This is the most structurally invasive change in M8 and must be done first.

### 2. zeros / ones (`malus-runtime/src/metal.rs`, `malus-codegen-cpu/src/lib.rs`)

`zeros` and `ones` are already registered in `malus-sema/src/builtins.rs` with `BuiltinKind::ShapeArgs` and return `Tensor<f32>`. Codegen currently returns `UnsupportedExpr`.

Add to the runtime C ABI:
```c
i64 tensor_alloc_zeros_gpu(const usize* shape_ptr, usize ndims)
i64 tensor_alloc_ones_gpu(const usize* shape_ptr, usize ndims)
```

In codegen-cpu, lower `Call { callee: "zeros", args: [d0, d1, ...] }` by packing shape dims onto a stack slot and calling `tensor_alloc_zeros_gpu`. Shape arg dims are `i64` scalars in the JIT; store as `usize` for the runtime call.

### 3. Matmul via `@` (`malus-runtime/src/metal.rs`, `malus-codegen-cpu/src/lib.rs`)

`BinOp::Matmul` already exists in the AST. Sema allows it. Codegen-cpu currently errors with "Matmul not supported in host fns."

Add to the runtime C ABI:
```c
i64 tensor_matmul(i64 handle_a, i64 handle_b)
```

Implementation options (in order of preference):
1. `MPSMatrixMultiplication` — use MPS `NDArray` descriptors. Requires the input TensorBuffers to have 2D shape.
2. Naive MSL kernel (fallback if MPS API is painful) — a simple `[M,K] x [K,N] -> [M,N]` kernel with a 1D grid over M×N output elements.

In codegen-cpu, lower `BinOp { op: Matmul, lhs: Tensor, rhs: Tensor }` as a `tensor_matmul` runtime call. Add `tensor_matmul` to `RuntimeSymbols`.

Output shape: `[lhs.shape[0], rhs.shape[1]]`.

### 4. Unary Math Builtins (`malus-sema/src/builtins.rs`, `malus-codegen-gpu/src/lib.rs`, `malus-codegen-cpu/src/lib.rs`)

Register `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `abs` in `builtins.rs` with `BuiltinKind::Fixed(vec![ResolvedTy::Tensor { dtype: ScalarTy::F32 }])` returning `Tensor<f32>`.

In `malus-codegen-gpu`, add `synthesize_unary_builtin(name, msl_fn)` following the M5.1 pattern from `elementwise_builtin_name`. Each becomes a built-in element-wise kernel:
```msl
kernel void malus_relu(
    device float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = fmax(0.0, a[tid]);
}
```

MSL function mappings:
- `relu` → `fmax(0.0, x)`
- `sigmoid` → `1.0 / (1.0 + exp(-x))`
- `tanh` → `tanh(x)`
- `exp` → `exp(x)`
- `log` → `log(x)`
- `sqrt` → `sqrt(x)`
- `abs` → `fabs(x)`

Built-in kernel IDs are appended after user kernels and scalar-broadcast built-ins (ADR-0010).

In codegen-cpu, lower builtin `Call` for these as kernel dispatches, the same way as `KernelCall`.

### 5. transpose (`malus-runtime/src/metal.rs`, `malus-codegen-cpu/src/lib.rs`)

Register `transpose` in `builtins.rs` with one tensor argument, returning a tensor.

Add to the C ABI:
```c
i64 tensor_transpose(i64 handle)
```

For a 2D tensor `[M, N]`, allocate an output buffer `[N, M]` and write `out[j*M + i] = in[i*N + j]`. Can be implemented as a simple MSL kernel or a CPU loop (CPU loop is simpler and correct; optimize later).

Output shape is the input shape with the last two dimensions swapped. Requires shape metadata from step 1.

### 6. sum (`malus-runtime/src/metal.rs`, `malus-codegen-cpu/src/lib.rs`)

Register `sum` in `builtins.rs` with one tensor argument, returning a single-element `Tensor<f32>`.

Add to the C ABI:
```c
i64 tensor_sum(i64 handle)
```

Allocate a single-element output buffer and sum all input elements. Can use MPS reduction or a CPU-side reduction over shared memory (since `StorageModeShared` allows CPU reads after a barrier).

### 7. Shape Inspection

Register `.shape` and `.len` as special-cased `FieldAccess` on tensor types in sema. `.len` returns `i64` (number of elements). `.shape` deferred to M11 if complex (requires returning a multi-element value); `.len` is sufficient for the done-when.

In codegen-cpu, lower `FieldAccess { field: "len" }` on a tensor handle as a `tensor_len` runtime call:
```c
i64 tensor_len(i64 handle)
```

## Out of Scope

- Non-2D matmul (batched matmul)
- Reductions with `dim` argument (`sum(x, dim=0)`)
- `reshape`, `flatten`, `squeeze`, `unsqueeze`, `concat`, `stack`
- Indexing and slicing
- Non-f32 math builtins
