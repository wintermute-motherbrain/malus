# M19 — Embeddings + Index Tensors

**Crates:** `malus-runtime`, `malus-sema`, `malus-codegen-cpu`, `malus-codegen-gpu`.

Add i32/i64 index tensors (the minimum dtype extension for embedding lookup), `gather`/`embedding` with scatter-add VJP, and `randn`/Philox random initialization for weight tensors.

## Done-When

`examples/embedding.ml` compiles and gradient-checks:

```malus
fn main():
    # Random init
    let w_emb = variable(randn(256, 16))
    let w_pos = variable(randn(8, 16))

    # Token indices (integer tensor)
    let tokens = tensor_int.cpu<i32>([3, 1, 4, 1, 5, 9, 2, 6])

    # Embedding lookup with gradient
    let tok_emb = embedding(w_emb, tokens)
    let pos_emb = embedding(w_pos, tensor_int.cpu<i32>([0, 1, 2, 3, 4, 5, 6, 7]))
    let x = tok_emb + pos_emb
    let loss = sum(x)
    backward(loss)
    tensor_print(w_emb.grad)
    println("embeddings: OK")
```

`w_emb.grad` is non-zero at rows `[1, 2, 3, 4, 5, 6, 9]` (the looked-up rows) and zero elsewhere. Gradient matches finite differences to 1e-4.

## Scope

### 1. Integer Tensor Type (i32 / i64)

This is a **narrow dtype addition** for index tensors only. f16/bf16 compute generality remains deferred.

**AST (`malus-syntax/src/ast.rs`):** `Ty::Tensor` already allows `ScalarTy::I32` / `I64`. No AST change.

**Builtins (`malus-sema/src/builtins.rs`):** Register `tensor_int.cpu<i32>(...)` (and `<i64>`) as a tensor literal constructor that produces a CPU-placed integer tensor. Not differentiable — integer tensors are never `Variable`.

**Runtime (`malus-runtime/src/metal.rs`):** Lift the non-f32 panic guard in `tensor_alloc_gpu` for `Dtype::I32` and `Dtype::I64`. Add an `element_size` branch for these dtypes (4 and 8 bytes respectively). `tensor_print` for integer tensors: format as integers. Integer tensors are CPU-only (no MSL kernel dispatch with integer data in M19 — they are index operands only).

**Codegen-gpu (`malus-codegen-gpu/src/lib.rs`):** No change needed in M19; integer tensors are not dispatched as kernel inputs.

### 2. `embedding(weight, indices)` + Scatter-Add VJP

**Builtins:** Register `embedding(weight: Variable<f32>, indices: Tensor<i32>) -> Variable<f32>`. `weight` is `[V, D]` (vocab × embed-dim); `indices` is `[T]`; output is `[T, D]`.

**Runtime:** `tensor_embedding(weight_handle: i64, indices_handle: i64) -> i64` — eager CPU gather: `out[t] = weight[indices[t]]` for `t in 0..T`.

**VJP (scatter-add):** The gradient of `embedding` w.r.t. `weight` is a scatter-add: `dweight[indices[t]] += dout[t]` for each `t`. `tensor_scatter_add(dout_handle, indices_handle, vocab_size) -> i64` computes this. The gradient w.r.t. `indices` is undefined (not differentiable); only `weight.grad` is updated.

### 3. `randn(d0, d1, ...)` — Philox RNG

**Builtins:** Register `randn(d0, d1, ...) -> Tensor<f32>` (GPU-placed, f32). Variadic dimension args matching `zeros`/`ones`.

**Runtime:** `tensor_randn(shape_ptr: *const usize, ndims: usize) -> i64` — Philox4x32-10 counter-based RNG seeded from a thread-local counter incremented per call. Each element `x[i]` is derived from Philox(counter, i), then Box-Muller transformed to a standard normal. CPU-side generation, result placed in a `StorageModeShared` buffer (ready tensor).

The seed is not user-settable in M19 — reproducibility via a fixed default seed. User-settable seed is post-V3.

## Out of Scope

- f16/bf16 compute tensors (still deferred)
- GPU-side Philox kernel (Philox RNG is CPU-side in M19)
- `gather` with multi-dimensional index tensors (M19 covers 1-D index only)
- `scatter` as a standalone op (VJP-internal only in M19)
- User-settable random seed (post-V3)
- `embedding_bag` / weighted embeddings (post-V3)
