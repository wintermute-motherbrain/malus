# M17 — Shapes + Batched Matmul

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime`.

Add `reshape`/`view`, multi-axis `transpose(dims)`, and 3-D/batched matmul with VJPs. Unlocks the `(B, T, C)` tensor shapes that attention requires.

## Done-When

`examples/batched_matmul.ml` compiles and gradient-checks:

```malus
fn main():
    # reshape: (4,8) -> (2,2,8)
    let x = variable(ones(4, 8))
    let y = reshape(x, 2, 2, 8)
    tensor_print(y.data)

    # batched matmul: (B,T,C) @ (B,C,H) -> (B,T,H)
    let q = variable(ones(2, 4, 8))
    let k = variable(ones(2, 8, 4))
    let scores = q @ k
    let loss = sum(scores)
    backward(loss)
    tensor_print(q.grad)
    tensor_print(k.grad)

    println("shapes + batched matmul: OK")
```

`scores` shape is `[2, 4, 4]`. `q.grad` shape is `[2, 4, 8]`, `k.grad` shape is `[2, 8, 4]`. Results match finite differences to 1e-4.

## Scope

### 1. `reshape` / `view`

**Builtins (`malus-sema/src/builtins.rs`):** Register `reshape(t, d0, d1, ...)` accepting variadic `i64` dimension arguments. The total element count must equal the input; checked at runtime (panic with shape info on mismatch, consistent with ADR-0013).

**Runtime (`malus-runtime/src/metal.rs`):** `tensor_reshape(handle: i64, dims: *const usize, ndims: usize) -> i64` reuses the same `MTLBuffer` (no data copy), updating only the `shape` field of a new `TensorBuffer` that shares the buffer via `tensor_retain`. Zero-copy reshape: the output is a ready tensor pointing at the same memory. CTMM emits a `Release` for the output at last use (balancing the retain).

**VJP:** `reshape` VJP is `reshape(grad, input_shape)` — trivially correct since reshape is a no-op on data.

### 2. `transpose(dims)` — Multi-Axis

**Builtins (`malus-sema/src/builtins.rs`):** Extend the existing `transpose(t)` (2-D only, no args) with `transpose(t, dim0, dim1)` (swap two dimensions) and `transpose(t, d0, d1, d2, ...)` (permute all dimensions). The no-arg form stays valid for 2-D tensors.

**Runtime (`malus-runtime/src/metal.rs`):** Generalize `tensor_transpose` to accept a dims permutation. The current 2-D transpose becomes the `(1,0)` case. N-D permute: a new eager CPU loop with stride-based index remapping.

**VJP:** `transpose(dims)` VJP is `transpose(grad, inverse_perm)`.

### 3. Batched / 3-D Matmul

**Sema (`malus-sema/src/check.rs`):** Extend `check_binop` for `BinOp::Matmul`: accept 3-D inputs `(B, M, K) @ (B, K, N) -> (B, M, N)`. The batch dimension `B` must match exactly (no broadcasting over the batch dim in M17 — that's post-V3).

**Runtime (`malus-runtime/src/metal.rs`):** In `tensor_matmul`, detect 3-D inputs and loop over the batch dimension, calling the existing 2-D matmul kernel per batch slice and copying results into an output buffer of shape `[B, M, N]`. Correct but serial — MPS migration in M21 accelerates this.

**VJP (batched matmul):** Same rule as 2-D, applied per batch slice: `dA[b] = dC[b] @ B[b]ᵀ`, `dB[b] = A[b]ᵀ @ dC[b]`. The batch-reduce VJP wrapper sums over `b` when needed.

## Out of Scope

- `squeeze` / `unsqueeze` as explicit builtins (internally used in VJPs but not user-facing in M17)
- Strided / non-contiguous views (post-V3; M17's reshape always returns contiguous)
- Broadcasting over the batch dimension in batched matmul (post-V3)
- 4-D+ matmul (post-V3; transformers use 3-D)
- `einsum` (post-V3)
