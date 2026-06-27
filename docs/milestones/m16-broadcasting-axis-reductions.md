# M16 — Broadcasting + Axis Reductions

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime`.

Add NumPy-style right-aligned broadcasting for tensor arithmetic and axis-parameterized reductions (`sum`, `mean`, `max`, `var`) with `keepdim`. Both are differentiable: VJPs included. This milestone retires the `ones41 @ b` bias-broadcast trick used since M8 and unlocks layernorm from primitives.

## Done-When

`examples/broadcasting.ml` compiles and gradient-checks:

```malus
fn main():
    # Broadcasting: (4,8) + (1,8) bias without the ones41 trick
    let x  = variable(ones(4, 8))
    let b  = variable(zeros(1, 8))
    let y  = x + b
    let loss = sum(y)
    backward(loss)
    tensor_print(b.grad)

    # Axis reduction with keepdim
    let m = variable(ones(4, 8))
    let row_mean = mean(m, axis=1, keepdim=true)
    let loss2 = sum(row_mean)
    backward(loss2)
    tensor_print(m.grad)

    println("broadcasting + axis reductions: OK")
```

`b.grad` shape is `[1, 8]` (broadcast gradient summed over the batch axis). `m.grad` shape is `[4, 8]` (each element's gradient is `1/8`). Results match finite differences to 1e-4.

## Scope

### 1. NumPy-Style Broadcasting

Broadcasting rule: right-align shapes; each dimension must match, be 1, or be absent. A dimension of size 1 is broadcast-expanded to match its peer. Example: `[4,8] + [1,8]` → `[4,8]`.

**Sema (`malus-sema/src/check.rs`):** In `check_binop`, replace the current element-count-equality check with a shape-compatibility check using the broadcast rule. Compute the output shape as the broadcast result. Emit `SemaError::ShapeMismatch` (or a new `BroadcastMismatch`) when shapes are incompatible.

**Typed IR (`malus-sema/src/typed_ir.rs`):** No new node needed — BinOp semantics expand to include broadcasting. Annotate the typed BinOp with `broadcast_shape: Option<Vec<usize>>` for codegen.

**Runtime (`malus-runtime/src/metal.rs`):** Implement `tensor_broadcast_add`, `tensor_broadcast_sub`, `tensor_broadcast_mul`, `tensor_broadcast_div` that read shapes from `TensorBuffer`, iterate with index remapping (`i % dim_size` for broadcast dims), and write to an output buffer of the broadcast shape. Used only when a broadcast dimension is detected; equal-shape ops continue through the existing element-wise kernel path.

**VJP (broadcast ops):** The gradient for a broadcast input is the gradient of the output summed over the broadcast dimensions. Add `tape_broadcast_binop_vjp` that calls `tensor_sum_axes(grad, broadcast_dims)` to produce the input gradients.

### 2. Axis Reductions

Add `sum(t, axis=N, keepdim=true/false)`, `mean(t, axis=N, keepdim=true/false)`, `max(t, axis=N, keepdim=true/false)`, `var(t, axis=N, keepdim=true/false)` as builtins. The existing no-arg `sum(t)` (whole-tensor sum returning `[1]`) is unchanged.

**Builtins (`malus-sema/src/builtins.rs`):** Register the new variants. `axis` is an `i32` scalar argument; `keepdim` is a `bool` with default `false`.

**Parser (`malus-syntax/src/parser.rs`):** Keyword arguments (`axis=N`, `keepdim=true`) in call expressions. The existing `CallArg` struct supports named args — verify it works for builtin calls.

**Runtime (`malus-runtime/src/metal.rs`):** Implement `tensor_reduce_sum_axis`, `tensor_reduce_mean_axis`, `tensor_reduce_max_axis`, `tensor_reduce_var_axis` as eager CPU loops iterating over the reduction axis with correct output shape. `keepdim=true` inserts a size-1 dimension at the reduced axis.

**VJPs:**
- `sum(axis)` VJP: `dx = unsqueeze_broadcast(dout, axis, input_shape)` — broadcast the scalar/reduced gradient back to the input shape.
- `mean(axis)` VJP: `dx = unsqueeze_broadcast(dout / N, axis, input_shape)` where N is the axis size.
- `max(axis)` VJP: `dx = dout * (x == max_val)` mask (one-hot over the max position).
- `var(axis)` VJP: `dx = dout * 2 * (x - mean) / N`.

## Out of Scope

- Multi-axis reduction in a single call (`sum(t, axes=[0,1])`) — post-V3
- MPS-accelerated reductions (M21 handles the ones the transformer needs)
- Reduction over all axes with a single call and no `axis` argument for new builtins (`mean()` without axis stays whole-tensor) — post-V3
- `argmax` / `argsort` (post-V3)
