# M15 — Differentiable Stdlib + Capstone

**Crates:** All crates.

Add the remaining differentiable surface — `zero_grad`, the `no_grad` SGD update loop, and the `variable()` ergonomics for leaf parameter management — then rewrite `examples/xor.ml` as `examples/mlp_autograd.ml` with the manual backward pass deleted. This is the V2 capstone.

## Done-When

`examples/mlp_autograd.ml` compiles and runs on an M-series Mac, printing decreasing loss over 10,000 steps with final predictions rounding to `0, 1, 1, 0`. The manual backward pass from `examples/xor.ml` must not appear — it is replaced entirely by `backward(loss)`:

```
step 0: loss = [...]
step 500: loss = [...]
step 9999 (final): loss = [...]
predictions: [~0.0, ~1.0, ~1.0, ~0.0]
```

Running `examples/xor.ml` (the V1 capstone, manual backward) still produces identical convergence, confirming V1 fidelity is preserved.

## Scope

### 1. `zero_grad`

**Builtins (`malus-sema/src/builtins.rs`):** Register `zero_grad(v1, v2, ...)` as a variadic builtin that accepts any number of `Variable<f32>` arguments.

**Runtime (`malus-runtime/src/tape.rs`):** `tape_zero_grad(handles: *const i64, count: usize)` iterates the handles and, for each leaf registered in the tape's leaf registry, replaces its accumulated gradient tensor with a zeros tensor of the same shape. Called at the start of each training step to clear accumulated gradients before the next `backward`.

**Codegen-cpu (`malus-codegen-cpu/src/lib.rs`):** Lower `zero_grad(v1, v2, ...)` to an array of handles + `tape_zero_grad(ptr, count)` call.

### 2. `variable()` as Leaf Re-Wrap

After a `no_grad` parameter update (`w = variable(w.data - lr * w.grad)`), the new `variable(...)` call wraps the updated tensor as a fresh leaf. This re-registers the handle with the tape as a leaf. The old `Variable` handle is released (its retain from M13 is balanced by the reassignment drop). This round-trip — extract `.data`, update on tensor, re-wrap as `variable` — is the V2 SGD idiom. No new syntax is required; it is a composition of M13's `variable()` builtin and M14's `.data` accessor.

### 3. V2 Capstone (`examples/mlp_autograd.ml`)

Write `examples/mlp_autograd.ml` as a direct re-expression of `examples/xor.ml` with:

- `sigmoid_backward` kernel deleted (no longer needed)
- All manual gradient variables (`dout`, `dz2`, `dw2`, `db2`, `dh`, `dz1`, `dw1`, `db1`) deleted
- The backward pass lines replaced with `zero_grad(w1, b1, w2, b2)` + `backward(loss)`
- The SGD update rewritten using `no_grad:` + `.data` + `.grad` + `variable()` re-wrap
- Parameters (`w1`, `b1`, `w2`, `b2`) declared as `Variable<f32>` with the same initial values as `xor.ml`
- Forward pass unchanged in structure (same `ones41 @ b` bias trick, same sigmoid activations, same MSE loss)

The program must converge to the same loss trajectory as `xor.ml` (within numerical tolerance of the same random seed / initial weights).

### 4. CTMM Validation

The new program exercises: `Variable` bindings in a `for` loop body (released each iteration), `let mut Variable` bindings across iterations (drop-old-retain-new on reassignment), `no_grad` block with `Tensor` ops (no RC in the update). Run the full leak-check harness to confirm no leaks.

## Out of Scope

- AdamW optimizer (M20)
- `.grad` accumulation across multiple `backward` calls with `retain_graph=True` (post-V3)
- Gradient clipping (post-V3)
- `zero_grad()` with no arguments clearing all leaves globally (post-V3)
