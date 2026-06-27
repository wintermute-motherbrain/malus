# M14 — The Tape + `backward()`

**Crates:** `malus-runtime`, `malus-sema`, `malus-codegen-cpu`.

Add the define-by-run gradient tape. Forward ops on `Variable` values record onto a global thread-local tape; `backward(loss)` walks it in reverse and accumulates gradients into leaf `.grad` slots. Covers VJPs for all V1 ops, the `no_grad` scoped region, and a gradient-check done-when that proves autograd grads match finite differences. See ADR-0015.

## Done-When

`examples/gradient_check.ml` compiles, runs, and prints agreement between autograd and finite-difference gradients:

```malus
fn forward(w: Variable<f32>, x: Tensor<f32>) -> Variable<f32>:
    let h = sigmoid(variable(x) @ w)
    return sum(h)

fn main():
    let x = tensor.gpu<f32>([[1.0, 2.0], [3.0, 4.0]])
    let w = variable(tensor.gpu<f32>([[0.1, 0.2], [0.3, 0.4]]))

    let loss = forward(w, x)
    backward(loss)
    tensor_print(w.grad)

    println("autograd vs finite-diff: OK")
```

The printed gradient matches a finite-difference estimate computed separately to within 1e-4. The tape is empty after `backward` returns. Variables are released on tape clear; no leaks.

## Scope

### 1. Tape Runtime

**Runtime (`malus-runtime/src/lib.rs` + new `src/tape.rs`):** A `TapeNode` holds: a per-op backward closure (a function pointer or enum variant), retained handles for saved inputs (e.g. matmul saves both operands; sigmoid saves the output for the `sig*(1-sig)` rule), and output handle slots for gradient accumulation.

The global tape is a `thread_local! { static TAPE: RefCell<Vec<TapeNode>> }`. `tape_push(node)` appends; `tape_clear()` releases all saved handles and clears the vec.

`tape_recording() -> bool` returns whether recording is active (used by `no_grad`).

**`backward(loss_handle: i64)`:** Exported as `extern "C"` for JIT injection. Allocates a gradient of `ones_like(loss)` for `loss`. Walks `TAPE` in reverse: calls each node's backward closure with the output gradient, accumulates results into each saved input's `.grad` slot using `tensor_retain`-safe accumulation (`leaf_grad += incoming`). Calls `tape_clear()` when done.

### 2. VJP Rules

One backward closure per op, registered in `malus-runtime/src/tape.rs`. All closures operate on `i64` handles:

| Op | Forward | VJP |
|---|---|---|
| `@` (matmul) | `C = A @ B` | `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC` |
| `+` | `C = A + B` | `dA = dC`, `dB = dC` |
| `-` | `C = A - B` | `dA = dC`, `dB = -dC` |
| `*` | `C = A * B` | `dA = dC * B`, `dB = A * dC` |
| `/` | `C = A / B` | `dA = dC / B`, `dB = -dC * A / (B*B)` |
| `sigmoid` | `s = σ(x)` | `dx = dout * s * (1 - s)` |
| `relu` | `r = max(x,0)` | `dx = dout * (x > 0)` |
| `tanh` | `t = tanh(x)` | `dx = dout * (1 - t*t)` |
| `exp` | `e = exp(x)` | `dx = dout * e` |
| `log` | `l = log(x)` | `dx = dout / x` |
| `sqrt` | `s = sqrt(x)` | `dx = dout / (2*s)` |
| `abs` | `a = abs(x)` | `dx = dout * sign(x)` |
| `sum` (whole-tensor) | `s = sum(x)` | `dx = ones_like(x) * dout` |
| `transpose` (2-D) | `B = Aᵀ` | `dA = dBᵀ` |
| Unary negation | `y = -x` | `dx = -dout` |

Intermediate VJP tensors (e.g. `Bᵀ` in matmul backward) are `Tensor` values; they are static-`Drop`ped by CTMM in the Rust backward closure body. Gradient accumulation into leaf `.grad` uses `tensor_retain` to keep the leaf alive across accumulation.

### 3. `variable()` and Recording in Codegen-cpu

**Codegen-cpu (`malus-codegen-cpu/src/lib.rs`):** When lowering a `Variable`-typed op (BinOp, unary builtin call, matmul `@`), emit a JIT call to the corresponding tape-recording runtime function (e.g. `tape_matmul(a_handle, b_handle) -> i64`) rather than the raw `tensor_matmul`. The tape-recording wrapper performs the forward op, pushes a `TapeNode` with saved inputs retained, and returns the output handle.

`variable(t)` call: call `tensor_retain(t_handle)` to mark it as a leaf, then register it with `tape_register_leaf(handle)` so `backward` knows to accumulate `.grad` into it.

The `Variable`-vs-`Tensor` typing from M13 tells codegen which call path to emit — no runtime flag needed.

### 4. `no_grad` Scope

**AST (`malus-syntax/src/ast.rs`):** Add `StmtKind::NoGrad { body: Vec<Stmt> }`.

**Lexer/Parser:** Parse `with no_grad:` followed by an indented block.

**Typed IR (`malus-sema/src/typed_ir.rs`):** Add `TypedStmt::NoGrad { body: Vec<TypedStmt> }`.

**Codegen-cpu:** Emit `tape_pause()` before the body and `tape_resume()` after. Inside a `no_grad` body, `Variable` ops still use `tensor_*` directly (not tape-recording wrappers); the pause flag in `tape_recording()` ensures no nodes are pushed.

**CTMM:** `Variable` bindings inside a `no_grad` body are still RC-managed (same type-directed retain/release). `no_grad` does not change ownership semantics, only tape recording.

### 5. `.grad` Accessor

**Sema (`malus-sema/src/builtins.rs`):** Register `.grad` as a field accessor on `Variable<f32>` returning `Tensor<f32>`. The gradient tensor is owned by the tape's leaf registry; `.grad` is a borrow (no retain for read access in M14/M15 — the optimizer `no_grad` block reassigns the whole `Variable`).

**Runtime:** `tape_get_grad(leaf_handle: i64) -> i64` returns the accumulated gradient tensor handle, or a zeros tensor if no gradient has been accumulated. Used by `.grad` accessor lowering in codegen-cpu.

## Out of Scope

- `.grad` mutation via assignment (M15's `zero_grad` handles clearing)
- `zero_grad` builtin (M15)
- `retain_graph=True` (post-V3)
- Double-backward / higher-order gradients (post-V3)
- User-defined VJPs / `custom_grad` (post-V3)
- Axis-reduced `sum`/`mean` VJPs (M16)
- Batched matmul VJP (M17)
- softmax, layernorm, GELU, cross-entropy VJPs (M18)
- Embedding gather / scatter-add VJP (M19)
