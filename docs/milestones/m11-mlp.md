# M11 — The 2-Layer MLP

**Crates:** All crates.

Put it all together. Fixed-length arrays, rich error diagnostics, and the V1 done-when program.

## Done-When

`examples/mlp.ml` compiles and runs on an M-series Mac, printing decreasing loss over 10 training steps:

```malus
struct MLP:
    w1: Tensor<f32>
    w2: Tensor<f32>

kernel relu_backward(grad_out: Tensor<f32>, x: Tensor<f32>) -> Tensor<f32>:
    let mask = x > 0.0
    return grad_out * mask

fn forward(x: Tensor<f32>, model: MLP) -> Tensor<f32>:
    let h = relu(x @ model.w1)
    return h @ model.w2

fn backward(x: Tensor<f32>, model: MLP, grad_output: Tensor<f32>) -> MLP:
    let h_pre = x @ model.w1
    let h = relu(h_pre)
    let dw2 = transpose(h) @ grad_output
    let dh = grad_output @ transpose(model.w2)
    let dh_pre = relu_backward(dh, h_pre)
    let dw1 = transpose(x) @ dh_pre
    return MLP(w1=dw1, w2=dw2)

fn main():
    let mut model = MLP(w1=ones(3, 4), w2=ones(4, 2))
    let x = ones(2, 3)
    let target = zeros(2, 2)
    let lr = 0.01

    for step in range(10):
        let out = forward(x, model)
        let diff = out - target
        let loss = sum(diff)
        println("step {}: loss = {}", step, loss)
        let grads = backward(x, model, diff)
        model = MLP(
            w1=model.w1 - lr * grads.w1,
            w2=model.w2 - lr * grads.w2
        )

    println("final output: {}", forward(x, model))
```

## Scope

### 1. Fixed-Length Arrays

Array literals with compile-time-known length, iteration, and indexing. The type `Array<T, N>` has `N` as a compile-time constant.

**AST** (`malus-syntax/src/ast.rs`):
- Add `ExprKind::ArrayLiteral { elements: Vec<Expr> }` — `[expr1, expr2, ...]`
- Add `StmtKind::ForIn { var: String, iterable: Expr, body: Vec<Stmt> }` — `for x in arr: body`

**Parser:**
- Parse `[expr, expr, ...]` as `ArrayLiteral`
- Extend `for` parsing to handle `for <ident> in <expr>` (where expr is an array binding), distinguished from `for <ident> in range(...)` from M9

**Type system** (`malus-sema/src/ty.rs`):
- Add `ResolvedTy::Array { elem: Box<ResolvedTy>, len: usize }`

**Sema:**
- Infer element type from the first element; check all elements match
- `ForIn` over an `Array<T, N>`: loop variable is typed as `T`, body is checked `N` is known

**Typed IR:**
- Add `TypedExpr::ArrayLiteral { elements: Vec<TypedExpr>, ty: ResolvedTy }`
- Add `TypedStmt::ForIn { var: String, iterable: TypedExpr, body: Vec<TypedStmt> }`

**Codegen-cpu:**
- `ArrayLiteral` allocates a stack slot of `N * sizeof(element_type)` bytes, stores each element
- Array indexing `arr[i]` loads from `base_ptr + i * elem_size`
- `ForIn` over an array: emit a counted loop from 0 to N, loading `arr[i]` each iteration

**CTMM:**
- When an `Array<Tensor<f32>, N>` goes out of scope, `tensor_release` each element (using RC from M9 — array elements are structurally ambiguous by the same logic as struct fields)

### 2. Rich Error Diagnostics

**Goal:** Rust/Elm-style error messages with source spans, underlines, and help suggestions.

**Current state:** `SemaError` variants carry a `span: Span` (byte range in the source). The CLI prints them as plain strings with no source context.

**Implementation:**
- Add `ariadne` crate (or `codespan-reporting`) to `malus-cli/Cargo.toml`
- In `malus-cli/src/main.rs`, format sema errors using the crate's report builder
- Each `SemaError` variant maps to a report with: error label, underlined span, and a help message

Example output:
```
error: dtype mismatch in binary op
  --> script.ml:5:13
   |
 5 |     let c = a + b
   |             ^---^ left is Tensor<f32>, right is Tensor<f16>
   |
   = help: cast with b.to<f32>()
```

Runtime panics — add concrete shape and dtype info to panic messages in `malus-runtime/src/metal.rs`. For matmul shape mismatches, print the actual shapes.

### 3. The MLP Example and Integration Bug Fixes

Write `examples/mlp.ml` with the done-when program above.

Run it and fix integration bugs. Likely problem areas based on the CTMM gaps:

- **Struct fields across training loop iterations:** `model` is reassigned each step. The old `MLP` struct's tensor fields must be released before rebinding. CTMM + RC should handle this, but verify.
- **Barrier insertion around matmul:** `tensor_matmul` is a runtime call that schedules GPU work. CTMM's `is_gpu_producing` in `ctmm.rs` must recognize `tensor_matmul` calls as GPU-producing so barriers are inserted correctly.
- **Nested GPU expressions in the backward pass:** `grad_output @ transpose(model.w2)` — the `transpose` result is a temporary that must be freed after the matmul. CTMM hoisting (`hoist_gpu_subexprs`) must handle `tensor_matmul` and `tensor_transpose` the same way it handles `KernelCall`.
- **Chained BinOps still leaking temporaries:** The `ctmm-v1-gaps.md` gap about unbound temporaries in nested BinOps (`a + b * c`) applies to the training step expressions like `model.w1 - lr * grads.w1`. Fix any remaining leaks.
- **Scalar-broadcast 1-element tensor temps leak (M7 gap):** `a * 0.5` materializes a 1-element GPU tensor for `0.5` via `tensor_alloc_gpu`, but this temp is invisible to CTMM (created inline in codegen-cpu) and is never freed. Fix alongside the chained-BinOp temp leak — both require CTMM-visible temp nodes or a post-dispatch sweep.

### 4. Write examples/mlp.ml

The actual done-when file. Once the example runs correctly and loss decreases, M11 is complete.

## Out of Scope

- SafeTensors loading (deferred — the MLP uses `ones`/`zeros` for initialization)
- GPU RNG (deferred)
- nanoGPT or any model larger than a 2-layer MLP
- Autograd / gradient tape
- Optimizer objects (learning rate is a plain scalar, update is manual)
