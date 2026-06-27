# M7 — Kernel Thickening

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-gpu`, `malus-codegen-cpu`

Make kernels useful beyond single expressions, and let host code accumulate values.

## Done-When

```malus
kernel relu_backward(grad_out: Tensor<f32>, x: Tensor<f32>) -> Tensor<f32>:
    let mask = x > 0.0
    return grad_out * mask

fn main():
    let x = Tensor.gpu<f32>([1.0, -2.0, 3.0, -4.0])
    let grad = Tensor.gpu<f32>([0.5, 0.5, 0.5, 0.5])
    let result = relu_backward(grad, x)
    println("relu_backward: {}", result)

    let mut acc = Tensor.gpu<f32>([0.0, 0.0, 0.0])
    let delta = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    acc = acc + delta
    acc = acc + delta
    println("accumulated: {}", acc)

    let scaled = acc * 0.5
    println("scaled: {}", scaled)
```

## Scope

### 1. Multi-Statement Kernel Bodies (`malus-codegen-gpu/src/lib.rs`)

Currently `lower_kernel_body` hard-errors on anything that is not a single `Return` statement. Extend it to:

- Handle any number of `Let { name, expr }` statements before the final `Return`
- Emit each `Let` as a local variable declaration in MSL: `float mask = x[tid] > 0.0;`
- Add comparison operators to `lower_expr`: `<`, `<=`, `>`, `>=`, `==`, `!=` — emit as MSL `<`, `<=`, etc. Comparison result is a float `0.0`/`1.0` (MSL bool-to-float implicit).
- Add `Lit` support to `lower_expr` so kernels can reference scalar constants like `0.0`, `1.0`
- Add ternary/select expression support: `select(a, b, cond)` in MSL (equivalent to `cond ? b : a`)

Kernel body constraint that stays: the final statement must be `Return`. No control flow (`if`/`else`, loops) inside kernel bodies in M7 — those need threadgroup controls, deferred to post-V1.

### 2. `let mut` + Reassignment

**Lexer** (`malus-syntax/src/lexer.rs` — `scan_ident_or_keyword`):
- Add `"mut" => TokenKind::Mut` to the keyword match. `match` is also not yet a keyword — add `"match" => TokenKind::Match` here too (used in M10, but the token needs to exist).

**AST** (`malus-syntax/src/ast.rs`):
- Add `StmtKind::LetMut { name: String, expr: Expr }` — mutable binding
- Add `StmtKind::Assign { target: String, expr: Expr }` — reassignment of a mutable binding

**Parser** (`malus-syntax/src/parser.rs`):
- Parse `let mut <ident> = <expr>` as `StmtKind::LetMut`
- Parse `<ident> = <expr>` as `StmtKind::Assign` — distinguish from `<ident> == <expr>` (comparison) by the single `=` token

**Sema** (`malus-sema/src/check.rs`, `malus-sema/src/env.rs`):
- Track mutability on bindings in `Env` — add a `mutable: bool` flag
- `LetMut` is type-checked like `Let` but marks the binding as mutable
- `Assign` type-checks: target must be in scope, must be mutable, new value type must match declared type
- Add `StmtKind::LetMut` and `StmtKind::Assign` handling to `check_stmt`

**Typed IR** (`malus-sema/src/typed_ir.rs`):
- Add `TypedStmt::Assign { name: String, expr: TypedExpr }` (LetMut maps to existing `TypedStmt::Let`; mutability is a sema concern, not IR)

**CTMM** (`malus-sema/src/ctmm.rs`):
- Reassignment of a tensor binding = drop the old value + bind the new one.
- The naïve "emit Drop immediately before Assign" is a use-after-free when the RHS reads the target (e.g. `acc = acc + delta` — the old `acc` is read by `acc + delta`). See ADR-0011.
- Correct approach: `hoist_gpu_subexprs` runs first and hoists `acc + delta` into a temp (`let __t0 = acc + delta; acc = __t0`), eliminating the self-reference. Then `insert_assign_drops` emits `Drop{acc}` before the Assign safely.
- The hoisted temp's allocation is "moved" into `acc` — `collect_escaping` marks it as escaping so it gets no separate Drop (the target's final Drop frees it).

**Codegen-cpu** (`malus-codegen-cpu/src/lib.rs`):
- `LetMut` declares a Cranelift variable the same way `Let` does
- `Assign` emits `def_var` to update the variable's value
- `Drop` nodes from CTMM handle the old-value free — codegen-cpu does not need to know about mutability directly

### 3. Scalar Broadcasting (`malus-sema/src/check.rs`, `malus-codegen-gpu/src/lib.rs`, `malus-codegen-cpu/src/lib.rs`)

This is M5.2 — specced but never implemented.

**Sema** — relax `check_binop` (currently line 325 of `check.rs`). The current check:
```rust
} else if tlhs.ty != trhs.ty {
    ctx.errors.push(SemaError::TypeMismatch { ... });
```
Extend the `else if` to allow `Tensor<dtype> op Scalar(dtype)` and `Scalar(dtype) op Tensor<dtype>` for arithmetic ops (`Add`, `Sub`, `Mul`, `Div`). Result type is `Tensor<dtype>`. Reject comparison ops and matmul with mixed tensor+scalar.

**Codegen-gpu** — synthesize six scalar-broadcast built-in kernels (see D7 in grilling session). Buffer layout: `a@0, scalar_val@1, out@2`. Pattern:
```msl
kernel void malus_kernel_N(
    device float* a [[buffer(0)]],
    device float* scalar_val [[buffer(1)]],  // single-element buffer
    device float* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = (a[tid] * scalar_val[0]);
}
```
The six kernels: `malus_add_scalar`, `malus_mul_scalar` (commutative; scalar-on-left uses the same kernel), `malus_sub_scalar`/`malus_div_scalar` (tensor-left), `malus_rsub_scalar`/`malus_rdiv_scalar` (scalar-left reversed). Follow the same sequential ID pattern from M5.1 (ADR-0010).

**Codegen-cpu** — detect `BinOp` where one operand is `Tensor` and the other is `Scalar`. Wrap the scalar in a single-element `tensor_alloc_gpu` call, then dispatch to the scalar broadcast built-in kernel.

## Out of Scope

- Loops or `if`/`else` inside kernel bodies (M9 + post-V1)
- NumPy-style shape broadcasting (`[3,1] * [1,4]` → `[3,4]`) — element-count must match
- Non-f32 scalar broadcasting
