# M20 — Lvalue Assignment + AdamW

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`.

Extend `Assign` to support indexed and field assignment targets (`a[i] = e`, `s.field = e`), then ship AdamW as a reusable stdlib construct built on them. Retires a V1 language gap and proves the language composes into a real optimizer.

## Done-When

`examples/adamw.ml` compiles and trains a small linear regression to a known minimum:

```malus
struct AdamW:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32
    wd: f32

fn adamw_step(opt: AdamW, params: Array<Variable<f32>, 4>,
              ms: Array<Tensor<f32>, 4>, vs: Array<Tensor<f32>, 4>,
              t: i32):
    let bc1 = 1.0 - opt.beta1 ^ t
    let bc2 = 1.0 - opt.beta2 ^ t
    for i in range(4):
        let g  = params[i].grad + opt.wd * params[i].data
        ms[i]  = opt.beta1 * ms[i] + (1.0 - opt.beta1) * g
        vs[i]  = opt.beta2 * vs[i] + (1.0 - opt.beta2) * g * g
        let m_hat = ms[i] / bc1
        let v_hat = vs[i] / bc2
        params[i] = variable(params[i].data - opt.lr * m_hat / (sqrt(v_hat) + opt.eps))

fn main():
    let x      = ones(8, 4)
    let target = ones(8, 1)
    let mut w  = variable(randn(4, 1))
    let mut b  = variable(zeros(1, 1))

    let mut params = [w, b, variable(zeros(1,1)), variable(zeros(1,1))]
    let mut ms = [zeros(4,1), zeros(1,1), zeros(1,1), zeros(1,1)]
    let mut vs = [zeros(4,1), zeros(1,1), zeros(1,1), zeros(1,1)]

    let opt = AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)
    for t in range(1, 201):
        let pred = variable(x) @ params[0] + variable(ones(8,1)) @ params[1]
        let loss = sum((pred - variable(target)) * (pred - variable(target)))
        zero_grad(params[0], params[1])
        backward(loss)
        adamw_step(opt, params, ms, vs, t)
        if t == 1 or t == 200:
            println("step {}: loss = {}", t, loss.data)
```

Loss decreases from step 1 to step 200.

## Scope

### 1. Indexed Assignment (`a[i] = e`)

**AST (`malus-syntax/src/ast.rs`):** Extend `StmtKind::Assign` — currently `target: String` — to `target: AssignTarget` where:

```rust
enum AssignTarget {
    Ident(String),
    Index { base: String, index: Expr },
    Field { base: String, field: String },
}
```

Only single-level lvalues in M20: `a[i]` and `s.f`. No chained `a.b[i].c`.

**Parser (`malus-syntax/src/parser.rs`):** In `parse_stmt`, after parsing an identifier as the start of an assignment, look ahead for `[` (index target) or `.field =` (field target) before the `=` sign.

**Sema (`malus-sema/src/check.rs`):** Type-check `AssignTarget::Index`: verify `base` is `let mut`, element type matches `e`. `AssignTarget::Field`: verify `base` is `let mut` struct, field type matches `e`. CTMM treat these like reassignment — drop the old element/field value before binding the new one.

**CTMM (`malus-sema/src/ctmm.rs`):** Extend `insert_assign_drops` to handle `AssignTarget::Index` and `AssignTarget::Field`. For an indexed assignment into an array of tensors, emit a drop for the old element (load element handle, free/release). For a field assignment into a struct, emit a drop for the old field value.

**Codegen-cpu (`malus-codegen-cpu/src/lib.rs`):** `AssignTarget::Index` → compute `base_ptr + i*8`, store the new value. `AssignTarget::Field` → compute `base_ptr + field_offset`, store.

### 2. Field Assignment (`s.field = e`)

Same pipeline as indexed assignment above — only the codegen offset calculation differs (field offset is fixed and determined at sema time from the struct definition).

### 3. AdamW as Stdlib

AdamW is implemented **in malus source** (not in Rust), using the lvalue assignment features above. A canonical `examples/stdlib/adamw.ml` defines the `AdamW` struct and `adamw_step` fn as shown in the done-when. Users import it.

The `^` operator for scalar `beta^t` (integer power of a float) requires a small sema/codegen addition: `ScalarBinOp::Pow` lowered to a `libm::powf` call registered as a JIT symbol, or unrolled for small integer exponents.

## Out of Scope

- Nested lvalue targets (`a.b[i]`, `a[i].f`) — post-V3
- Compound assignment operators (`+=`, `-=`, etc.) — post-V3
- Multi-level struct field mutation — post-V3
- AdamW with per-param lr scheduling — post-V3
- Gradient clipping — post-V3
