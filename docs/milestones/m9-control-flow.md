# M9 — Control Flow

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime`

if/else, for, while — and CTMM RC fallback for conditional tensor lifetimes.

## Done-When

```malus
fn main():
    let x = ones(2, 3)
    let w = ones(3, 2)

    for i in range(5):
        let out = x @ w
        let s = sum(out)
        println("step {}: sum = {}", i, s)
        if i > 2:
            println("  past halfway")

    println("done")
```

Tensors created inside the loop (`out`, `s`) are freed at the end of each iteration. CTMM does not leak.

## Scope

### 1. if/else

All three keywords (`if`, `else`, `for`, `in`, `while`) are already lexed as tokens.

**AST** (`malus-syntax/src/ast.rs`):
- Add `StmtKind::If { condition: Expr, then_body: Vec<Stmt>, else_body: Option<Vec<Stmt>> }`

**Parser** (`malus-syntax/src/parser.rs`):
- Parse `if <expr>: <indented-block> [else: <indented-block>]`
- The `else` branch is optional

**Typed IR** (`malus-sema/src/typed_ir.rs`):
- Add `TypedStmt::If { condition: TypedExpr, then_body: Vec<TypedStmt>, else_body: Option<Vec<TypedStmt>> }`

**Sema** (`malus-sema/src/check.rs`):
- Type-check condition as `Bool`
- Check both branches with the same environment (no new bindings from branches escape into the outer scope)
- Return type of if/else statement is `Unit` (if/else is a statement, not an expression, in V1)

**Codegen-cpu** (`malus-codegen-cpu/src/lib.rs`):
- Lower to Cranelift basic blocks: create `then_block`, `else_block`, `merge_block`
- Emit `brif condition, then_block, else_block` at the branch point
- Both branches jump to `merge_block` at their end
- Follow existing Cranelift patterns for block creation and `ins().jump()`

### 2. for Loop

**AST:**
- Add `StmtKind::For { var: String, start: Expr, end: Expr, body: Vec<Stmt> }`
- V1 only supports `range(N)` → start=0, end=N and `range(start, end)`

**Parser:**
- Parse `for <ident> in range(<expr>): <body>` and `for <ident> in range(<expr>, <expr>): <body>`
- The loop variable is scoped to the body

**Typed IR:**
- Add `TypedStmt::For { var: String, start: TypedExpr, end: TypedExpr, body: Vec<TypedStmt> }`

**Sema:**
- Loop variable is typed as `Scalar(I64)`
- `start` and `end` are type-checked as `Scalar(I64)`

**Codegen-cpu:**
- Declare loop variable as a Cranelift variable
- Create `header_block` (check condition), `body_block`, `exit_block`
- Initialize loop var to `start`, jump to `header_block`
- In `header_block`: if `var < end` → `body_block`, else → `exit_block`
- In `body_block`: compile body statements, increment var, jump to `header_block`

### 3. while Loop

**AST:** `StmtKind::While { condition: Expr, body: Vec<Stmt> }`

**Parser:** Parse `while <expr>: <body>`

**Typed IR:** `TypedStmt::While { condition: TypedExpr, body: Vec<TypedStmt> }`

**Sema:** Condition type-checked as `Bool`.

**Codegen-cpu:** Standard `header → body → header` loop with `brif` at header.

### 4. CTMM for Control Flow + RC Fallback

This is the hardest part of M9. The current linear-scan `find_last_uses` in `malus-sema/src/ctmm.rs` is unsound for branching (noted in `docs/milestones/ctmm-v1-gaps.md`).

**Strategy: RC fallback for ambiguous paths** (per ADR-0002).

Rather than implementing full dataflow liveness analysis, classify each tensor binding as either `Static` (CTMM can prove a unique drop point) or `Rc` (binding crosses a branch or loop boundary where CTMM cannot). For `Rc` bindings, emit `tensor_retain`/`tensor_release` instead of `Drop`.

**Runtime additions** (`malus-runtime/src/metal.rs`):

Add atomic refcount to `TensorBuffer`:
```rust
struct TensorBuffer {
    buffer: metal::Buffer,
    dtype: Dtype,
    len: usize,
    shape: Vec<usize>,          // added in M8
    ref_count: AtomicUsize,     // added in M9
}
```

Initial refcount is 1. Add to the C ABI:
```c
void tensor_retain(i64 handle)   // increment refcount
void tensor_release(i64 handle)  // decrement refcount; free when 0
```

`tensor_free` (existing) becomes `tensor_release` internally — it decrements and frees when zero. Existing call sites that call `tensor_free` directly remain correct (they start at refcount 1 and decrement to 0).

Add `tensor_retain` and `tensor_release` to `RuntimeSymbols`.

**CTMM classification** (`malus-sema/src/ctmm.rs`):

Update `annotate_body` to handle control flow statements. For each tensor binding that is created in an outer scope and used (or potentially freed) inside a branch:

- If the binding's last use is unambiguously in the linear part of the code (before any branch), keep it as `Static` (existing `Drop` behavior).
- If the binding is used inside an `if`/`else` branch: use RC. Emit `tensor_retain` at the branch entry (to increment), emit `tensor_release` at the end of each branch that is done with it.
- For `for`/`while` loop bodies: tensors created inside the loop body (`Let` statements inside `body`) are `Static` and dropped at the end of each iteration. Tensors created outside and used inside are unchanged (they outlive the loop).

**Typed IR additions** (`malus-sema/src/typed_ir.rs`):
- Add `TypedStmt::Retain { name: String }` — emit `tensor_retain` call
- Add `TypedStmt::Release { name: String }` — emit `tensor_release` call (used instead of `Drop` for RC-managed bindings)

**Codegen-cpu** additions:
- `Retain { name }` → call `tensor_retain` on the variable's handle value
- `Release { name }` → call `tensor_release` on the variable's handle value

## Out of Scope

- `for x in array` iteration (M11 — needs fixed arrays)
- `break` and `continue` statements
- Loop-carried mutation of tensor bindings (let mut inside a loop that reassigns a tensor works, but CTMM for it must be conservative — use RC)
- Dataflow liveness analysis (V2 optimization)
