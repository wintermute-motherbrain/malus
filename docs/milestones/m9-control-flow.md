# M9 — Control Flow ✅ done

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime`

if/else, for, while — and hierarchical CTMM analysis to correctly place Drop nodes across control flow boundaries.

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

Tensors created inside the loop (`out`, `s`) are freed at the end of each iteration. Outer tensors (`x`, `w`) survive the loop and are freed after it. CTMM does not leak.

## Scope

### 1. if/else

All keywords (`if`, `else`, `for`, `in`, `while`) are already lexed as tokens.

**AST** (`malus-syntax/src/ast.rs`):
- Add `StmtKind::If { condition: Expr, then_body: Vec<Stmt>, else_body: Option<Vec<Stmt>> }`

**Parser** (`malus-syntax/src/parser.rs`):
- Parse `if <expr>: <indented-block> [else: <indented-block>]`
- The `else` branch is optional

**Typed IR** (`malus-sema/src/typed_ir.rs`):
- Add `TypedStmt::If { condition: TypedExpr, then_body: Vec<TypedStmt>, else_body: Option<Vec<TypedStmt>> }`

**Sema** (`malus-sema/src/check.rs`):
- Type-check condition as `Bool`
- Check both branches with the same environment; bindings introduced inside a branch do not escape into the outer scope
- `if`/`else` is a statement, not an expression, in V1 — return type is `Unit`

**Codegen-cpu** (`malus-codegen-cpu/src/lib.rs`):
- Lower to Cranelift basic blocks: `then_block`, `else_block`, `merge_block`
- Emit `brif condition, then_block, else_block` at the branch point
- Both branches jump to `merge_block` at their end

### 2. for Loop

**AST:**
- Add `StmtKind::For { var: String, start: Expr, end: Expr, body: Vec<Stmt> }`
- `range(N)` desugars to `start=0, end=N`; `range(start, end)` desugars to `start=start, end=end`

**Parser:**
- Parse `for <ident> in range(<expr>): <body>` and `for <ident> in range(<expr>, <expr>): <body>` as a single special form
- `range` is not a runtime function — it is syntactic sugar for loop bounds, recognized only in this context
- The loop variable is scoped to the body; `range(start, end)` where `start >= end` produces an empty loop

**Typed IR:**
- Add `TypedStmt::For { var: String, start: TypedExpr, end: TypedExpr, body: Vec<TypedStmt> }`

**Sema:**
- Loop variable is typed as `Scalar(I64)` and added to the environment for the body only
- `start` and `end` are type-checked as `Scalar(I64)`

**Codegen-cpu:**
- Declare loop variable as a Cranelift variable; initialize to `start`
- Create `header_block` (check condition), `body_block`, `exit_block`
- In `header_block`: if `var < end` → `body_block`, else → `exit_block`
- In `body_block`: compile body statements, increment var, jump to `header_block`

### 3. while Loop

**AST:** `StmtKind::While { condition: Expr, body: Vec<Stmt> }`

**Parser:** Parse `while <expr>: <body>`

**Typed IR:** `TypedStmt::While { condition: TypedExpr, body: Vec<TypedStmt> }`

**Sema:** Condition type-checked as `Bool`.

**Codegen-cpu:** Standard `header → body → header` loop with `brif` at header.

### 4. CTMM: Hierarchical Analysis

The current flat `annotate_body` is unsound for control flow — it cannot see tensor references inside nested scopes (see `docs/milestones/ctmm-v1-gaps.md`). M9 replaces it with a hierarchical analysis. See ADR-0014 for the full rationale.

**Strategy: hierarchical scope analysis (no RC emitted in M9).**

`annotate_body` recurses into each inner scope (if/else branches, loop bodies) before running the outer-scope passes. The outer scope treats each `If`/`For`/`While` node as an opaque use site: any outer binding referenced anywhere inside the node has its Drop placed after the node in the outer body. This is always correct regardless of which branch ran or how many iterations executed.

**`annotate_body` pass order (updated):**

1. `hoist_gpu_subexprs` — extended to recurse into inner bodies
2. `hoist_gpu_producing_returns` — extended to recurse into inner bodies
3. Recursively call `annotate_body` on each inner scope (if branches, loop body)
4. `collect_local_bindings` — outer body only (no recursion — inner bindings are handled in step 3)
5. `collect_escaping` — recurse into inner bodies (a binding may escape via `return` inside an if/loop)
6. `insert_assign_drops` — recurse into inner bodies (see below)
7. `find_last_uses` — `collect_idents_in_stmt` recurses so outer analysis sees inner references
8. `insert_drops` — outer body only
9. `insert_barriers` — outer body with precise recursive check at control flow boundaries (see below)

**`insert_assign_drops` fix for outer-scope reassignment:**

The existing `locals.contains(name)` guard in `insert_assign_drops` prevents it from dropping outer-scope bindings reassigned inside a loop. Remove this guard: the type checker guarantees that only `let mut` bindings appear as Assign targets (never parameters), so the check `!escaping.contains(name) && expr.ty.is_tensor()` is sufficient. This allows inner-scope `insert_assign_drops` to correctly emit `Drop { name }` before each reassignment of an outer `let mut` tensor, without RC.

**`insert_barriers` at control flow boundaries:**

When `insert_barriers` in the outer body encounters a control flow node with a non-empty pending set, it recursively checks whether any pending name is referenced inside the node's bodies (reusing `collect_idents_in_stmt`). If yes, a `GpuBarrier` is emitted before the node and the pending set is cleared. Inner bodies have their own barriers for GPU work enqueued within them.

**Runtime additions** (`malus-runtime/src/metal.rs`):

Add atomic refcount to `TensorBuffer` — needed by M10 (struct fields):
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

`tensor_free` becomes `tensor_release` internally. Existing `tensor_free` call sites remain correct (start at refcount 1, decrement to 0).

Add `tensor_retain` and `tensor_release` to `RuntimeSymbols`.

**Typed IR additions** (`malus-sema/src/typed_ir.rs`):
- Add `TypedStmt::Retain { name: String }` — will emit `tensor_retain` (used by M10+)
- Add `TypedStmt::Release { name: String }` — will emit `tensor_release` (used by M10+)

M9's CTMM analysis emits zero `Retain`/`Release` nodes. They are added now so M10 can generate them without changing the IR or the runtime ABI.

**Codegen-cpu** additions (for M10 readiness):
- `Retain { name }` → call `tensor_retain` on the variable's handle value
- `Release { name }` → call `tensor_release` on the variable's handle value

## Out of Scope

- `for x in array` iteration (M11 — needs fixed arrays)
- `break` and `continue` statements
- Early `return` inside a control flow body (M11 — requires a "scope unwind" pass to drop all live outer-scope tensors before exiting)
- Dataflow liveness analysis (V2 optimization)
