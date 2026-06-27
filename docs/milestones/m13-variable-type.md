# M13 — The `Variable` Type

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`.

Introduce `Variable<f32>` — the differentiable tensor type. `Variable` is RC-managed purely by type, decided at compile time, so CTMM stays static-decidable: plain `Tensor` keeps its hierarchical static `Drop` everywhere; only `Variable` emits `tensor_retain`/`tensor_release`. The dormant RC ABI (`tensor_retain`/`tensor_release` in `malus-runtime/src/metal.rs:172–195`) and the unused `Retain`/`Release` typed-IR nodes (`malus-sema/src/typed_ir.rs:84–94`) are both activated for the first time. The tape and VJPs come in M14; M13 establishes the type representation and CTMM ownership model.

## Done-When

`examples/variable_rc.ml` compiles and runs with zero leaks under the leak-check harness:

```malus
fn wrap(t: Tensor<f32>) -> Variable<f32>:
    return variable(t)

fn identity(v: Variable<f32>) -> Variable<f32>:
    return v

fn main():
    let a = variable(ones(2, 2))
    let b = identity(a)
    let c = variable(zeros(3, 3))
    tensor_print(b.data)
    tensor_print(c.data)
```

CTMM emits balanced `tensor_retain`/`tensor_release` for every `Variable` binding (verified by the leak-check harness in `malus-codegen-cpu/src/tests.rs`). Plain `Tensor` bindings in the same program emit zero RC calls and retain their static `Drop`.

## Scope

### 1. AST and Type Form

**AST (`malus-syntax/src/ast.rs`):** Add `Ty::Variable { dtype: ScalarTy }` alongside `Ty::Tensor { dtype }`.

**Lexer (`malus-syntax/src/lexer.rs`):** Add `Token::Variable` keyword.

**Parser (`malus-syntax/src/parser.rs`):** Parse `Variable<f32>` as a type in `parse_ty`. Parse `variable(expr)` in `parse_primary` as `ExprKind::Call { callee: "variable", args: [expr] }` — the sema pass resolves the builtin.

### 2. Type System and Sema

**Type (`malus-sema/src/ty.rs`):** Add `ResolvedTy::Variable { dtype: ScalarTy }`. Add `is_variable(&self) -> bool` predicate. `Variable` is not a subtype of `Tensor` — the two are distinct in the type system. Mixed-type ops (a `Variable` BinOp a `Tensor`) are a type error in M13; M14 defines the rules when the tape is present.

**Builtins (`malus-sema/src/builtins.rs`):** Register `variable(t: Tensor<f32>) -> Variable<f32>` as a builtin. Register `.data` as a field accessor on `Variable<f32>` returning `Tensor<f32>` (the underlying tensor, no-retain). These are the only two `Variable` builtins in M13; `backward`, `.grad`, `zero_grad` are M14/M15.

**Check (`malus-sema/src/check.rs`):** Type-check `variable(expr)` calls. Type-check `v.data` field access on `Variable`. `Variable` parameters, return types, and `let`/`let mut` bindings are all legal.

### 3. CTMM — Type-Directed RC

**CTMM (`malus-sema/src/ctmm.rs`):** Add `ty_needs_rc(ty: &ResolvedTy) -> bool` that returns `true` for `ResolvedTy::Variable` (alongside the existing `is_heap_type` checks for `Struct`, `Enum`, `Array` with tensor elements). In `make_drop_stmt_for_ty`, when `ty_needs_rc` is true, emit `TypedStmt::Retain` at each new binding site and `TypedStmt::Release` at each last-use site instead of `TypedStmt::Drop`.

Key property: this decision is purely type-directed — the same `Variable` binding on every branch of an `if`/`for`/`while` emits the same retain/release regardless of control-flow path. The correctness argument (see ADR-0016) is that RC is correct for `Variable` on all paths, and static `Drop` remains correct for `Tensor` on all paths; no interprocedural analysis needed.

Ensure `collect_idents_in_stmt` recurses into all control-flow nodes (already required by M9's hierarchical CTMM) so `Variable` bindings inside loop bodies get their RC accounting right.

### 4. Codegen-cpu

**Codegen-cpu (`malus-codegen-cpu/src/lib.rs`):** Lower `TypedStmt::Retain { name }` by calling `tensor_retain(load_handle(name))` via the already-registered JIT symbol (`:65`). Lower `TypedStmt::Release { name }` by calling `tensor_release(load_handle(name))` (`:66`). `variable(t)` call lowers to: call `tensor_retain(t_handle)`, then store the same handle under the new `Variable` name (a `Variable` is the same `i64` handle as its underlying tensor — no new allocation, just a different ownership contract tracked by type).

`.data` field access on a `Variable` lowers to a plain handle load (no retain — `.data` is a borrow for printing/inspection only in M13).

## Out of Scope

- Tape recording (M14)
- `backward`, `.grad`, `zero_grad` (M14/M15)
- `no_grad` scope (M14)
- Differentiable ops on `Variable` (M14)
- Mixed `Variable` + `Tensor` arithmetic (M14 defines the rules)
- `Variable` fields in structs (post-V3; would require RC-in-struct reasoning)
