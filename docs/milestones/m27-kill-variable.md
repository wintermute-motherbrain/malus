# M27 — Kill `Variable`

**Crates:** `malus-sema`, `malus-codegen-cpu`  
**Track:** frontend (converges here with GPU track)  
**Depends on:** M26 (GPU track), generics frontend work (parallel track)

Eliminate `ResolvedTy::Variable`. There is one tensor type: `Tensor<dtype>`. Grad-tracking becomes a statically-inferred sema dataflow property that drives tape emission and RC placement. See ADR-0030.

## Done-When

1. `grep -r "ResolvedTy::Variable" crates/` returns zero matches.
2. `examples/nanogpt.ml` compiles with `Variable<f32>` replaced by `Tensor<f32>` throughout, and still passes the M26 full-step CPU-counter==0 gate and the gradient_check.
3. Grad-inference unit tests pass: a tensor that derives from `variable(x)` is marked grad-tracked; the same tensor inside a `no_grad` scope is not; a tensor that does not derive from `variable(x)` is not.
4. `cargo test --workspace` passes.

## Scope

### 1. Remove `ResolvedTy::Variable` (`malus-sema/src/ty.rs`)

Delete the `Variable { dtype }` variant. Update `Display`, `PartialEq`, `from_ast_ty`, `is_tensor_like`, and all match sites.

### 2. Grad-inference pass (`malus-sema/src/check.rs` or new `src/grad_inference.rs`)

A new sema pass (runs after type-checking, before CTMM) computes a `grad_tracked: HashSet<BindingId>` set.

**Rules:**
- `variable(t)` → marks the result binding as a grad leaf.
- If all operands of a `BinOp` or builtin `Call` are grad-tracked, the result is grad-tracked.
- Bindings inside a `with no_grad: body` block are NOT grad-tracked even if they derive from leaves.
- Bindings in a `let (a, b) = tuple_expr` destructure inherit grad-tracking from the tuple.
- Function calls: if any argument is grad-tracked and the function is known to be differentiable (listed in `builtins.rs`), the return is grad-tracked.

**What grad-tracking drives:**
- `codegen-cpu`: tape-record calls (`tape_record_*`) are emitted only for grad-tracked results. Currently gated on `is_variable()`; replace with `is_grad_tracked(&binding_id)`.
- `ctmm.rs`: `make_drop_stmt_for_ty` currently dispatches on `Variable` to emit `Release` instead of `Drop`. Replace with: if the binding is in the grad-inference escape set (saved to tape), emit `Release`; else emit `Drop`. The escape set is the intersection of `grad_tracked` bindings and those that actually appear in `tape_record_*` call arguments.

### 3. Replace all `is_variable()` call sites (`malus-sema/src/check.rs`)

~39 call sites. Each falls into one of:
- **Type-checking operator rules** (e.g. "Variable + Variable → Variable"): rewrite as "grad-tracked + grad-tracked → grad-tracked Tensor". The type is always `Tensor`; grad-tracking is a separate property.
- **RC emit decision**: replace with `is_in_escape_set(&binding_id)`.
- **Tape-record emit decision**: replace with `is_grad_tracked(&binding_id)`.

Use `grep -n "is_variable" crates/malus-sema/src/check.rs` to enumerate all sites before editing.

### 4. Update `variable()` builtin (`malus-sema/src/builtins.rs`)

`variable(t: Tensor<f32>) -> Tensor<f32>` — return type changes from `Variable<f32>` to `Tensor<f32>`. Semantics: marks the result as a grad leaf in the grad-inference pass. At codegen-cpu, `variable(x)` emits `tape_register_leaf(handle)` and returns the same handle (identity).

`variable(t)` is still syntactically valid; it is not removed (backward compat, and it is the only way to create a grad leaf).

### 5. `.data` accessor

`.data` on a tensor is now an identity accessor. Sema accepts `t.data` on any `Tensor`; codegen emits the same handle. No-op in V4. Keep it for backward compat with existing examples.

### 6. `.grad` accessor

`.grad` on a leaf tensor returns its accumulated gradient (`Tensor<f32>`). Sema: error if accessed on a tensor that is not in the grad-leaf set. Codegen: unchanged (already reads the grad slot by handle from the tape's leaf map).

### 7. Codemod all examples

Replace `Variable<f32>` → `Tensor<f32>` across all `.ml` files. `variable(x)` calls remain syntactically identical (they now return `Tensor` not `Variable` but the source is the same). Delete `examples/variable_rc.ml` (its entire content is Variable RC behavior, which is gone).

### 8. Update tests

All sema tests that assert `ResolvedTy::Variable` → assert `ResolvedTy::Tensor` instead. Tests for `is_variable()` → tests for `is_grad_tracked()`. Dead error variants (`AssignVariableField`) → remove.

## Out of Scope

- `List<T>` and generics (M28).
- Borrow-inference RC (M29); this milestone lays the groundwork (grad-inference escape set) but does not yet eliminate retain/release ops.
- Changing the tape data structure.
