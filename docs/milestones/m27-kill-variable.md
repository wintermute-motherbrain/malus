# M27 — Kill `Variable`

**Status:** ✅ done (2026-06-30)  
**Crates:** `malus-sema`, `malus-codegen-cpu`, `malus-codegen-gpu`, `malus-syntax`  
**Track:** frontend (converges here with GPU track)  
**Depends on:** M26 (GPU track). Despite the header below, this milestone did **not** depend on M28 generics frontend work — `examples/nanogpt.ml` (the done-when target) uses no generics/`List`/`Module`; it is entirely concrete structs and hand-written fns. Grad-inference runs on concrete fns and structs and is forward-compatible with M28's later monomorphized generics.

Eliminate `ResolvedTy::Variable`. There is one tensor type: `Tensor<dtype>`. Grad-tracking becomes a statically-inferred sema dataflow property that drives tape emission and RC placement. See ADR-0030.

**Correction to the scope below, decided during implementation planning:** the "Grad-inference pass" section originally described a **local** `HashSet<BindingId>` pass. That undersold the real shape of the analysis — `Variable<f32>` today crosses function parameters, function return types, and struct fields (see `examples/nanogpt.ml`'s `fn forward(...) -> Variable<f32>` and `struct Block: wq: Variable<f32> ...`), none of which a local pass can infer. The implemented pass (`malus-sema/src/grad_inference.rs`) is a **whole-program, field-sensitive, interprocedural fixpoint**: local propagation within each fn body, plus `fn_param_grad`/`fn_ret_grad` maps (interprocedural) and a `struct_field_grad` map (field-sensitive), all monotone (flags only flip false→true) on `TypedProgram`. See ADR-0030 for full detail. Two further corrections folded into the same design pass: `.data`/`.grad` are **detach points** (force non-grad-tracked), not the no-op identity item 5 below describes; and the escape set item 2 describes as a subset of grad-tracked is in practice **equal to** grad-tracked at M27, because the tape retains every recorded op's operands and output uniformly.

## Done-When

1. ✅ `grep -r "ResolvedTy::Variable" crates/` returns zero matches.
2. ✅ `examples/nanogpt.ml` compiles with `Variable<f32>` replaced by `Tensor<f32>` throughout, and still passes the M26 full-step CPU-counter==0 gate and the gradient_check. Verified on real Metal hardware: `test_v4_m3_full_step_zero_cpu_compute`, `test_nanogpt_forward_zero_cpu_compute`, `test_gradient_check_all_ops` all pass; `nanogpt.ml` trains end-to-end (loss 4.86 → ~2.5–2.7 over 300 steps) and samples without panicking.
3. ✅ Grad-inference unit tests pass: a tensor that derives from `variable(x)` is marked grad-tracked; the same tensor inside a `no_grad` scope is not; a tensor that does not derive from `variable(x)` is not. Plus: interprocedural param/return propagation, struct-field grad-carrying, `.data`/`.grad` detach.
4. ✅ `cargo test --workspace` passes.

## Scope

### 1. Remove `ResolvedTy::Variable` (`malus-sema/src/ty.rs`)

Delete the `Variable { dtype }` variant. Update `Display`, `PartialEq`, `from_ast_ty`, `is_tensor_like`, and all match sites.

### 2. Grad-inference pass (implemented as `malus-sema/src/grad_inference.rs`)

Implemented as a **whole-program, field-sensitive, interprocedural fixpoint** (see the correction note at the top of this doc and ADR-0030), not the local `HashSet<BindingId>` pass originally sketched here. Runs after type-checking, before CTMM. Produces `grad_tracked: bool` on each `TypedExpr`, plus `fn_param_grad`, `fn_ret_grad`, and `struct_field_grad` maps on `TypedProgram`.

**Rules:**
- `variable(t)` → marks the result binding as a grad leaf.
- If any operand of a `BinOp` or differentiable builtin `Call` is grad-tracked, the result is grad-tracked.
- `x.data` and `x.grad` → always non-grad-tracked (detach points), regardless of the receiver.
- Bindings inside a `with no_grad: body` block are NOT grad-tracked even if they derive from leaves.
- Bindings in a `let (a, b) = tuple_expr` destructure inherit grad-tracking from the tuple.
- Function calls: a param is grad-tracked if any call site passes a grad-tracked argument in that position; a fn's return is grad-tracked if its return expression is; a call's result inherits the callee's `fn_ret_grad` (or, for builtins, the differentiable-builtin rule above).
- Struct fields: a `(StructType, field)` pair is grad-tracked if any construction or field-assign ever stores a grad-tracked value into it; `s.field` reads inherit that flag.
- `let mut` reassignment / repeated field-assign: the binding's/field's flag is the union over all assigned values (monotone, once true stays true).

**What grad-tracking drives:**
- `codegen-cpu`: tape-record calls (`tape_record_*`) are emitted only for grad-tracked results — reads `TypedExpr.grad_tracked` (previously `is_variable()`).
- `ctmm.rs`: `make_drop_stmt_for_ty` takes the grad-tracked flag directly: `Release` for grad-tracked bindings, `Drop` otherwise. At M27, `escape_set == grad_tracked` — the tape retains every recorded op's operands and output uniformly, so there is no smaller escape set to compute yet (that's real M29 work). **Gotcha:** `grad_tracked` is content-based and can be true for an `Array`/`Struct`/`Tuple`-typed expression too (e.g. `[variable(x), variable(y)]`), unlike the old `is_variable()` which was type-safe by construction. Sites that emit scalar `tensor_retain`/`tensor_release` (the caller-retains ARC pass, fn-param RC seeding) must additionally gate on `.ty.is_tensor()`, or they call a Tensor RC op on an aggregate box pointer.

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

**Corrected from the original "identity/no-op" framing** (see the note at the top of this doc): `.data`'s real job is *detach*, not identity. At runtime `.data` is still a same-handle passthrough (sema accepts `t.data` on any `Tensor`; codegen emits the same handle) — but the grad-inference pass forces the *result* to be non-grad-tracked regardless of the receiver. This preserves the old behavior, where `.data` returned a different (non-`Variable`) type and severed tape recording; e.g. an optimizer's `w.data - lr * grad` must not be tape-recorded.

### 6. `.grad` accessor

`.grad` returns the accumulated gradient (`Tensor<f32>`) and is itself a detach point (no double-backward). Sema: legal on any grad-tracked receiver — gated on `grad_tracked`, not a separate leaf set (a leaf-only gate would require a second interprocedural analysis for no consumer that needs it; non-leaf `.grad` returns zeros, same as before). Codegen: unchanged (already reads the grad slot by handle from the tape's leaf map).

### 7. Codemod all examples

Replace `Variable<f32>` → `Tensor<f32>` across all `.ml` files. `variable(x)` calls remain syntactically identical (they now return `Tensor` not `Variable` but the source is the same). Delete `examples/variable_rc.ml` (its entire content is Variable RC behavior, which is gone).

### 8. Update tests

All sema tests that assert `ResolvedTy::Variable` → assert `ResolvedTy::Tensor` instead. Tests for `is_variable()` → tests for `is_grad_tracked()`. Dead error variants (`AssignVariableField`) → remove.

## Out of Scope

- `List<T>` and generics (M28).
- Borrow-inference RC (M29); this milestone lays the groundwork (grad-inference escape set) but does not yet eliminate retain/release ops.
- Changing the tape data structure.
