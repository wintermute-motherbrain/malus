# Distinct `Variable` type with type-directed RC

Amends ADR-0002 (CTMM RC fallback) for the autograd use case.

## Decision

Introduce a distinct `Variable<f32>` type for grad-tracked tensors. CTMM treats `Variable` as always RC-managed — emitting `tensor_retain`/`tensor_release` for every `Variable` binding on every control-flow path, purely by type. Plain `Tensor` retains its existing hierarchical static Drop on all paths. No new CTMM analysis is required.

## Why this is surprising

ADR-0002 and `docs/milestones/ctmm-v1-gaps.md` describe a deferred "dataflow liveness RC fallback" — a compiler analysis to decide which tensor lifetimes are ambiguous enough to need RC. V2 never builds this analysis. Instead, `Variable` is a distinct type and RC is *unconditionally* correct for it: every `Variable` binding retains when created, releases at last use, and the refcount is the sole source of lifetime truth. This sidesteps the analysis entirely.

The insight: the reason V1 deferred RC was that RC-by-analysis is expensive (an interprocedural fixed-point) and easy to get wrong at control-flow boundaries. RC-by-type is trivially correct — the same retain/release is emitted on every code path, so there is no control-flow ambiguity to analyze. The cost is that every `Variable` pays RC overhead. That is acceptable because `Variable` values *are* the hot path in a training loop and the tape already retains them for backward; the RC simply makes that explicit.

Plain `Tensor` keeps static `Drop` everywhere. The inference path (no tape, no `Variable`) has zero RC overhead, preserving CTMM's original guarantee.

## Considered alternatives

**Merged: `Tensor` gains a `requires_grad` flag (PyTorch ≥0.4 style).** One type, no stdlib duplication. Rejected because `requires_grad` is a runtime flag CTMM cannot read at compile time — the compiler would need to conservatively RC-manage any tensor that *might* be taped, abandoning static Drop across most of a training loop.

**One type, all-RC when autograd is active.** Switch all tensors to RC in functions that use `Variable`. Simpler but surrenders CTMM's static-free advantage in the training workload, which is the whole point of V2.

**Structural-ambiguity check at Variable creation sites.** Emit RC only when a `Variable` escapes its creation scope. Too close to the deferred liveness analysis — complex, error-prone, and unnecessary since type-directed RC is already correct.

## Consequences

- `ResolvedTy::Variable { dtype }` is added alongside `ResolvedTy::Tensor { dtype }`. They are distinct; mixed ops are a type error unless explicitly lifted.
- CTMM's `make_drop_stmt_for_ty` emits `Retain`/`Release` for `Variable` (using the dormant `TypedStmt::Retain`/`Release` nodes added in M9/M10 but never emitted until now).
- The `tensor_retain`/`tensor_release` runtime ABI (added in M9 for struct tensor fields) is now also used for `Variable` lifetime management.
- **Amended in M22:** `Variable` fields in structs are now supported. CTMM's `variable_arc_retains_for_expr` gained a `StructInit` arm that retains Variable-typed `Ident` fields at struct construction (the same pattern as the `ArrayLiteral` arm). `DropStruct` already released Variable fields on drop via `droppable_struct_fields`. The store path (`blk.wq = variable(...)`) is balanced by the existing `emit_drop_field` in the Field-assign codegen which releases the old Variable before storing the new one. Requires a `mut` binding on the struct to permit field assignment (same gate as other mutation).
