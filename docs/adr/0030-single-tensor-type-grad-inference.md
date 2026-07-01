# ADR-0030 — Single `Tensor` Type + Static Grad-Inference

**Status:** Implemented (M27, 2026-06-30)  
**Supersedes:** ADR-0016 (Variable type-directed RC)

## Context

ADR-0016 introduced `Variable<f32>` as a distinct type to give CTMM a compile-time signal for RC-managed grad-tracked tensors. The cost:

- `Tensor`/`Variable` are disjoint; mixed ops are type errors — forcing constant `variable()`/`.data`/`.grad` ceremony.
- The nanoGPT capstone is dominated by this boilerplate.
- `is_variable()` is checked ~39 times in `check.rs`; every new op must dual-implement it.
- The type split is a workaround for not having real escape analysis. ADR-0026 builds the real analysis, making the workaround unnecessary.

## Decision

**V4-M4 eliminates `ResolvedTy::Variable`. There is one tensor type: `Tensor<dtype>`.**

Grad-tracking becomes a **statically-inferred dataflow property** computed in a new sema pass, `malus-sema/src/grad_inference.rs`. The property: a tensor binding is "grad-tracked" if it derives from a leaf node (created by `variable(...)`) and is not inside a `no_grad` scope. This property drives:
1. **Tape emission** in codegen-cpu: only grad-tracked tensors cause `tape_record_*` calls. Previously this was `is_variable()`.
2. **Drop-vs-release** in CTMM: grad-tracked tensors that escape to the tape are RC-managed (per ADR-0026). Grad-tracked tensors that do not escape get static-free just like plain tensors.
3. **`.grad` access**: only grad-tracked tensors have a `.grad` slot (see "Refinement: `.grad` gates on grad-tracked, not leaf" below).

**Refinement: the pass is interprocedural and field-sensitive, not local.** The original framing above ("computed in a new sema pass") undersold the shape of the analysis. `Variable<f32>` crossed function parameters (`fn attention(q: Variable<f32>, ...)`), function return types (`fn forward(...) -> Variable<f32>`), and struct fields (`Block.wq`, `GPT.wte`) in every real example — a purely local per-function pass cannot recover any of that, since a parameter's grad-ness comes from its callers and a field's comes from every store into it. The implemented pass is a whole-program, field-sensitive fixpoint (monotone, flags only flip false→true) with three parts: local propagation within a function body (seeded at `variable(...)`, propagated through `BinOp`/differentiable-builtin operands, killed inside `with no_grad:`), an interprocedural component (`fn_param_grad`: a param is grad-tracked if any call site passes a grad-tracked argument in that position; `fn_ret_grad`: a function's return is grad-tracked if its return expression is), and a field-sensitive component (`struct_field_grad`: a `(StructType, field)` pair is grad-tracked if any construction or field-assign ever stores a grad-tracked value into it). These three maps live on `TypedProgram`; the per-expression result is a `grad_tracked: bool` on `TypedExpr`.

**Refinement: `escape_set == grad_tracked` at M27, not a subset.** The original framing implied the escape set (the RC-managed subset) is smaller than the grad-tracked set. In practice the tape retains uniformly — `tape_record_*` calls `tensor_retain` on every saved operand *and* the output, and `variable()` retains its leaf — so every grad-tracked tensor is, in fact, retained by the tape on some path. CTMM's `make_drop_stmt_for_ty` therefore takes the grad-tracked flag directly: `Release` for grad-tracked bindings, `Drop` otherwise. This is byte-identical RC placement to the old type-directed scheme. Computing a strictly smaller escape set (RC only the tensors an op's specific VJP actually needs to keep) requires per-VJP-minimal save-sets, which changes the tape's retention discipline — that is real M29 (borrow-inference) work, not M27's.

**Refinement: `.data` and `.grad` are detach points, not identity.** `.data`'s actual role pre-M27 was *detach*: it returned `Tensor` (a different type from `Variable`), severing grad-tracking so an optimizer's `w.data - lr * grad` was plain arithmetic, not tape-recorded. With one type, detach cannot ride on a type change, so `grad_inference.rs` special-cases `x.data` and `x.grad`: the result is unconditionally forced non-grad-tracked, regardless of the receiver. Treating `.data` as a pass-through identity (as first drafted) would record the AdamW update onto the tape — it runs outside `no_grad` — and corrupt the following step's `backward()`. This is silent-wrong-gradients, not a compile error, so it is covered by a dedicated unit test rather than relying on the gates alone.

**Refinement: `.grad` gates on grad-tracked, not leaf.** Building a precise leaf set (which bindings/fields/returns trace back specifically to a `variable(...)` call, as opposed to any grad-tracked derivation) is a second interprocedural fixpoint parallel to `grad_tracked`, for a diagnostic-only benefit no downstream consumer needs. `.grad` is instead legal on any grad-tracked receiver — matching the old behavior, where `.grad` was allowed on any `Variable` and simply returned zeros for a non-leaf.

**API surface changes:**
- `variable(x: Tensor<f32>) -> Tensor<f32>` — now a leaf-marker builtin that calls `tape_register_leaf` and returns the same tensor. Type: Tensor→Tensor (no Variable return).
- `.data` → same handle at runtime; detached (non-grad-tracked) per the refinement above.
- `.grad` → still returns `Tensor<f32>`; also a detach point (no double-backward).
- `zero_grad` → accepts `Tensor<f32>` (previously `Variable<f32>`).
- `backward(loss: Tensor<f32>)` → accepts the single tensor type.

**Migration:** codemod `Variable<f32>`→`Tensor<f32>` across all examples; `variable(x)` calls remain syntactically valid (leaf-marker semantics). `variable_rc.ml` is retired.

**Implementation gotcha:** `grad_tracked` is content-based, unlike `is_variable()` which was type-safe by construction (an `Array`/`Struct`/`Tuple` could never itself be `Variable`-typed). An aggregate literal built from grad-tracked elements (e.g. `[variable(x), variable(y)]`, or an `Array<Tensor<f32>, N>` fn param fed grad-tracked args) can be `grad_tracked == true` while its type is not `Tensor`. Two CTMM sites — the caller-retains ARC pass and function-param RC seeding — must additionally gate on `.ty.is_tensor()` before treating a grad-tracked flag as "emit a scalar `tensor_retain`/`tensor_release`", or they call a Tensor RC op on an aggregate box pointer (type confusion; segfaults on aggregate-of-Variable params such as `examples/adamw.ml`'s weight array).

## Why this is correct

The ADR-0016 insight — "RC is always correct for Variable, so we don't need the analysis" — is valid. But it purchases correctness by conservatively RC-managing ALL grad-tracked tensors, even those that never escape. The static grad-inference pass recovers the same correctness with precision: only the tape-escaping subset pays RC, the rest get static-free. This is the founding CTMM promise, applied to the autograd path.

## Considered alternative: keep Variable, add borrow-inference

Keep `Variable` as a type; build borrow-inference on top. Rejected: the type split is an ergonomic tax (the user's entire mental model must track which tensors are Variables), and the analysis (ADR-0026) already resolves the lifetime question type-agnostically. Keeping Variable adds no information once the analysis exists.

## Consequences

- `ResolvedTy::Variable { dtype }` removed from `ty.rs`.
- All `is_variable()` call sites in `check.rs` converted to grad-inference property lookups.
- `ctmm.rs`: `make_drop_stmt_for_ty` no longer dispatches on Variable; the grad-tracked flag drives RC (see "escape_set == grad_tracked" above).
- `codegen-cpu`: tape-emission predicate switches from `is_variable()` to `TypedExpr.grad_tracked`.
- All examples: `Variable<f32>` → `Tensor<f32>` (mechanical codemod); `examples/variable_rc.ml` deleted.
- Gate: zero `ResolvedTy::Variable` refs in the typed IR; all autograd examples pass — verified: `cargo test --workspace` green; `test_v4_m3_full_step_zero_cpu_compute`, `test_nanogpt_forward_zero_cpu_compute`, `test_gradient_check_all_ops` pass on Metal hardware; `examples/nanogpt.ml` trains end-to-end (loss 4.86 → ~2.5–2.7 over 300 steps) and samples without panicking.
