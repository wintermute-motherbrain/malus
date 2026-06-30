# ADR-0030 — Single `Tensor` Type + Static Grad-Inference

**Status:** Accepted (V4-M4)  
**Supersedes:** ADR-0016 (Variable type-directed RC)

## Context

ADR-0016 introduced `Variable<f32>` as a distinct type to give CTMM a compile-time signal for RC-managed grad-tracked tensors. The cost:

- `Tensor`/`Variable` are disjoint; mixed ops are type errors — forcing constant `variable()`/`.data`/`.grad` ceremony.
- The nanoGPT capstone is dominated by this boilerplate.
- `is_variable()` is checked ~39 times in `check.rs`; every new op must dual-implement it.
- The type split is a workaround for not having real escape analysis. ADR-0026 builds the real analysis, making the workaround unnecessary.

## Decision

**V4-M4 eliminates `ResolvedTy::Variable`. There is one tensor type: `Tensor<dtype>`.**

Grad-tracking becomes a **statically-inferred dataflow property** computed in a new sema pass. The property: a tensor binding is "grad-tracked" if it derives from a leaf node (created by `variable(...)`) and is not inside a `no_grad` scope. This property drives:
1. **Tape emission** in codegen-cpu: only grad-tracked tensors cause `tape_record_*` calls. Previously this was `is_variable()`.
2. **Drop-vs-release** in CTMM: grad-tracked tensors that escape to the tape are RC-managed (per ADR-0026). Grad-tracked tensors that do not escape get static-free just like plain tensors.
3. **`.grad` access**: only tensors that are grad-tracked *and* are leaves have a `.grad` slot. Non-leaves do not accumulate `.grad` (unchanged behavior; intermediate tensors never had `.grad`).

**API surface changes:**
- `variable(x: Tensor<f32>) -> Tensor<f32>` — now a leaf-marker builtin that calls `tape_register_leaf` and returns the same tensor. Type: Tensor→Tensor (no Variable return).
- `.data` → identity (access the tensor directly; `.data` becomes a no-op accessor for backward compat).
- `.grad` → still returns `Tensor<f32>` (unchanged; `.grad` was always `Tensor`, not `Variable`).
- `zero_grad` → accepts `Tensor<f32>` (previously `Variable<f32>`).
- `backward(loss: Tensor<f32>)` → accepts the single tensor type.

**Migration:** codemod `Variable<f32>`→`Tensor<f32>` across all examples; `variable(x)` calls remain syntactically valid (leaf-marker semantics). `variable_rc.ml` is retired.

## Why this is correct

The ADR-0016 insight — "RC is always correct for Variable, so we don't need the analysis" — is valid. But it purchases correctness by conservatively RC-managing ALL grad-tracked tensors, even those that never escape. The static grad-inference pass recovers the same correctness with precision: only the tape-escaping subset pays RC, the rest get static-free. This is the founding CTMM promise, applied to the autograd path.

## Considered alternative: keep Variable, add borrow-inference

Keep `Variable` as a type; build borrow-inference on top. Rejected: the type split is an ergonomic tax (the user's entire mental model must track which tensors are Variables), and the analysis (ADR-0026) already resolves the lifetime question type-agnostically. Keeping Variable adds no information once the analysis exists.

## Consequences

- `ResolvedTy::Variable { dtype }` removed from `ty.rs`.
- All `is_variable()` call sites in `check.rs` converted to grad-inference property lookups.
- `ctmm.rs`: `make_drop_stmt_for_ty` no longer dispatches on Variable; escape analysis drives RC.
- `codegen-cpu`: tape-emission predicate switches from `is_variable()` to grad-inference property.
- All examples: `Variable<f32>` → `Tensor<f32>` (mechanical codemod).
- Gate: zero `ResolvedTy::Variable` refs in the typed IR; all autograd examples pass.
