# ADR-0026 — V4 Memory Model: Lobster-Style Borrow-Inference RC

**Status:** Accepted (V4)  
**Supersedes:** ADR-0002 (CTMM RC fallback), ADR-0016 (Variable type-directed RC)

## Context

malus's founding memory-management promise — "lifetime without annotations, static-free on the hot path, RC fallback only where structurally ambiguous" — was never built. V1–V3 shipped two approximations instead:

1. Conservative lexical last-use analysis (find_last_uses overwrites to the last statement index; control-flow nodes are opaque use sites). This is CTMM's shape, not its substance.
2. Type-directed RC for `Variable` (ADR-0016): RC is emitted unconditionally for every `Variable` binding by type, sidestepping the analysis entirely.

The struct/array RC fallback described in ADR-0002 and `docs/spec/04-memory.md` was never implemented (the `RetainAgg`/`retained_agg_slots` IR nodes are always empty).

## Decision

V4 builds the actual analysis. The reference model is the Lobster programming language's ownership model (https://aardappel.github.io/lobster/memory_management.html).

**Mechanism:**

1. Runtime RC is the baseline (already present in `TensorBuffer.ref_count`).
2. An ownership-analysis pass (new `malus-sema` pass) picks a **single owner** per tensor allocation — the first binding, field, or element the allocation is assigned to.
3. Every other use of that allocation is demoted to a **borrow** — zero refcount cost (no `tensor_retain`/`tensor_release` emitted).
4. RC ops survive only where ownership is genuinely shared or escaping: the tape (a tensor saved for backward escapes its creation scope), and any struct field/array element where the analysis cannot determine a unique owner.
5. The programmer shares tensors freely with no annotations. The compiler infers ownership vs. borrow at each use site.

**The autograd tape is the canonical RC-survivor case.** A tensor saved onto the tape for use in `backward()` has a lifetime that cannot be statically bounded (it persists until `backward()` clears the tape). Borrow-inference will identify these as escaping → RC. Tensors that never touch the tape receive static-free as before.

**Lobster techniques to evaluate in the implementation:**
- Per-call-site function specialization on ownership kind (owner call vs. borrow call may generate different IR).
- l-value borrow stack tracking for determining which bindings can be demoted.
- The single-owner invariant eliminates the conservative `is_variable()` dispatch used in V1–V3.

**Sequencing (M4→M6):**
- M4 implements static grad-inference (the escape set for tape-survivors), which is a prerequisite for M6.
- M6 implements the borrow-inference pass consuming the M4 escape set. RC ops surviving M6 should be ≤ ~5% of allocations.

## Why this supersedes ADR-0002 and ADR-0016

ADR-0002's "structural ambiguity → RC fallback" was the right goal but was never built. ADR-0016 worked around it with type-directed RC on `Variable`, which is being eliminated in ADR-0030. With `Variable` gone and a single `Tensor` type, the type-dispatch shortcut disappears — we must build the real analysis.

## Consequences

- `RetainAgg`/`retained_agg_slots` IR nodes (always empty) are removed.
- `is_variable()` dispatch in `ctmm.rs` is removed.
- `make_drop_stmt_for_ty` is replaced by escape-set + owner-set driven logic.
- A wrong borrow = use-after-free. The tape RC-survivor set must exactly equal the grad-inference escape set from ADR-0030.
- RC-op-count CI gate: `retain`/`release` calls ≤ ~5% of allocations in the nanoGPT hot-path test.
