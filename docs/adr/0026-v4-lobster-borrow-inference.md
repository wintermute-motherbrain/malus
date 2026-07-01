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

- **The tape is not an RC-survivor case.** The original plan text above ("A tensor saved onto the tape for use in `backward()`... will be identified as escaping → RC") turned out to be unnecessary: every `tape_record_*` fn (`malus-runtime/src/tape.rs`) retains its own saved operands synchronously, before control ever returns to a point where CTMM could drop them. So a scalar `Tensor` binding's drop can *always* be a static `Drop`, unconditionally — `make_drop_stmt_for_ty` (ctmm.rs) drops the `grad_tracked` parameter entirely for the `Tensor` case rather than being replaced by a separate escape-set computation. RC survives only for genuinely structurally-ambiguous cases: `List<T>` (ADR-0034) and struct fields where a single owner can't be proven.
- Tensor params are a uniform zero-cost borrow ABI (`malus-sema/src/ctmm.rs::annotate_fns`): the caller never retains before a call, the callee never independently owns (or drops) a param. Returning a borrowed param straight through (`fn identity(x) { return x }`) is the one escape path that still needs a `Retain`, inserted on the return statement itself.
- Same-scope tensor aliases (`let b = a`, `let t = v.data`) and `StructInit`/`ArrayLiteral` field/element aliases of a source whose own last use *is* the construction statement are demoted from a Retain+Drop pair to a zero-cost ownership transfer when the compiler can prove it's safe (`malus-sema/src/borrow_inference.rs::demote_safe_borrows`) — implemented as a post-process cleanup over CTMM's already-correct output, not a separate whole-program pass ahead of CTMM (binding names aren't globally unique across scopes; CTMM's own hierarchical per-scope recursion remains the authority on what's in scope where).
- `is_variable()` dispatch in `ctmm.rs` — already removed in M27 (ADR-0030), not an M29 change.
- A wrong demotion = double-free. Caught deterministically by an always-on over-release guard in `tensor_release` (`prev == 0` ⇒ abort, `malus-runtime/src/metal.rs`) rather than relying on ASAN.
- **RC-op-count gate is a compile-time reduction ratio, not a runtime allocation percentage**: `retain`+`release`+`retainAgg`+`releaseAgg` node count with borrow-inference active, divided by the same count with it disabled, ≤ ~5% (Lobster's number is a reduction ratio — residual RC ÷ naive RC — not RC-ops-per-allocation). Tape self-retains are runtime bookkeeping, not compiler-emitted nodes, and are excluded. A separate, non-gating runtime check (`malus_rc_counts()`, compared step-over-step for a constant per-iteration delta) catches leaks the compile-time ratio and the over-release guard don't.
- **A genuine pre-existing bug was found and fixed during M29's implementation**, unrelated to the design change above: `insert_assign_drops` recursed into every nested control-flow body itself, redundantly duplicating `annotate_body_seeded`'s own `recurse_into_inner_scopes` step — every `let mut` reassignment inside any if/for/while got a *second*, redundant `Drop` inserted, an unconditional double-free the instant refcount hit zero. Present since `let mut` shipped in V1; silent because unit tests use a mock runtime and no prior Metal-integration test exercised a `let mut` reassignment inside a loop against real Metal handles. Fixed by removing the redundant self-recursion (`insert_assign_drops`, ctmm.rs).
