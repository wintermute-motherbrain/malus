# ADR-0034 — Generic Optimizer Write-Back: List-Backed Model + `List<T>` as an RC Aggregate

**Status:** Accepted (M28 planning, 2026-06-30)
**Relates to:** ADR-0025 (interior mutation / mut params), ADR-0026 (Lobster borrow-inference), ADR-0030 (single Tensor type + static grad-inference)

## Context

M28 (`docs/milestones/m28-module-trait.md`) asks for a `Module` trait and a single generic
`fn adamw<M: Module>` that replaces nanoGPT's hand-unrolled optimizer. The spec's own sketch
of `impl Module for GPT` is unsound as written:

```malus
impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return [self.tok_emb, self.pos_emb, ...]   # fresh literal
```

A fresh list literal is a *snapshot*. If the generic optimizer updates parameters by
reassigning slots in the returned list (`ps[i] = variable(...)` — the only write-back
mechanism `examples/adamw.ml` already proves works, via `mut`-param shared-heap aliasing per
ADR-0025), the mutation lands on the snapshot, not on `GPT`'s own fields. The next
`forward()` call reads `gpt.tok_emb` etc. directly and never observes the update. This is a
silent-wrong-training bug, not a compile error — exactly the class of defect ADR-0030 warns
about for detach points.

Three write-back mechanisms were considered:

1. **List-backed model + identity-return + slot reassignment.** Model stores one
   `List<Tensor<f32>>` field; `parameters()` returns it *by identity*; `forward()` indexes
   the same list; the optimizer mutates slots directly. Generalizes `examples/adamw.ml`'s
   proven pattern. Pure frontend (`malus-syntax`/`malus-sema`/`malus-codegen-cpu`, matching
   M28's crate fence).
2. **Bundle optimizer state into the parameter list** (`List<Param>` where `Param { w, m, v }`),
   mutating fields of each element in place. No aliasing problem (each slot's *box* is
   shared, no snapshot). Rejected: conflates model and optimizer state — Adam's moments are
   optimizer-owned in every ML framework this project tracks parity with (ADR-0022); baking
   them into the model breaks the moment a second optimizer (plain SGD) is added, and it
   changes `parameters()`'s return type away from the spec's `List<Tensor<f32>>`.
3. **In-place tensor mutation primitive** (e.g. `p.copy_(new_values)`), keeping named struct
   fields in `forward()` for readability. Rejected: requires a new `malus-runtime` op,
   breaking M28's frontend-only crate fence, and punctures tensor immutability — a property
   grad-inference (ADR-0030) and the tape currently rely on. Also stacks a new
   immutability-breaking primitive onto the same milestone cycle as M29's borrow-inference,
   the highest-risk pass in V4 (ADR-0026); the two changes are safer decoupled.

## Decision

**Write-back is List-backed identity-return + slot reassignment (option 1).**

- The model stores its parameters in one canonical `List<Tensor<f32>>` field
  (`struct GPT { params: List<Tensor<f32>> }` — see the companion decision below to collapse
  the model to a single struct).
- `impl Module for GPT: fn parameters(self) -> List<Tensor<f32>>: return self.params` — the
  field is returned **by identity**, not rebuilt as a literal.
- `forward()` reads parameters by indexing that same list (named locals bound at the top of
  the function body for readability: `let wq = model.params[1]`).
- The optimizer receives the list (aliased through `parameters()`) plus parallel `mut`
  moment-state lists (`ms`, `vs`), and updates by slot reassignment: `ps[i] = variable(...)`.
- `self` in trait methods is an **immutable borrow** — it may read fields and return one by
  identity, but never frees or rebinds the receiver (reuses the ADR-0025 mut-param
  no-free rule). No `mut self` in M28; nothing needs to mutate the receiver directly, since
  write-back flows through the returned list's shared box.
- A `len(lst) -> i64` builtin is added so the optimizer can iterate `ps[i]`/`ms[i]`/`vs[i]`
  in lockstep — bare `for p in list` gives neither an index nor access to the parallel state.
- The nanoGPT model collapses `GPT`/`Block` into a single struct holding one flat
  `params: List<Tensor<f32>>` (12 tensors, single transformer block) — a multi-struct model
  with per-substruct parameter lists is deferred (see "Consequences").

**`List<T>` is a reference-counted aggregate, not an `Array`-style static-drop container.**

Returning `self.params` by identity creates aliasing between the model's own field and every
caller's binding — an *interprocedural* alias, created across a call boundary. Neither M28's
CTMM (last-use static drop; ADR-0030's `escape_set == grad_tracked`, computed
intraprocedurally) nor M29's borrow-inference (explicitly scoped intraprocedural-only per
its own milestone spec, deferring interprocedural analysis post-V4) can prove this alias
safe by static reasoning alone. Reference counting is the sound fallback — precisely the
role CTMM's design reserves RC for (ADR-0002/0026): values that genuinely escape a single
statically-provable owner.

Concretely: `List<T>` gets an 8-byte ARC header via the *existing* `call_aggregate_alloc`
helper (the same one struct/tuple/enum boxes already use) plus an 8-byte length word:
`[refcount | len | h0 | h1 | ...]`. CTMM emits `RetainAgg`/`ReleaseAgg` around the aliasing
sites (`parameters()`'s identity-return; the optimizer's borrowed state lists). These are
**not new runtime primitives** — `aggregate_retain`/`aggregate_release` already exist in
`malus-codegen-cpu` and are fully wired to Cranelift; `RetainAgg`/`ReleaseAgg` are typed-IR
nodes that exist today but are never emitted by any sema pass. M28 activates them for `List`,
the same way M13 activated the dormant tensor RC ABI for `Variable`.

Element *tensors* inside a `List` are unaffected — they keep the ordinary tape-RC /
static-free rules from ADR-0030, gated on `.ty.is_tensor()` per that ADR's "implementation
gotcha" (a `List` can be `grad_tracked == true` while its own type is not `Tensor`; the
container's RC and an element's RC must not be confused).

## Why this is correct

The container-level RC on `List` is cheap and narrow: at most a few retains/releases per
training step (once per `parameters()` call), not per-parameter, not per-element — it does
not touch M29's tensor-RC ratio gate (`retain_count + release_count ≤ 5% of tensor
alloc_count`), which measures *element* tensor RC, not container RC. If M29 or a later
milestone lands interprocedural borrow-inference, the container retain/release can be
demoted to static drop additively, without touching this ADR's frontend surface.

## Considered alternative: fix the alias with a narrow CTMM special case

Instead of making `List` RC, add an ad-hoc CTMM rule: "a binding assigned the direct result
of a call that returns `param.field` verbatim is a borrow (no drop)." Rejected: this
hard-codes a fragile syntactic pattern-match ahead of M29's general borrow-inference
mechanism, and it only handles the one call shape this milestone happens to need — the
`List`-is-RC design generalizes to any future aliasing return without new CTMM special-casing.

## Consequences

- `docs/milestones/m28-module-trait.md` corrected: the `impl Module for GPT` example, the
  `List<T>` runtime-representation section, the method-call-syntax section (no new
  `Expr::MethodCall` AST node — method-call-shaped calls already parse as
  `Call{callee: FieldAccess{...}}`), the nanoGPT rewrite example, and the no-unroll lint
  (retargeted from a now-inapplicable `_m`/`_v` struct-field heuristic to "`.grad` reads
  confined to the one generic optimizer fn").
- `CONTEXT.md`'s `List<T>` glossary entry corrected from "escape-analysis RC same as Array"
  to "reference-counted aggregate."
- **Generic structs are out of scope for M28** (separate but related scope-tightening
  decision made in the same planning pass): only `fn` items take type parameters. No
  done-when requires `struct Wrapper<T>`, and monomorphizing aggregate field layouts,
  `DropStruct`, and per-instantiation `struct_field_grad` keys is unexercised complexity.
  Deferred post-V4, additive.
- The nanoGPT capstone's `Block` sub-model is absorbed into `GPT` as one flat parameter list
  rather than kept as a nested named struct with its own `List`. Named submodule nesting
  (`state_dict`-style parameter trees) is explicitly out of scope for M28 already (per the
  original spec) — this ADR fixes the concrete shape that constraint takes for the capstone.
