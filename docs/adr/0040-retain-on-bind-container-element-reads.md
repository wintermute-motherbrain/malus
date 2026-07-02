# Retain-on-bind for container-element reads

## Status

Accepted (M34, 2026-07-02)

## Context

Binding a container-element read to a name — `let w = model.params[k]` on a
`List`, or the same shape on a fixed `Array` — produced unsound code. CTMM
emits a static `Drop` for every tensor-typed local binding, but the handle an
`Index` read loads is the one the container itself owns and will
independently release when the container is dropped. `retain_sites`'s `Index`
arm was an explicit no-retain fence (ADR-0026 "Out of Scope"), so the
binding's drop *stole* the container's reference.

The failure was long-fused and shape-dependent, which is why it survived from
M28 to M34:

- Untaped, repeatedly binding the same element was a deterministic
  over-release panic once the element's surplus references ran out.
- Under the tape, each theft was masked one-for-one by the tape's
  saved-operand retains — until `backward()` auto-cleared the tape, after
  which the container's elements were freed out from under it. Symptoms
  ranged from `loss.data` reading a freed/pool-recycled buffer (the M32
  addendum's bug (b)) to a hard segfault when two binds per element per step
  exceeded the masking retains.

The workaround was a documented *capstone design constraint*: never bind a
container element; read `model.params[IDX]` inline at every use site. M34's
named submodules make that rule untenable — `List<Block>` element access,
`blocks[i].params[j]` chains, and per-submodule optimizer recursion all want
bindable element reads with sane semantics.

## Decision

Every container-element read that is **bound or moved** gets its own
reference, bumped on the *new binding* immediately after the bind — there is
no source name to retain before the statement, which is what distinguishes
this from every pre-existing retain shape:

- `retain_sites.rs` gains `RetainTarget::{Source(name), Binding}`. All
  pre-M34 shapes are `Source`-targeted; the `Index` arm now produces a
  `Binding`-targeted site — `RetainKind::Tensor` for tensor elements,
  `RetainKind::Agg` (renamed from `ListAgg`; it is the generic ARC-header
  primitive) for list/struct/enum elements. Scalar, `Buffer`, and
  `.shape[i]`/`.strides[i]` reads produce no site.
- A new CTMM pass, `insert_container_read_retains`, consumes `Binding` sites
  for the three statement shapes that bind or move an element read:
  `let w = base[i]` and `x = base[i]` insert `Retain`/`RetainAgg` on the
  bound name right after the statement; `return base[i]` and slot-target
  assigns (`a[j] = base[i]`) hoist the read into a function-unique
  `__malus_tmp_N` first.
- Transient inline reads (operands, call arguments) are untouched: they
  borrow the container's reference for the duration of the enclosing
  statement and cost nothing.
- `borrow_inference` never demotes `Binding`-targeted retains: the
  container's ownership of the element is real, not an intraprocedural
  liveness artifact.

The refcount trajectory for a bound element is: bind `+1`, binding's own drop
`-1`, container's element release at container drop `-1` against the
container's own `+1` ownership — every reference has exactly one owner.

## Consequences

- Binding container-element reads is sound. The inline-read rule documented
  in `examples/nanogpt.ml` and the "capstone design constraint" sema test are
  retired as *constraints*; inline reads remain the idiom in hot paths purely
  as an optimization (no retain/release pair per bind).
- A bound element read costs one `tensor_retain`/`aggregate_retain` plus the
  existing drop. The M29 RC-reduction-ratio gate is unaffected on the
  existing corpus (its programs bind no container elements in hot loops);
  M34's modular capstone form accepts this cost as the documented structural
  price of `List` aliasing (ADR-0034).
- An *unused* binding (`let w = params[0]` never referenced) leaks its retain
  — CTMM has never emitted drops for unused bindings. Pre-existing behavior,
  now with one more way to trigger it; a dead-binding lint remains future
  work.
- Struct-typed element binds additionally require DropStruct's field release
  to be refcount-guarded (a binding's drop of a shared struct box must not
  free fields the container's copy still needs) — delivered by M34's
  peek-guard unification in the recursive drop emitter.
