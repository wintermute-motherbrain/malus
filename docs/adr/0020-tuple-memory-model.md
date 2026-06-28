# ADR-0020: Tuple memory model — heap-always, flat-only

## Status

Accepted

## Context

Tuples are anonymous product types with positional fields, added as a standalone milestone between M13 and M14. Two structural questions arose during design:

**Allocation strategy.** Tuple size is known statically (like fixed-length arrays), which in principle allows stack allocation for scalar-only tuples. However, stack allocation requires two lowering paths in codegen — one for scalar-only tuples, one for tuples containing tensors/variables that need RC. `Struct` is always heap-allocated via `malloc`/`free` with RC on tensor fields; a uniform model reuses that machinery entirely.

**Nesting and aggregate containment.** Allowing tuples as elements of other tuples, struct fields, or array elements requires `DropTuple` to recurse into inner tuple boxes and `DropStruct`/the RC fallback path to call a tuple-free function. This is a non-trivial extension of the CTMM drop machinery.

## Decision

**Heap-always.** Tuples are heap-allocated via `malloc` regardless of field types, mirroring `Struct`. `DropTuple` iterates fields, calls `tensor_release` on tensor/variable fields, then `free`s the box. No stack allocation path.

**Flat-only.** Tuple element types may not themselves be tuples. Tuples may not appear as struct fields or array element types. Sema rejects these positions with a clear error. The restriction can be lifted in a later milestone when recursive drop is needed for a concrete reason.

## Consequences

- One CTMM code path for tuples (mirrors `DropStruct`); no conditional stack/heap split.
- `malloc` overhead on every tuple construction, including scalar-only `(1.0, 2.0)`. Acceptable for current scale.
- Nested tuple types (`((f32, f32), f32)`) parse and type-check as `ResolvedTy::Tuple` but are rejected by sema at the tuple-construction and field-position checks.
