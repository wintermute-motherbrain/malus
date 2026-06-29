# ADR-0021 — tensor_release calls tape_on_release on the 1→0 transition

## Status

Accepted (M15)

## Context

The SGD re-wrap idiom (`w = variable(w.data - lr * w.grad)`) mints a new leaf
handle every training step. Before M15, `tape_register_leaf` only ever inserted
into `LEAVES`; `LEAF_GRAD` survived `tape_clear`; the only full reset was
`tape_reset()` (not JIT-injected, test-only). Over 10,000 steps × 4 params this
leaks ~40,000 stale `LEAVES`/`LEAF_GRAD` entries and the grad tensors they hold —
conflicting with the M15 done-when requirement of no leaks.

## Decision

When a `TensorBuffer`'s refcount drops from 1 to 0 inside `tensor_release`
(metal.rs), call `crate::tape::tape_on_release(handle)` before freeing the box.
`tape_on_release` removes the handle from `LEAVES` and releases + removes its
`LEAF_GRAD` entry. Both callee and caller are in `malus-runtime`, so no new
crate dependency is introduced.

To prevent double-borrow panics (the two existing sites that called
`tensor_release` while holding `LEAVES`/`LEAF_GRAD` borrows), we refactored
`backward()`'s fold loop and `tape_reset()`'s drain to stash handles in a `Vec`,
drop the borrows, then release. This establishes the invariant: *never call
`tensor_release` while holding a tape borrow*.

## Alternatives considered

**`tape_reset()` at program exit.** Drain all `LEAVES`/`LEAF_GRAD` once at
teardown. Balances net alloc/free counts but allows ~40k live tensors to pile up
mid-run (memory growth). Also requires wiring `tape_reset` into program teardown
(no current hook) — more plumbing for a worse result.

**`zero_grad` deregisters stale handles.** `zero_grad` only ever receives the
*current* handles; it cannot reach the 9,999 abandoned old handles from prior
iterations. Does not fix the re-wrap leak.

## Consequences

- Leaf lifetime == tensor lifetime. Leaves are automatically deregistered when
  their tensor is freed, with no programmer discipline required.
- Fixes both the end-of-run leak and mid-run memory growth.
- `tensor_release` (the hot refcount path) now does a `HashSet::remove` +
  `HashMap::remove` on the 1→0 transition. At V2 scale (4 params × 10k steps)
  the overhead is negligible.
- The invariant "no `tensor_release` inside a tape borrow" must be maintained by
  future code. Violations cause a RefCell double-borrow panic at runtime (loud
  and obvious, not a silent memory bug).
- `tape_on_release` calling `tensor_release(g)` on the grad tensor is bounded:
  grad tensors are never leaves, so the recursive call is a lookup-miss and
  returns immediately.
