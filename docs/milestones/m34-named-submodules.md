# M34 — Named Submodules (Module Composition)

**Crates:** `malus-sema`, `malus-codegen-cpu`, `malus-cli` (lint), `examples/`
**Track:** capability
**Depends on:** M28 machinery; independent of M31–M33
**Status:** done (2026-07-02) — see the M34 addendum in `m29-benchmark-results.md` and ADR-0040

**Implementation decisions (2026-07-02 grilling session + findings):**
- The Block sketch below is superseded: a Block stores its tensors in ONE
  per-block identity list (`struct Block { params: List<Tensor<f32>> }`) with
  block-LOCAL index constants (`WQ = 0` …), read inline in `forward`. Named
  tensor fields would break the ADR-0034 write-back invariant — the optimizer
  writes list slots, so field reads would go stale after step 1.
- Optimizer state: `struct Moments { ms, vs }`, one per submodule in a
  `List<Moments>` parallel to `blocks`, plus one for GPT's own tensors. The
  generic `adamw` signature is unchanged; recursion happens at the call site.
- `blk.forward(x)` required a small language-surface change: a trait impl may
  carry methods the trait doesn't declare (registered as inherent, same
  mangling/dispatch); name-matches-trait-with-diverging-signature still errors.
- Drop semantics unified on the refcount-peek guard: ALL ARC'd aggregates
  (struct/enum/tuple/List) release contents only on last-ref. Pre-M34
  struct/tuple field release was unconditional — unsound for a struct bound
  out of a List element.
- Done-when #0 grew a third bug, found by systematic probing: non-grad-tracked
  tensor alias retains were grad-gated while the alias's Drop was not
  (double-release). All three fixes are in; see ADR-0040 and the M34 addendum.

Let the capstone be written the way a PyTorch user would write it: `GPT { blocks: List<Block>, ... }` with `impl Module for Block`, six layers deep, no flat-list index arithmetic. See ADR-0036 for the composition contract.

## The design problem this solves

M28's write-back invariant: `parameters()` returns the model's stored `List<Tensor<f32>>` **by identity**, so the optimizer's slot reassignment (`ps[i] = variable(...)`) is visible to the model's next `forward()`. Concatenating six sub-lists into one would produce a fresh snapshot — the optimizer would mutate the snapshot and training silently stops updating the real weights (the exact hazard ADR-0034 documents). Therefore: **no concat.** The optimizer recurses instead — `for blk in model.blocks { adamw_step(blk, ...) }` — so each submodule's own identity list is what gets updated. Concat is not added to the language in this milestone.

## Done-When

0. **[must-resolve, deferred from M33 by decision 2026-07-02]** The two pre-existing autograd lifetime bugs discovered at M32 (see the M32 addendum in `m29-benchmark-results.md`) are root-caused and fixed, with regression tests: (a) a loop-carried `let mut` tensor reassigned across `for` iterations under the tape trips the M29 over-release detector in `backward()`; (b) a 6-deep chain of block-fn calls frees the loss tensor early (`loss.data` read-after-free). These sit squarely in the drop/RC machinery this milestone rebuilds, and the capstone's multi-block structure (M35) cannot be trusted while (b) is open — M34 may not close without them.
1. `List<Struct>` is sound: `DropList` recursively drops non-tensor elements (currently silently skipped → leak, `malus-codegen-cpu/src/lib.rs:1035-1037`). Type-directed recursion covers struct elements containing tensors and `List`s; a leak test (RC counters, M29 pattern) proves a dropped `List<Block>` frees every tensor inside every block.
2. Method dispatch on list elements works end-to-end: `for blk in self.blocks { let y = blk.forward(x) ... }` — static dispatch on the known element type through the M28 monomorphization path (believed to mostly work; the milestone verifies and closes gaps).
3. The optimizer-recursion pattern compiles and trains: per-submodule `adamw` application where each `Block`'s `parameters()` identity list receives the writes, plus a top-level application for GPT's own non-block tensors (wte, wpe, lm_head, final layernorm params). Optimizer state (m/v lists) is held per-submodule to match.
4. The no-unroll lint (`malus-cli/src/lint.rs`) is updated for the recursive form: still exactly one `Module`-generic optimizer `fn`, `.grad` reads confined to it, but its `.parameters()` may now be reached once per submodule at runtime via one call site. Lint remains green on the capstone and still fails on a hand-unrolled counterexample (add that negative test).
5. A 2-layer smoke-scale nanoGPT written with named submodules trains, loss decreases, no leaks; all prior gates green.
6. ADR-0036 written.
7. `cargo test --workspace` passes.

## Scope

### 1. Recursive DropList

Extend the drop lowering to recurse by element type: `Tensor` → `tensor_release` (existing), struct → field-wise drop (existing `DropStruct` logic, reused), nested `List` → recurse. `List<T>`'s container RC (ARC header, ADR-0034) is unchanged — this fixes only what happens when the container's refcount hits zero.

### 2. Grad-inference and borrow-inference over struct-element lists

Verify M27 grad-inference propagates through `List<Struct>` element field loads (`blocks[i].wq`) and M29 borrow-inference stays conservative there (list elements are already an RC-fallback case; no new demotion rules — measure the RC ratio after, gate stays ≤5%).

### 3. Capstone structure (assembled fully in M35)

```
struct Block { wq, wk, wv, wo, w1, w2, ln1_g, ln1_b, ln2_g, ln2_b, ... }
impl Module for Block { fn parameters(self) -> List<Tensor<f32>> { return self.params } }
struct GPT { blocks: List<Block>, wte, wpe, lm_head, lnf_g, lnf_b, ... }
```

(Exact field layout decided at implementation; the constraint is that every trainable tensor lives in exactly one submodule's identity list.)

## Out of Scope

- `concat` builtin (rejected for `parameters()`, ADR-0036; a general list-concat may return post-V5 for non-Module uses).
- Growable `List` (`push`) — still deferred.
- Generic structs, `state_dict` naming trees, nested `Module` trait bounds (`List<M: Module>`) — the recursion is over a concrete `List<Block>`; full trait-object-style heterogeneous module trees stay post-V5.
