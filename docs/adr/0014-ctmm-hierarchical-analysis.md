# CTMM M9: hierarchical scope analysis eliminates RC for control flow

The original M9 plan (written against the flat linear scan) said to use RC fallback whenever a tensor binding's last use was ambiguous due to branching. With a hierarchical analysis, this is unnecessary — `if`/`else`/`for`/`while` as *statements* (not expressions) never create genuinely ambiguous drop points.

## Decision

Replace the flat linear scan in `annotate_body` with a hierarchical one:

1. **Recurse first.** Before running any outer-scope passes, recursively call `annotate_body` on each inner scope (if/else branches, loop bodies). This gives inner bindings their own Drops and Barriers.

2. **Outer scope treats control flow nodes as opaque use sites.** `collect_idents_in_stmt` recurses into inner bodies so that `find_last_uses` correctly records the outer position of the `If`/`For`/`While` node as the last-use site for any outer-scope binding referenced inside. `Drop` is then inserted after the control flow node in the outer body — which is always correct regardless of which branch ran or how many iterations executed.

3. **M9 emits zero RC nodes.** `tensor_retain` and `tensor_release` are added to the runtime ABI and `TypedStmt` (M10 needs them for struct fields), but M9's CTMM analysis emits only `Drop`, `GpuBarrier`, and `Retain`/`Release` (the latter two left unused until M10).

4. **`insert_assign_drops` drops `locals.contains(name)` guard.** For loop-carried tensor mutation (`let mut acc; for ...: acc = acc + delta`), the inner scope's `insert_assign_drops` must drop the old tensor before each reassignment. Since the type checker already guarantees that only `let mut` bindings appear as Assign targets (never parameters), the `locals` check is unnecessary and dropping it gives correct Drop-before-Assign behavior for outer-scope bindings reassigned inside loops.

5. **Precise barrier insertion at control flow boundaries.** The outer `insert_barriers` pass, when it encounters a control flow node with a non-empty pending set, recursively checks whether any pending name is referenced inside the node's bodies. If yes, a `GpuBarrier` is emitted before the node. This avoids unnecessary barriers while ensuring CPU reads of in-flight tensors inside any branch are always preceded by a sync.

## Why this is surprising

The ctmm-v1-gaps doc and the M9 spec milestone explicitly planned RC for conditional paths. A future reader seeing `annotate_body` emit no `Retain`/`Release` nodes for M9 control flow would reasonably assume this is a bug or an incomplete implementation. It is deliberate: the hierarchical placement of `Drop` nodes after control flow statements makes RC unnecessary for `if`/`else`/`for`/`while` as statements.

RC remains necessary for M10+ (struct fields, array elements) where a tensor's lifetime is decoupled from any lexical scope. The runtime ABI additions are included in M9 so M10 doesn't need to change the ABI.

## Considered alternatives

- **RC for all conditional paths (original plan):** Correct, but adds atomic refcount overhead to every tensor that flows through a branch or loop — the hot path in a training loop. The hierarchical approach preserves static Drops for all M9 constructs.
- **Dataflow liveness analysis:** Would further reduce RC usage post-V1 by precisely computing live ranges across branches. Deferred as a V2 optimization per ADR-0002.
