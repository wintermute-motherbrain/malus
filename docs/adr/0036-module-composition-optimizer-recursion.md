# Module composition: optimizer recursion over per-submodule identity lists; parameters() concat rejected

## Decision

V5 (M34) adds named submodules — `GPT { blocks: List<Block>, ... }` with `impl Module for Block` — without changing the `Module` trait's contract. Composition works by **recursion, not aggregation**: the generic optimizer is applied per submodule (`for blk in model.blocks { adamw over blk.parameters() }`, plus once over the top-level module's own tensors). Each `parameters()` call still returns that one module's stored `List<Tensor<f32>>` **by identity** (ADR-0034), and the optimizer's slot writes (`ps[i] = variable(...)`) land in the submodule's own field.

A concatenating alternative — `GPT.parameters()` returns sub-lists merged into one flat list, PyTorch-style — is rejected for V5, and no `concat` builtin is added.

Supporting work this decision forces (M34): `DropList` becomes type-directed and recursive so `List<Struct>` elements are actually freed (today they silently leak — `malus-codegen-cpu/src/lib.rs:1035-1037` drops only tensor elements); the no-unroll lint learns that the single optimizer call site may execute once per submodule.

## Why this is surprising

PyTorch users expect `model.parameters()` to yield *every* parameter in the tree, and the obvious implementation is concatenation. In malus that obvious implementation trains silently wrong: ADR-0034's write-back mechanism works because `parameters()` returns an aliased identity list — the optimizer mutates the very list the model reads on its next forward. A concatenated list is a fresh snapshot; the optimizer would faithfully update the snapshot while the model's weights never change, with no error, no NaN, just a loss curve that quietly stops improving. That failure mode (documented as the central hazard in ADR-0034) is why the composition rule is "never merge identity lists," even though it makes malus's `parameters()` narrower than PyTorch's.

ADR-0022 (API parity) is respected in the letter that matters: nothing shipped here *breaks* a future PyTorch-shaped `parameters()`. If a later version adds true tree aggregation (e.g. an iterator/visitor that yields tensor slots rather than a materialized list, or in-place parameter mutation that removes the slot-write mechanism entirely), it lands additively; the recursive form keeps working.

## Considered alternatives

**Concat with re-scatter.** Merge for the optimizer, write results back into each sub-list afterward. Rejected: doubles the bookkeeping, the re-scatter is exactly the kind of index arithmetic the milestone exists to remove, and it silently breaks the moment a module holds a tensor not covered by the scatter map.

**In-place parameter update ABI** (mutate the tensor's buffer instead of replacing the list slot). Removes the identity requirement entirely and is closer to how PyTorch optimizers actually work. Rejected for V5: it rewrites the optimizer/autograd-leaf contract (`variable()` identity, `.grad` slots keyed by handle) and CTMM reassignment lowering (ADR-0011) in one motion — far more invasive than the recursion, for the same capstone outcome. Worth revisiting alongside V6 fusion work if profiling shows slot-replacement overhead.

**Flat 72-tensor list with index arithmetic** (no language work at all). Works today, lint-clean. Rejected as the capstone form: `params[l*12 + WQ]` is precisely the hand-unrolled approximation V4 was chartered to kill, and the capstone source is the published artifact.

## Consequences

- Optimizer state (Adam m/v lists) is held per-submodule, mirroring the parameter lists. The optimizer signature grows a recursion pattern; the no-unroll lint's "exactly one `.parameters()` call site" rule is reinterpreted as one *call site*, executed once per submodule.
- `List<Struct>` becomes a supported, leak-free type — a general language capability, not just a Module feature.
- Heterogeneous module trees (`List<M: Module>` trait-object style) are NOT enabled; the recursion is over a concrete element type (`List<Block>`). Deep nesting beyond one level composes by the same rule (each level recurses) but is not exercised by the capstone.
- No `concat` builtin exists, so nothing tempts a future `parameters()` implementation into the snapshot bug. If general list concatenation is added later for non-Module uses, this ADR is the warning label on using it for `parameters()`.
- `state_dict`-style named parameter trees (checkpointing, V6) will follow the same recursion shape: per-submodule traversal, never a merged flat registry.
