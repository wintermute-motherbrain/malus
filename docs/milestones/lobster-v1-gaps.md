# Lobster Memory Model — v0.1 vs v1 Gaps

## What v0.1 implements

`malus-sema` implements **last-use analysis** — the simplest correct subset of the Lobster model:

1. Walk each `fn` body's statements, tracking where each local tensor binding (`let`) is last used.
2. If a binding does not escape the function (not returned), inject a `Drop` node immediately after its last-use statement.
3. If the binding was passed to a `KernelCall`, it is **in-flight** — inject a `GpuBarrier` node before the first `Drop` in that in-flight group, so codegen can emit the Metal sync before freeing.

This covers the MVP's `add_tensors.malus` perfectly: all tensor flows are linear (no heap, no closures, no non-trivial escapes).

## What is NOT implemented (v1 gaps)

### 1. RC fallback for structurally ambiguous lifetimes

**Gap:** When a tensor is stored in a struct field, a dynamic array, or any heap-allocated container, its lifetime cannot be determined statically. Lobster's full model falls back to reference counting in these cases.

**Current behavior:** Struct types are not supported in v0.1, so this case cannot arise. When structs are added in v1, tensors stored in struct fields will need RC insertion.

**What's needed for v1:**
- After type checking, classify each tensor binding: `Static` (last-use deterministic) or `Rc` (stored in a struct/container).
- For `Rc` bindings, emit `rc_retain()` / `rc_release()` calls instead of `Drop`.
- Runtime (`malus-runtime`) must implement a lightweight RC mechanism for tensor buffers.

### 2. Escape through function return (cross-function analysis)

**Gap:** v0.1 only tracks escapes within a single function body. A tensor returned from `make()` and used in `main()` is correctly not dropped in `make()` — but its lifetime in `main()` is tracked independently. This is correct for v0.1 (the caller receives ownership and last-use analysis runs on `main()`'s body separately). However, it will not correctly handle cases where ownership chains across more than two frames.

**What's needed for v1:** Interprocedural escape analysis to track tensor ownership across call boundaries. In practice, this is only needed if a tensor is passed into a function that may or may not store it.

### 3. Closure captures

**Gap:** If malus adds closures or higher-order functions in v1, a tensor captured by a closure escapes its lexical scope. The current last-use analysis does not detect this.

**What's needed for v1:** Closure capture analysis — any binding referenced in a closure body is treated as escaped.

### 4. `inout` parameter tracking

**Gap:** `inout` kernel parameters (v1 feature) mutate tensors in-place. Lobster should not insert a `Drop` for the input buffer in this case — it is the same buffer as the output. v0.1 does not handle `inout` because it is not in the MVP language.

**What's needed for v1:** When a tensor is passed as `inout` to a kernel, mark it as "reused" — suppress `Drop` for that binding.

### 5. Conditional last-use (branching)

**Gap:** If/else branches are parsed but not yet exercised in the MVP. If a binding is used in one branch but not another, its true last-use may depend on the branch taken. v0.1's linear last-use analysis does not account for this.

**What's needed for v1:** A proper liveness analysis (dataflow) that computes last-use per-branch and inserts `Drop` at join points where liveness drops to zero.

## Summary table

| Scenario | v0.1 | v1 needed |
|---|---|---|
| Linear tensor flows in `fn main()` | Correct | — |
| Tensor escapes via `return` | Correctly suppressed | — |
| In-flight tensors via kernel dispatch | GpuBarrier + Drop | — |
| Tensor stored in struct field | Not possible (structs are v1) | RC fallback |
| Closure captures | Not possible (closures are v1) | Capture analysis |
| `inout` kernel parameters | Not possible (inout is v1) | Suppress Drop |
| Branching / if-else liveness | Unsound (last-use may be wrong) | Dataflow liveness |
| Cross-function ownership chains | Correct for simple cases | Interprocedural analysis |
