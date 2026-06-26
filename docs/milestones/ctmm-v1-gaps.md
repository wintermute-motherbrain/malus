# CTMM — v0.1 vs v1 Gaps

CTMM (Compile-Time Memory Management) is malus's automatic memory model: escape analysis inserts static `free` calls for tensor bindings at compile time, with reference counting as a fallback for structurally ambiguous lifetimes.

## What v0.1 implements

`malus-sema` implements **last-use analysis** — the simplest correct subset of CTMM:

1. Walk each `fn` body's statements, tracking where each local tensor binding (`let`) is last used.
2. If a binding does not escape the function (not returned), inject a `Drop` node immediately after its last-use statement.
3. If the binding was passed to a `KernelCall`, it is **in-flight** — inject a `GpuBarrier` node before the first `Drop` in that in-flight group, so codegen can emit the Metal sync before freeing.

This covers the MVP's `add_tensors.ml` perfectly: all tensor flows are linear (no heap, no closures, no non-trivial escapes).

## Gap Summary

| Gap | v0.1 behavior | V1 fix |
|---|---|---|
| Conditional last-use (if/else) | Unsound — last-use may be wrong per branch | M9: RC fallback for ambiguous paths |
| Loop-carried tensor lifetimes | Not possible (no loops in v0.1) | M9: loop-body tensors dropped each iteration |
| RC fallback for struct-stored tensors | Not possible (no structs in v0.1) | M10: tensor_retain on store, tensor_release on struct drop |
| Unbound temporaries from nested BinOps | Leaked — no Drop inserted | M11: fix during integration testing |
| Cross-function ownership chains | Correct for simple cases | Post-V1: interprocedural analysis |
| Closure captures | Not possible (closures not planned) | Post-V1 if closures added |
| `inout` parameter tracking | Not possible (inout is post-V1) | Post-V1: suppress Drop for inout inputs |

## Gap Detail

### 1. Conditional Last-Use (Branching) → M9

**Problem:** The linear scan sees one last-use per binding across the entire function. With `if`/`else`, the "last use" might be inside one branch but not the other.

**V1 fix (M9):** RC fallback for conditional paths. When a tensor binding's last use is ambiguous due to branching, CTMM emits `Retain` at the branch entry and `Release` at the end of each branch that is done with the binding. Static `Drop` is preserved for bindings whose last use is unambiguously in the linear part of the code. Full dataflow liveness analysis (which would reduce how often RC is needed) is a V2 optimization.

### 2. Loop-Carried Tensor Lifetimes → M9

**Problem:** Not possible in v0.1 (no loops), but: tensors created inside a loop body must be freed at the end of each iteration, not just at the end of the function.

**V1 fix (M9):** Treat each loop body as a scope. Tensors created inside (`Let` statements in the body) are `Static`-dropped at the end of each iteration. Tensors created outside and used inside use RC fallback if their use inside the loop is ambiguous.

### 3. RC Fallback for Struct-Stored Tensors → M10

**Problem:** When a tensor is stored in a struct field, its lifetime is structurally ambiguous — the struct controls when it dies. CTMM cannot statically determine the drop point.

**V1 fix (M10):** When a tensor binding is stored into a struct field during construction, emit `Retain`. When a struct binding goes out of scope, emit `Release` for each tensor field. Nested structs recurse. This uses the `tensor_retain`/`tensor_release` C ABI added in M9.

### 4. Unbound Temporaries from Nested GPU Expressions → M11

**Problem:** When `a + b * c` is lowered in a `fn` body, the inner `b * c` result is a temporary tensor that CTMM never frees (it's not a named binding). The `hoist_gpu_subexprs` pass creates synthetic `let __tmp_N = ...` bindings, but `find_last_uses` may not insert `Drop` for all of them correctly.

**V1 fix (M11):** During integration testing of the backward pass expressions (many chained matmuls and transposes), trace all hoisted temporaries and verify `Drop` is inserted. Fix any remaining leaks before the done-when is considered passing.

### 5. Cross-Function Ownership Chains → Post-V1

**Problem:** v0.1 analyzes each function body independently. Ownership tracking across more than two call frames may be incorrect in edge cases.

**V1 mitigation:** Tensors that flow through struct fields (cross-function via struct parameters) use RC, which handles the cross-frame case correctly. Post-V1: interprocedural escape analysis to reduce RC usage for non-struct cross-function flows.

### 6. Closure Captures → Post-V1

Closures are not planned for V1. When added, a tensor captured by a closure must be treated as escaped.

### 7. `inout` Parameter Tracking → Post-V1

`inout` is post-V1. When implemented, CTMM must suppress `Drop` for `inout` inputs — the caller retains ownership of the buffer.
