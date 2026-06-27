# CTMM reassignment lowering: guard-hoist then Drop-before-Assign

When a `let mut` binding is reassigned (`acc = acc + delta`), CTMM must free the old allocation before the new one is bound. The naïve lowering — emit `Drop{acc}` immediately before `Assign{acc}` — is a use-after-free: the RHS `acc + delta` still reads the old `acc`.

The implementation uses a two-step approach: guard-hoist first, then Drop.

## Decision

1. **`hoist_gpu_subexprs` runs first.** The existing hoisting pass lifts tensor BinOp RHSs into temporaries. For `acc = acc + delta`, it produces `let __t0 = acc + delta; acc = __t0`. After hoisting, the Assign's RHS no longer references `acc`, making a Drop of the old `acc` safe.

2. **Guard inside `hoist_gpu_subexprs`.** After hoisting subexpressions, if an Assign's RHS is still GPU-producing and references the target name, it is hoisted into one more temp. This covers edge cases where the BinOp hoisting did not already eliminate the self-reference.

3. **`insert_assign_drops` runs after hoisting.** A dedicated pass scans the body for `Assign` nodes and inserts `Drop{name}` immediately before each one where the target is a non-escaping local tensor binding. This is a plain `TypedStmt::Drop` node — not embedded in `Assign` codegen — so `insert_barriers` sees it and auto-inserts a `GpuBarrier` before freeing a buffer that is still GPU-in-flight.

4. **`TypedStmt::Assign` reaching codegen-cpu is a pure rebind.** `def_var` on the existing Cranelift `Variable`, no `next_var` bump, no free call. All memory management is handled by the CTMM Drop nodes above.

5. **The hoisted temp is "escaping".** After guard-hoist, `acc = __t0` means `__t0`'s allocation is moved into `acc`. `collect_escaping` is extended to mark tensor idents in Assign RHSs as escaping, preventing a double-free when CTMM later drops `__t0` (the Drop for `acc` covers it).

## Why this is surprising

The spec for M7 originally said "emit Drop before Assign." That is a use-after-free when the RHS reads the target — a fact that is only obvious once you trace the hoisting order. The guard-hoist is the real fix; the Drop-before-Assign is only correct after hoisting has eliminated the self-reference.

## Trade-off

Each reassignment of an in-flight GPU tensor causes a `GpuBarrier` (the Drop triggers it). Two `acc = acc + delta` calls in a loop serialize into two barrier points. This is acceptable for V1 where correctness is the goal; barrier coalescing is a post-V1 optimization.

## Why not a codegen-cpu intrinsic for Assign?

ADR-0008 requires codegen-cpu to stay Metal-unaware. Embedding a `gpu_barrier + tensor_free` sequence inside the Assign codegen path would require codegen-cpu to know whether the old value is GPU-in-flight — a Metal concept. The Drop-node approach keeps the barrier decision in CTMM, which already understands GPU-pending state.
