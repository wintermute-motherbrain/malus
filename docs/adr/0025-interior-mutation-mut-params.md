# ADR-0025 — Interior Mutation Requires a Mutable Binding; `mut` Parameters Are Interior-Only Borrows

**Status:** Accepted (M20)

## Context

M20 adds indexed and field assignment (`a[i] = e`, `s.f = e`). ADR-0014 said aggregate parameters are never assign targets. We need to reconsider that for AdamW, which mutates `params`, `ms`, and `vs` arrays in place.

## Decision

**1. Interior mutation requires a mutable binding.**

The `let` keyword binds a deep-immutable value. `let a = [...]; a[0] = e` is a compile error. Interior mutation (`a[i] = e`, `s.f = e`) requires:
- A `let mut` local: `let mut a = [...]`, **or**
- A `mut` parameter (see below).

This preserves the invariant that `let` values are fully immutable and simplifies the reasoning burden for the programmer and for CTMM.

**2. `mut` parameters are interior-only borrows.**

A parameter declared `fn f(mut a: Array<T, N>)` permits interior mutation (`a[i] = e`) but rejects bare rebinding (`a = new_value`). This is because:
- Aggregate parameters already pass a shared heap pointer — the callee mutates through the same allocation the caller owns.
- Bare rebind would assign to the callee's local register copy of the pointer, which is invisible to the caller, AND would cause a double-free if the callee's CTMM tried to drop the parameter box (it would drop the caller's allocation).
- Interior mutation is safe: the heap pointer itself is borrowed, not moved. CTMM already does not free aggregate parameter boxes (the param is a borrow, not a move); this invariant is unchanged.

**3. `mut` parameter restriction: non-Variable aggregate params only.**

This amends ADR-0014 ("aggregate parameters are never assign targets"): interior mutation via `mut` params is now permitted. However, `Variable` struct fields remain unassignable post-V3 (ADR-0016 is unchanged).

**4. `**` operator (scalar power), lowered via `malus_powf`.**

A `**` (power) operator is added for `f32 ** {f32 | i32 | i64} → f32`. Non-integer exponents (e.g., `beta^t` in AdamW bias correction) require the full `powf` path — the spec's suggestion to "unroll for small integer exponents" is infeasible because the exponent (`t`) is a runtime loop counter. The operator is lowered to an `extern "C" fn malus_powf(f32, f32) -> f32` shim registered in the JIT. `**` is right-associative at the highest binary precedence (12, 11).

**5. Heap-aggregate aliasing exception.**

Malus's move semantics normally prevent aliasing. `mut` params are a deliberate exception: the heap pointer is shared between caller and callee for the duration of the call. This is safe because:
- Only the callee mutates (write conflicts are prevented by single-threaded execution).
- The callee never frees the box.
- The caller retains ownership and remains responsible for dropping the box when its scope ends.

## Amendments

- **ADR-0014** ("CTMM for conditional paths"): aggregate params are no longer restricted from assignment targets; `mut` params allow interior assignment.
- `**` operator: the spec used `^`; we ship `**` (Python parity, see ADR-0022).

## Consequences

- `let` = deep immutable. `let mut` = mutable binding (permits both rebind and interior mutation). `mut` param = interior-only borrow. Three clearly distinct mutability forms.
- CTMM does not emit drop stmts for Index/Field assign targets — old element release is performed inline in codegen (load old handle → release → store new) to keep the precise slot address computation at codegen time.
- RHS of element/field assignment is always evaluated before the old element is released (CTMM hoists GPU-producing RHS into a temp), preventing use-after-free when the RHS reads from the same slot.
