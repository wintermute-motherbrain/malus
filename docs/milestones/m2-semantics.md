# M2 — Semantics

**Crate:** `malus-sema`
**Done when:** `malus-sema` type-checks the M1 AST for `add_tensors.malus` without errors, annotates every expression with its type, and produces a typed IR with Lobster free-point annotations for both input tensors.

## Scope

### Type checker

Walk the typed AST and resolve types for every expression and binding.

**Rules for the MVP:**
- `fn` and `kernel` parameters and return types are explicitly annotated — take them as ground truth
- `let` bindings infer their type from the right-hand side expression
- `Tensor.gpu<f32>(...)` → `Tensor<f32>` with placement `Gpu`
- `a + b` where `a: Tensor<f32>`, `b: Tensor<f32>` → `Tensor<f32>` (element-wise)
- Function call `add(a, b)` → look up `kernel add`, check argument types match parameter types, return the declared return type
- `return expr` — check `expr` type matches the declared return type

**Type errors to detect:**
- Dtype mismatch in binary op (`Tensor<f32> + Tensor<f16>`) — panic with span and both dtypes
- Argument count mismatch in call
- Unknown identifier
- Return type mismatch

### Lobster escape analysis (linear flows only)

For the MVP, only handle the case where lifetimes are fully linear (no heap storage, no closures — those fall back to RC in v1). The analysis must:

1. Build a **use chain** for each tensor binding: where it is created, where it is used, and where it is last used
2. Insert a **free point** immediately after the last use of each tensor that does not escape the function (is not returned, not stored in a struct, not captured)
3. For tensors passed to a `kernel` call, mark them **in-flight** — their free point must be preceded by a GPU barrier (handled by M4/M5; M2 just annotates them)

**For `add_tensors.malus`:**
- `a` last used as argument to `add(a, b)` → free after GPU barrier following kernel return
- `b` same
- `c` last used in `print(c)` → free after `print` returns

### Typed IR

Output a `TypedProgram` that mirrors the AST but with:
- Every `Expr` annotated with its resolved `Type`
- Every tensor binding annotated with its `Placement`
- `FreePoint` nodes inserted by Lobster at last-use sites
- `KernelCall` nodes tagged with the list of in-flight tensors

## Tests

- Type-check `add_tensors.malus` → no errors, `c` resolves to `Tensor<f32>`
- Dtype mismatch → error at correct span citing both dtypes
- Lobster annotates `a` and `b` as in-flight after the `add(a, b)` call
- Lobster annotates `c` as freed after `print(c)`
- Unknown identifier → error naming the identifier and its span

## Out of scope for M2

- Struct/enum type checking
- RC fallback for structurally ambiguous lifetimes
- Borrow checking for `inout` parameters
- Checking `@` matmul dtype compatibility (no matmul in MVP stdlib)
