# M2 — Semantics

**Crate:** `malus-sema`
**Done when:** `malus-sema` type-checks the M1 AST for `add_tensors.malus` without errors, annotates every expression with its resolved type and placement, produces a typed IR with CTMM last-use free-point annotations, and the CLI prints the typed IR.

## Design decisions

| Decision | Choice |
|---|---|
| Output representation | New `TypedProgram` IR — not an annotated AST or side-table |
| Import items | Stripped from typed IR — only `TypedFn` and `TypedKernel` |
| Type checking strategy | Bidirectional — explicit signatures, infer `let` from RHS |
| Type representation | Separate `ResolvedTy` enum in malus-sema (not the AST's `Ty`) |
| Tensor placement in types | NOT in the type — tracked as `Option<Placement>` on bindings and `TypedExpr` |
| Mixed placement in binary exprs | Type error |
| CTMM scope for v0.1 | Last-use analysis only (see [ctmm-v1-gaps.md](ctmm-v1-gaps.md)) |
| Free-point representation | Explicit `TypedStmt::Drop` nodes injected at last-use points |
| GPU barrier representation | Explicit `TypedStmt::GpuBarrier` nodes before first `Drop` of any in-flight group |
| Kernel call distinction | Separate `TypedExprKind::KernelCall` variant (distinct from `Call`) |
| Error reporting | Collect errors across items, bail within a body when inference is unreliable |
| Symbol resolution order | Two-pass: collect all signatures first, then check bodies (order-independent) |
| Import naming conflicts | Errors — `as` aliasing deferred to v1 ([m7-v1-features.md](m7-v1-features.md)) |
| Builtin functions | Registered in a `HashMap<String, BuiltinSig>` table (extensible) |
| Tensor ops in fn vs kernel | Same typing — codegen distinguishes context via `TypedFn` vs `TypedKernel` |
| Tensor literal coercion | Int literals widen to float losslessly; lossy coercion (float into int tensor) is an error |

## Scope

### Type checker (`crates/malus-sema/src/check.rs`)

Two-pass structure:

**Pass 1 — collect signatures:**
- Walk all items, convert AST `Ty` → `ResolvedTy` for params and return types
- Register `FnSig` and `KernelSig` in the environment
- Detect duplicate definitions (including from-import conflicts)
- Detect missing `fn main()`

**Pass 2 — check bodies:**
- Each fn/kernel body checked with params pre-bound in a fresh scope
- `let` bindings: synthesize type from RHS, bind in scope with placement
- `return`: check against declared return type
- Binary ops: check dtype match, placement match (same-placement required for tensor operands)
- Calls: resolve callee (fn/kernel/builtin), check arg types, emit `Call` or `KernelCall`
- Tensor literals: check elements against declared dtype; int→float widening allowed, float→int rejected
- Qualified calls (`ops.add`): resolve via module aliases

### CTMM last-use analysis (`crates/malus-sema/src/ctmm.rs`)

Runs on `TypedFn` bodies after type checking:

1. Collect local `let` bindings
2. Collect escaping bindings (those appearing in `return` exprs)
3. Find the last-use statement index for each non-escaping local tensor binding
4. Group bindings by their last-use index
5. Inject `GpuBarrier` (if any binding in the group is in-flight) + `Drop` nodes after the last-use statement

**For `add_tensors.malus`:**
- `a`, `b` last used at the `KernelCall` → `GpuBarrier` + `Drop(a)` + `Drop(b)` after it
- `c` last used in `print(c)` → `Drop(c)` after that statement (no barrier — `print` is not a kernel)

### Typed IR (`crates/malus-sema/src/typed_ir.rs`)

```
TypedProgram { fns: Vec<TypedFn>, kernels: Vec<TypedKernel> }

TypedStmt:
  Let { name, expr }  |  Return { expr }  |  Expr(expr)
  Drop { name }       |  GpuBarrier

TypedExpr { kind: TypedExprKind, ty: ResolvedTy, placement: Option<Placement>, span }

TypedExprKind:
  Lit | Ident | BinOp | Unary
  Call { callee, args }
  KernelCall { callee, args, in_flight: Vec<String> }
  Index | TensorLiteral | FieldAccess

ResolvedTy:
  Tensor { dtype } | Scalar(ScalarTy) | Bool | Tuple(Vec<ResolvedTy>) | Unit
```

## Tests

See `crates/malus-sema/src/tests.rs`. All 12 tests pass:

- MVP round-trip: `add_tensors.malus` type-checks, `c` resolves to `Tensor<f32>`, CTMM inserts correct Drop/GpuBarrier nodes
- Escaped tensor gets no Drop
- Int literal in `Tensor<f32>` → coerced (ok)
- Float literal in `Tensor<i32>` → LossyCoercion error
- Dtype mismatch in binary op
- Unknown identifier → UnknownIdent error
- Argument count mismatch → ArgCountMismatch error
- Return type mismatch
- Duplicate definition
- Missing `fn main()` → MainNotFound error
- Kernel called from kernel → KernelCalledFromKernel error
- Qualified call via module alias (`ops.add`)

## Out of scope for M2

- Struct/enum type checking (M7a)
- RC fallback for structurally ambiguous lifetimes (M7c — see [ctmm-v1-gaps.md](ctmm-v1-gaps.md))
- Borrow checking for `inout` parameters (M7f)
- `@` matmul dtype compatibility
- Branching/liveness analysis for if/else
- GPU intrinsic validation in kernel bodies
- Import aliasing (`as` syntax) — see [m7-v1-features.md](m7-v1-features.md) M7g
