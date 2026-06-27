# M11 — The 2-Layer MLP

**Crates:** All crates.

Put it all together. Fixed-length arrays, 2-D tensor literals, rich error diagnostics, recursive aggregate drops, and the V1 capstone: a 2→8→1 MLP that learns XOR.

## Done-When

`examples/xor.ml` compiles and runs on an M-series Mac, printing decreasing loss over 10k training steps with final predictions that round to `0, 1, 1, 0`:

```
step 0: loss = [1.0402256]
step 500: loss = [0.006860058]
step 1000: loss = [0.001959838]
step 2000: loss = [0.0007679773]
step 5000: loss = [0.00026143057]
step 9999 (final): loss = [0.00012191984]
predictions: [0.0056860363, 0.99518234, 0.9939387, 0.005444207]
```

## Scope

### 1. Fixed-Length Arrays

`Array<T, N>` — heap-allocated uniform 8-byte slots, `ForIn`, single-index, recursive per-element drop.

**AST:** `ExprKind::ArrayLiteral { elements }`, `StmtKind::ForIn { var, iter, body }`, `Ty::Array { elem, len }`.

**Parser:** `[expr, ...]` in expression position → `ArrayLiteral`. `for x in arr: body` → `ForIn` (dispatched from the existing `for` parser after the `range(...)` fast-path fails).

**Type system:** `ResolvedTy::Array { elem, len }`. Element type unified from first element; remaining checked to match. `ForIn` binds `var: elem_ty`.

**Codegen-cpu:** `ArrayLiteral` → `malloc(N*8)` + store each element. `Index` → `base + i*8` load. `ForIn` → counted loop 0..N with SSA-correct `use_var(idx)` in body block.

**CTMM:** `DropArray { name, elem_ty, len }` emitted for array bindings at last use. Codegen emits a counted loop releasing each element recursively before calling `heap_free`.

### 2. 2-D Nested Tensor Literals

`Tensor.gpu<f32>([[r0c0, r0c1], [r1c0, r1c1]])` — row-major, validated rectangular, produces a real 2-D shape buffer passed to `tensor_alloc_gpu(dtype, shape_ptr, ndims, data)`.

**Parser:** `parse_tensor_rows` inside `parse_tensor_literal` owns all brackets and validates rectangularity (row count × cols = total elements). Disjoint from `parse_primary`'s `[` → `ArrayLiteral` path.

**Sema:** `TensorShapeMismatch` error if `product(shape) != elements.len()`.

**Codegen-cpu:** `TensorLiteral` emits real `ndims` and N-slot shape stack buffer instead of the old hardcoded `ndims=1`.

### 3. Rich Error Diagnostics

`ariadne` crate renders `SemaError` and `ParseError` with source context, spans, and underlines. Each error variant maps to a labeled `Report` with a `help` annotation.

**Runtime panics:** `tensor_matmul`, `tensor_transpose`, non-f32 dtype panics now include actual shapes and dimension info.

### 4. Recursive Aggregate Drops + DropEnum

**DropEnum:** `TypedStmt::DropEnum { name, variants }` emitted by CTMM for enum bindings at last use. Codegen emits a tag-check dispatch: load tag → branch to the matching variant's payload-release block → shared `heap_free`.

**Recursive DropStruct:** `DropStruct` now carries `Vec<(usize, ResolvedTy)>` field info so codegen can load child pointers and recursively drop aggregate-typed fields before freeing the parent box.

**Aggregate reassign fix:** `insert_assign_drops` and D6 hoist guard now fire for Struct/Enum/Array targets (not just Tensor), so `model = MLP(...)` inside a loop correctly drops the old value on each iteration.

### 5. Temp-Leak Hoist

CTMM `hoist_gpu_subexprs` hoists GPU-producing BinOp/Unary operands into named `__malus_tmp_N` Lets so `find_last_uses`/`insert_drops` can free them. Fixes leaks from nested expressions like `x @ w1 + ones41 @ b1` where `x @ w1` and `ones41 @ b1` were previously invisible to CTMM.

D6 guard: `hoist_gpu_subexprs` also hoists Assign RHS to a temp when it is GPU-producing+tensor, breaking self-reference cycles (`w1 = w1 - lr*dw1`).

### 6. Early-Return Unwind

CTMM's `inject_early_return_unwinds` detects `TypedStmt::Return` nested inside `If`/`For`/`While`/`ForIn`/`Match` bodies and emits `Drop`/`DropStruct`/`DropEnum`/`DropArray` for all live enclosing-scope bindings not referenced by the return expression.

### 7. GPU Kernel Collection Inside Control Flow

`collect_binops_in_stmt` and `collect_unary_builtins_in_stmt` in `malus-codegen-gpu` now recurse into `If`/`For`/`While`/`ForIn` bodies. Previously they skipped control-flow nodes, so any BinOp or unary builtin used inside a loop body was not compiled as a Metal kernel — the runtime would panic with "unknown kernel".

### 8. XOR MLP (Capstone)

`examples/xor.ml`: 2→8→1 sigmoid-sigmoid MLP, MSE loss, `sigmoid_backward` user kernel, manual weight updates, 10k steps, lr=1.5, asymmetric init.

Architecture:
- **Hidden:** sigmoid with 8 units
- **Output:** sigmoid with 1 unit
- **Bias:** `ones41 @ b1` trick (4×1 column of ones matmul'd with 1×H bias row)
- **Backward:** `sigmoid_backward(grad, sig_z) = grad * sig_z * (1 - sig_z)` kernel reused for both output and hidden layers

## Out of Scope

- SafeTensors loading
- GPU RNG
- nanoGPT or any model larger than a 2-layer MLP
- Autograd / gradient tape
- Optimizer objects
- ScalarBroadcast typed IR node (inline scalar-broadcast BinOps continue to work; dedicated IR node deferred)
