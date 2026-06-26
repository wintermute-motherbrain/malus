# Tensor BinOp in host `fn` bodies is not lowered in M3

Tensor binary operations (`a + b`, `a * b`, etc.) appearing inside `fn` bodies currently return `CodegenError::UnsupportedExpr` rather than being lowered to Cranelift IR. This is a deliberate deferral, not an oversight.

## The problem

On first glance, lowering `a + b` on tensors seems straightforward: add a `tensor_binop(lhs, rhs, op)` runtime stub and call it from the JIT'd code. But this commits to CPU-side element-wise execution, which is the wrong default for Apple Silicon.

On Apple Silicon with `MTLResourceStorageModeShared`, GPU tensors share memory with the CPU — there is no copy cost to GPU dispatch. The GPU has thousands of ALUs available; executing tensor arithmetic single-threaded on the CPU is almost always slower than dispatching to GPU, even for small tensors. A `tensor_binop` stub backed by a CPU loop would be correct but wrong.

## The language design question

Tensor arithmetic in `fn` bodies should ultimately lower to one of:

1. **MPS (Metal Performance Shaders)** — for stdlib ops like matmul, reductions (per ADR-0005)
2. **A built-in MSL kernel** — for element-wise ops like `+`, `*`, etc., dispatched the same way user kernels are

Both options require infrastructure that does not exist until M5 (GPU codegen) or later. Picking a CPU fallback now would set a precedent that is hard to remove.

## Decision

`BinOp` on `Tensor` types in `fn` bodies returns `UnsupportedExpr` in M3. The `import_demo` example (which uses tensor arithmetic inside `fn add`) does not work until this is resolved.

## When to revisit

Deferred to M5.1 (follow-up to M5 GPU codegen). M5 delivers the `KernelRegistry` and `kernel_dispatch` infrastructure that built-in element-wise kernels require. At that point, tensor BinOp in `fn` bodies should lower to a `kernel_dispatch` call to a built-in element-wise kernel, consistent with how user-written kernels are dispatched. See `docs/milestones/m5.1-builtin-elementwise-kernels.md`.

## Considered Options

- **CPU `tensor_binop` stub**: Rejected — establishes CPU-side tensor arithmetic as the default, wrong for Apple Silicon.
- **Lower to `kernel_dispatch` immediately**: Rejected for M3 — `kernel_dispatch` is a no-op stub returning an empty tensor in M3; the result would be silently wrong.
- **Defer with `UnsupportedExpr`**: Chosen — honest about the gap, forces the language design decision to be made with the right infrastructure in place.
