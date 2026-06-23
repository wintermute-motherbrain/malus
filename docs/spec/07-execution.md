# 07 — Execution Model

## Compilation pipeline

Every malus execution — script or REPL — passes through the same pipeline:

```
source
  │
  ▼
malus-syntax::parse()
  Lexer emits tokens including synthetic INDENT/DEDENT.
  Parser produces a typed AST (Program).
  │
  ▼
malus-sema::check()
  Type checker annotates every expression with its resolved type.
  Lobster escape analysis annotates tensor bindings with free points
  and marks in-flight tensors at kernel call sites.
  Produces: TypedProgram
  │
  ├──► malus-codegen-gpu::compile_kernels()
  │      Walks all `kernel` items in TypedProgram.
  │      Emits MSL source for each kernel.
  │      Returns: KernelRegistry (kernel_id → MSL string)
  │            │
  │            ▼
  │      malus-runtime::load_kernels(registry)
  │            Compiles each MSL string via Metal's newLibraryWithSource.
  │            Caches MTLComputePipelineState per kernel_id.
  │
  ▼
malus-codegen-cpu::compile_and_run(typed_program)
  Lowers `fn` items to Cranelift IR.
  Emits calls to the malus-runtime C ABI for tensor ops and kernel dispatch.
  JIT-compiles to native aarch64.
  Calls fn main().
```

There is no ahead-of-time compilation mode in v1. The full pipeline runs on every invocation. Cranelift's fast JIT compilation keeps startup latency low.

## Script execution `[MVP]`

```sh
malus script.malus
```

Runs the full pipeline and calls `fn main()`. Exit code 0 on success; 1 on any error (parse, type, runtime panic).

Errors are printed to stderr:

```
error: dtype mismatch in binary op
  --> script.malus:5:13
   |
 5 |     let c = a + b
   |             ^~~~^ left is Tensor<f32>, right is Tensor<f16>
   |
   = help: cast with b.to<f32>()
```

Runtime panics (shape mismatches, OOM, invalid index) also print to stderr with the concrete values involved:

```
panic: shape mismatch in matmul
  left:  [32, 64]
  right: [128, 64]
  dim 1 of left (64) must equal dim 0 of right (128)
```

## Terminal REPL `[v1]`

```sh
malus
```

Drops into an interactive session. Each input is compiled and executed incrementally using the same JIT pipeline.

```
malus> let a = Tensor.gpu<f32>([1.0, 2.0, 3.0])
malus> let b = a * 2.0
malus> print(b)
[2.0, 4.0, 6.0]
malus>
```

**Block entry:** A line ending with `:` puts the REPL into block mode. Subsequent lines are collected until a blank line is entered, then the full block is compiled and run.

```
malus> fn double(x: Tensor<f32>) -> Tensor<f32>:
...      return x * 2.0
...
malus> print(double(a))
[2.0, 4.0, 6.0]
```

**State:** The REPL maintains a persistent GPU context across inputs. Tensor bindings remain alive until the session ends or the binding is explicitly reassigned.

**Errors:** Errors in one REPL input do not terminate the session. The error is printed and the REPL continues.

## Jupyter kernel `[v1, priority follow-up]`

malus provides a Jupyter kernel so users can work in notebooks. The kernel implements the Jupyter messaging protocol (ZMQ) and delegates execution to the same JIT pipeline.

Cell semantics: each cell is compiled and executed as a block. Bindings persist across cells within a session. Tensor display shows shape, dtype, and a preview of values inline.

## Error model

malus uses a panic-only error model. Errors are not recoverable at the language level.

### Compile-time errors

Detected during parsing or type checking. Always include:
- Source file, line, and column
- The offending expression or token underlined
- A plain-English description of what went wrong
- A `help:` suggestion where applicable

Multiple errors may be reported in a single pass (the type checker continues after a non-fatal error to surface additional issues).

### Runtime panics

Triggered by:
- Shape mismatch in any operation
- Out-of-memory on GPU
- Invalid index (out of bounds)
- Dtype mismatch detected at runtime (e.g. from a loaded file)

Runtime panics always print the concrete values that caused the failure (shapes, dtypes, indices) before aborting.

### User-level error handling

Users who need explicit failure handling can use `Option<T>` (v1). There are no exceptions and no `Result` type in the language. Functions that might not have a result return `Option<T>`; functions that must succeed or die panic.
