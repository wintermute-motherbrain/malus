# M6 — Integration

**Crate:** `malus-cli`
**Done when:** Running `malus examples/add_tensors.malus` on an M-series Mac prints `[6.0, 8.0, 10.0, 12.0]` and exits cleanly.

## Scope

Wire all five previous milestones into a single end-to-end pipeline in `malus-cli`:

```
source file
    │
    ▼
malus-syntax::parse()          → Program (AST)
    │
    ▼
malus-sema::check()            → TypedProgram (with CTMM free-point annotations)
    │
    ├──► malus-codegen-gpu::compile_kernels()  → KernelRegistry (MSL strings)
    │         │
    │         ▼
    │    malus-runtime::load_kernels()         → compiled MTLComputePipelineStates
    │
    ▼
malus-codegen-cpu::compile_and_run()           → JIT fn main(), execute
    │
    ▼
  stdout: [6.0, 8.0, 10.0, 12.0]
```

### CLI entry point

```
malus <path>     — run a script
malus            — print usage (REPL placeholder)
```

Error handling at each stage: if any stage returns an error, print the diagnostic to stderr and exit with code 1. Errors from `malus-sema` include source spans and are formatted as:

```
error: dtype mismatch
  --> examples/add_tensors.malus:7:12
   |
 7 |     let c = add(a, b)
   |             ^^^ expected Tensor<f32>, got Tensor<f16>
```

### Initialization order

1. `malus-runtime::init()` — create `MTLDevice` and `MTLCommandQueue`
2. Parse → type-check → GPU codegen (in sequence, all fast)
3. `malus-runtime::load_kernels(registry)` — compile MSL pipelines
4. `malus-codegen-cpu::compile_and_run(typed_program)` — JIT and execute `fn main`

### CTMM free-point wiring

The Cranelift-compiled `fn main` calls into the runtime's C ABI. CTMM's annotations in the typed IR drive the code generator to emit:
- `gpu_barrier()` before freeing any in-flight tensor
- `tensor_free(handle)` at each free-point site

Verify with Metal's validation layer (`MTL_DEBUG_LAYER=1`) that no buffer is accessed after free.

## End-to-end test

```sh
cargo build --release
./target/release/malus examples/add_tensors.malus
```

Expected output:
```
[6.0, 8.0, 10.0, 12.0]
```

Additional checks:
- Exit code 0
- No output to stderr
- No Metal validation errors (`MTL_DEBUG_LAYER=1 ./target/release/malus examples/add_tensors.malus`)
- No memory leaks (run under `leaks --atExit -- ./target/release/malus examples/add_tensors.malus`)

## Out of scope for M6

- `zeros` and `ones` stdlib ops — add as a fast follow after M6 passes; they are single-function additions to the runtime with no new compiler work
- Error recovery (multiple errors in one pass) — deferred to v1
- Rich diagnostic formatting beyond basic span display — deferred to v1
