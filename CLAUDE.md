# malus — Agent Guide

## What this is

malus is a compiled ML DSL for Apple Silicon. Python-like syntax, dual compilation pipeline: `fn` bodies → Cranelift JIT (CPU), `kernel` bodies → Metal Shading Language (GPU). The CTMM memory model inserts static `free`/barrier calls at compile time; no GC, no RC on the fast path.

## Current state: M5 done, **M5.1 is next**

| Milestone | Status | Crate |
|---|---|---|
| M1 — Syntax (lexer, parser, AST, loader) | ✅ done | `malus-syntax`, `malus-loader` |
| M2 — Semantics (type checker + CTMM last-use) | ✅ done | `malus-sema` |
| M3 — CPU Codegen (Cranelift JIT for `fn` bodies) | ✅ done | `malus-codegen-cpu` |
| M4 — Metal Runtime | ✅ done | `malus-runtime` |
| M5 — GPU Codegen (MSL for `kernel` bodies) | ✅ done | `malus-codegen-gpu` |
| M5.1 — Built-in element-wise kernels for fn-body BinOp | **← next** | `malus-codegen-gpu`, `malus-codegen-cpu` |
| M6 — Integration (end-to-end CLI) | not started | `malus-cli` |

Full milestone specs: `docs/milestones/`. Architecture decisions: `docs/adr/`. Domain vocabulary: `CONTEXT.md`.

## Codebase map

```
crates/
  malus-syntax/        # lexer, parser, AST  (src/lexer.rs, src/parser.rs, src/ast.rs)
  malus-loader/        # module resolution + flattening  (src/lib.rs)
  malus-sema/          # type checker + CTMM  (src/{check,ctmm,env,builtins,ty,typed_ir,error}.rs)
  malus-codegen-cpu/   # Cranelift JIT  — M3 complete (src/lib.rs, src/tests.rs)
  malus-codegen-gpu/   # MSL codegen    — M5 complete (src/lib.rs, src/tests.rs)
  malus-runtime/       # Metal API      — M5 complete (src/lib.rs, src/metal.rs, src/tests.rs)
  malus-cli/           # entry point    (src/main.rs)
examples/
  add_tensors.ml       # MVP golden example
  import_demo/         # multi-file import demo (main.ml, ops.ml)
docs/milestones/       # m1–m7 specs, m5.1 spec
docs/spec/             # language spec (01-overview … 09-modules)
docs/adr/              # architecture decision records
```

## The pipeline (M1 + M2 + M3 + M4 + M5 complete)

```
.ml source file
  │
  ▼  malus_loader::ModuleLoader::new().load(path)
LoadedProgram { program: Program, module_aliases, sources }
  │
  ▼  malus_sema::check(&program, &module_aliases)
TypedProgram { fns: Vec<TypedFn>, kernels: Vec<TypedKernel> }
  │
  ▼  malus_codegen_gpu::compile_kernels(&typed_program)
(KernelRegistry, HashMap<String, u64>)  — MSL source per kernel_id, name→id map
  │
  ▼  malus_runtime::runtime_init(&registry.into_hashmap())   [macOS only]
Compiles all MSL to MTLComputePipelineState, cached by kernel_id
  │
  ▼  malus_codegen_cpu::compile_and_run(&typed_program, &runtime_symbols, &kernel_ids)
     execute fn main()  →  kernel_dispatch(kernel_id, handles, count)  →  real GPU work
```

`malus-cli/src/main.rs` runs all stages. `compile_and_run` is fully implemented; `fn main()` is JIT-compiled and executed via Cranelift. The `RuntimeSymbols` struct of five `extern "C" fn` pointers is injected by the CLI (real Metal fns from `malus-runtime` on macOS); tests inject mock fns. The `kernel_ids` map (`&HashMap<String, u64>`) is produced by `compile_kernels` and passed to `compile_and_run` so the JIT'd code can bake `u64` kernel ids at `KernelCall` sites.

## Runtime C ABI (M5)

The five runtime functions are real Metal implementations in `malus-runtime/src/metal.rs`, injected into the JIT via a `RuntimeSymbols` struct. `compile_and_run` accepts `&RuntimeSymbols` and `&HashMap<String, u64>` (kernel name → id map); the CLI constructs it from `malus-runtime`'s exported fns on macOS, tests construct mock fns. codegen-cpu stays platform-agnostic and Metal-unaware (ADR-0008).

```c
i64  tensor_alloc_gpu(i32 dtype_tag, i64 len, const float* data)
i64  kernel_dispatch(u64 kernel_id, const i64* handles, usize count)
void gpu_barrier()
void tensor_print(i64 handle)
void tensor_free(i64 handle)
```

The `i64` handle is a raw pointer to a heap-allocated `TensorBuffer { buffer: metal::Buffer, dtype: Dtype, len: usize }` wrapping a real `MTLBuffer` (`StorageModeShared`). The runtime owns it; `tensor_free` drops the box.

**dtype_tag** uses `ScalarTy` enum discriminant order: F32=0, F16=1, Bf16=2, I8=3, I16=4, I32=5, I64=6, U8=7, U16=8, U32=9, U64=10. `malus-runtime` defines an independent `Dtype` enum with `from_tag(i32)`/`to_tag() -> i32`; a drift-detection test asserts all 11 mappings. **M5 supports f32 only** — non-f32 panics per ADR-0006.

**Device/queue:** lazy `OnceLock<MetalContext { device, command_queue, current_command_buffer, pipelines }>`; first Metal fn call triggers `Device::system_default()` (panics if absent). `runtime_init` must be called before any `kernel_dispatch` to compile MSL kernels.

**gpu_barrier:** if a `current_command_buffer` exists, commits it and waits for completion; otherwise no-op.

**kernel_dispatch:** looks up the `MTLComputePipelineState` by `kernel_id`, allocates an output buffer matching the first input's dtype/len, encodes a compute pass (`setComputePipelineState`, `setBuffer` per input + output, `dispatchThreads:threadsPerThreadgroup:`), does NOT commit (commit happens in `gpu_barrier`).

**Known M5 limitations / deferred work:**
- `BinOp` on tensor types in host `fn` bodies returns `UnsupportedExpr` — deferred to M5.1; see `docs/adr/0007-tensor-binop-in-fn-bodies.md` and `docs/milestones/m5.1-builtin-elementwise-kernels.md`.
- `zeros` / `ones` builtins return `UnsupportedExpr` — not needed for the golden example.
- Non-f32 dtypes panic — the `Dtype` enum exists but only `F32` is functional.
- Zero-length tensors crash Metal's `new_buffer(0, ...)` — not needed for the golden example.
- CTMM's GPU-pending barrier logic (ADR-0009) coalesces barriers conservatively; drops between chained kernel calls may cause two command buffers instead of one.

## Coding conventions

- No comments unless the why is non-obvious
- No docstrings
- Rust 2021 edition throughout
- Tests live in `src/tests.rs` (a separate file, not inline) — see `malus-sema` for the pattern
- `cargo test --workspace` must pass before committing
- File extension for malus source files: `.ml`
