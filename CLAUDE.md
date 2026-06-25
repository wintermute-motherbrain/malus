# malus — Agent Guide

## What this is

malus is a compiled ML DSL for Apple Silicon. Python-like syntax, dual compilation pipeline: `fn` bodies → Cranelift JIT (CPU), `kernel` bodies → Metal Shading Language (GPU). The CTMM memory model inserts static `free`/barrier calls at compile time; no GC, no RC on the fast path.

## Current state: M4 done, **M5 is next**

| Milestone | Status | Crate |
|---|---|---|
| M1 — Syntax (lexer, parser, AST, loader) | ✅ done | `malus-syntax`, `malus-loader` |
| M2 — Semantics (type checker + CTMM last-use) | ✅ done | `malus-sema` |
| M3 — CPU Codegen (Cranelift JIT for `fn` bodies) | ✅ done | `malus-codegen-cpu` |
| M4 — Metal Runtime | ✅ done | `malus-runtime` |
| M5 — GPU Codegen (MSL for `kernel` bodies) | **← next** | `malus-codegen-gpu` |
| M6 — Integration (end-to-end CLI) | not started | `malus-cli` |

Full milestone specs: `docs/milestones/`. Architecture decisions: `docs/adr/`. Domain vocabulary: `CONTEXT.md`.

## Codebase map

```
crates/
  malus-syntax/        # lexer, parser, AST  (src/lexer.rs, src/parser.rs, src/ast.rs)
  malus-loader/        # module resolution + flattening  (src/lib.rs)
  malus-sema/          # type checker + CTMM  (src/{check,ctmm,env,builtins,ty,typed_ir,error}.rs)
  malus-codegen-cpu/   # Cranelift JIT  — M3 complete (src/lib.rs, src/tests.rs)
  malus-codegen-gpu/   # MSL codegen    — STUB
  malus-runtime/       # Metal API      — M4 complete
  malus-cli/           # entry point    (src/main.rs)
examples/
  add_tensors.ml       # MVP golden example
  import_demo/         # multi-file import demo (main.ml, ops.ml)
docs/milestones/       # m1–m7 specs
docs/spec/             # language spec (01-overview … 09-modules)
docs/adr/              # architecture decision records
```

## The pipeline (M1 + M2 + M3 + M4 complete)

```
.ml source file
  │
  ▼  malus_loader::ModuleLoader::new().load(path)
LoadedProgram { program: Program, module_aliases, sources }
  │
  ▼  malus_sema::check(&program, &module_aliases)
TypedProgram { fns: Vec<TypedFn>, kernels: Vec<TypedKernel> }
  │
  ▼  malus_codegen_cpu::compile_and_run(&typed_program, &runtime_symbols)
     execute fn main()
```

`malus-cli/src/main.rs` runs all three stages. `compile_and_run` is fully implemented; `fn main()` is JIT-compiled and executed via Cranelift. The `RuntimeSymbols` struct of five `extern "C" fn` pointers is injected by the CLI (real Metal fns from `malus-runtime` on macOS); tests inject mock fns.

## What M4 built (what M5 extends)

M4 is fully implemented. The five runtime functions are real Metal implementations in `malus-runtime/src/metal.rs`, injected into the JIT via a `RuntimeSymbols` struct. `compile_and_run` accepts `&RuntimeSymbols`; the CLI constructs it from `malus-runtime`'s exported fns on macOS, tests construct mock fns. codegen-cpu stays platform-agnostic and Metal-unaware (ADR-0008).

**Runtime C ABI** — preserved from M3; M5 migrates `kernel_dispatch` to `kernel_id: u64` / `usize` when the `KernelRegistry` is introduced:
```c
i64  tensor_alloc_gpu(i32 dtype_tag, i64 len, const float* data)
i64  kernel_dispatch(const char* name, const i64* handles, i32 nhandles)
void gpu_barrier()
void tensor_print(i64 handle)
void tensor_free(i64 handle)
```

The `i64` handle is a raw pointer to a heap-allocated `TensorBuffer { buffer: metal::Buffer, dtype: Dtype, len: usize }` wrapping a real `MTLBuffer` (`StorageModeShared`). The runtime owns it; `tensor_free` drops the box.

**dtype_tag** uses `ScalarTy` enum discriminant order: F32=0, F16=1, Bf16=2, I8=3, I16=4, I32=5, I64=6, U8=7, U16=8, U32=9, U64=10. `malus-runtime` defines an independent `Dtype` enum with `from_tag(i32)`; a drift-detection test asserts all 11 mappings. **M4 supports f32 only** — non-f32 panics per ADR-0006.

**Device/queue:** lazy `OnceLock<MetalContext { device, command_queue }>`; first Metal fn call triggers `Device::system_default()` (panics if absent). No explicit init API.

**gpu_barrier:** creates+commits+waits an empty command buffer. No persistent command buffer state — M5 adds a `current_command_buffer` when `kernel_dispatch` encodes real compute passes.

**Known M4 limitations / deferred work:**
- `BinOp` on tensor types in host `fn` bodies returns `UnsupportedExpr` — the semantics (CPU compute vs. implicit MPS dispatch) are unresolved; see `docs/adr/0007-tensor-binop-in-fn-bodies.md`. M5 should resolve this by lowering to `kernel_dispatch` calls to built-in element-wise kernels.
- `kernel_dispatch` returns a zeroed output buffer matching the first input's dtype/len — real GPU execution is M5. The stub is **replaced wholesale**, not extended.
- `zeros` / `ones` builtins return `UnsupportedExpr` — not needed for the golden example.
- Non-f32 dtypes panic — the `Dtype` enum exists but only `F32` is functional.
- Zero-length tensors crash Metal's `new_buffer(0, ...)` — not needed for the golden example.

## M5 implementation guide

See `docs/milestones/m5-gpu-codegen.md` for full spec. Key points:

**Goal:** Compile malus `kernel` bodies to MSL, register them in a `KernelRegistry` (`kernel_id: u64` → `msl_source: String`), and replace the `kernel_dispatch` stub with real compute dispatch. `malus examples/add_tensors.ml` should print `[6, 8, 10, 12]`.

**Two crates involved:**
- `malus-codegen-gpu` — walks `TypedProgram`'s `kernel` items, emits MSL source as a `String`, produces a `KernelRegistry`. Currently a STUB.
- `malus-runtime` — compiles each MSL entry to a `MTLComputePipelineState` at startup (cached by `kernel_id`), and implements real `kernel_dispatch`.

**ABI migration (M5):** `kernel_dispatch` changes from `(name: *const u8, handles: *const i64, n: i32)` to `(kernel_id: u64, handles: *const i64, count: usize)`. This requires updating the `RuntimeSymbols` struct in `malus-codegen-cpu`, the `KernelCall` IR emission, and the CLI wiring. The `KernelRegistry` makes the `u64` id meaningful.

**`kernel_dispatch` (M5 real impl):** allocate output buffer via `tensor_alloc_gpu`, encode a compute pass (`setComputePipelineState`, `setBuffer` per input + output, `dispatchThreads:threadsPerThreadgroup:`), do NOT commit (commit happens in `gpu_barrier`). Introduce a persistent `current_command_buffer` in `MetalContext`.

**Element-wise detection rule:** a binary op on two tensors of the same shape with no explicit thread indexing lowers as element-wise. Output buffer is the same size as inputs. Thread ID is implicit (`thread_position_in_grid`).

## Coding conventions

- No comments unless the why is non-obvious
- No docstrings
- Rust 2021 edition throughout
- Tests live in `src/tests.rs` (a separate file, not inline) — see `malus-sema` for the pattern
- `cargo test --workspace` must pass before committing
- File extension for malus source files: `.ml`
