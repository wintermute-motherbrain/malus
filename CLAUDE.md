# malus — Agent Guide

## What this is

malus is a compiled ML DSL for Apple Silicon. Python-like syntax, dual compilation pipeline: `fn` bodies → Cranelift JIT (CPU), `kernel` bodies → Metal Shading Language (GPU). The CTMM memory model inserts static `free`/barrier calls at compile time; no GC, no RC on the fast path.

## Current state: M3 done, **M4 is next**

| Milestone | Status | Crate |
|---|---|---|
| M1 — Syntax (lexer, parser, AST, loader) | ✅ done | `malus-syntax`, `malus-loader` |
| M2 — Semantics (type checker + CTMM last-use) | ✅ done | `malus-sema` |
| M3 — CPU Codegen (Cranelift JIT for `fn` bodies) | ✅ done | `malus-codegen-cpu` |
| M4 — Metal Runtime | **← start here** | `malus-runtime` |
| M5 — GPU Codegen (MSL for `kernel` bodies) | not started | `malus-codegen-gpu` |
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
  malus-runtime/       # Metal API      — STUB
  malus-cli/           # entry point    (src/main.rs)
examples/
  add_tensors.ml       # MVP golden example
  import_demo/         # multi-file import demo (main.ml, ops.ml)
docs/milestones/       # m1–m7 specs
docs/spec/             # language spec (01-overview … 09-modules)
docs/adr/              # architecture decision records
```

## The pipeline (M1 + M2 + M3 complete)

```
.ml source file
  │
  ▼  malus_loader::ModuleLoader::new().load(path)
LoadedProgram { program: Program, module_aliases, sources }
  │
  ▼  malus_sema::check(&program, &module_aliases)
TypedProgram { fns: Vec<TypedFn>, kernels: Vec<TypedKernel> }
  │
  ▼  malus_codegen_cpu::compile_and_run(&typed_program)
     execute fn main()
```

`malus-cli/src/main.rs` runs all three stages. `compile_and_run` is fully implemented; `fn main()` is JIT-compiled and executed via Cranelift.

## What M3 built (what M4 replaces)

M3 is fully implemented. The Cranelift JIT pipeline compiles and runs `fn` bodies. All five runtime functions are **currently stubbed in `crates/malus-codegen-cpu/src/lib.rs`** behind a `HashMap<i64, Vec<f32>>` backed by a global `Mutex<TensorStore>`. M4's job is to replace those stubs with real Metal implementations in `malus-runtime`.

**Runtime C ABI** — currently stubbed in `malus-codegen-cpu`, moving to `malus-runtime` in M4:
```c
i64  tensor_alloc_gpu(i32 dtype_tag, i64 len, const float* data)
i64  kernel_dispatch(const char* name, const i64* handles, i32 nhandles)
void gpu_barrier()
void tensor_print(i64 handle)
void tensor_free(i64 handle)
```

The `i64` handle is an opaque token. In M3 it is a HashMap key (incrementing integer). In M4 it becomes a raw pointer to a heap-allocated `TensorBuffer` wrapping a real `MTLBuffer`.

**dtype_tag** uses `ScalarTy` enum discriminant order: F32=0, F16=1, Bf16=2, I8=3, I16=4, I32=5, I64=6, U8=7, U16=8, U32=9, U64=10.

**Known M3 limitations / deferred work:**
- `BinOp` on tensor types in host `fn` bodies returns `UnsupportedExpr` — the semantics (CPU compute vs. implicit MPS dispatch) are unresolved; see `docs/adr/0007-tensor-binop-in-fn-bodies.md`
- `kernel_dispatch` returns an empty dummy tensor — real GPU execution is M5
- `zeros` / `ones` builtins return `UnsupportedExpr` — not needed for the golden example

## M4 implementation guide

See `docs/milestones/m4-metal-runtime.md` for full spec. Key points:

**Goal:** Replace the five `extern "C"` stub functions in `malus-codegen-cpu/src/lib.rs` with real Metal implementations in `malus-runtime/src/lib.rs`. Wire `malus-codegen-cpu` to link against `malus-runtime` instead of defining the stubs itself.

**The JIT already finds these symbols by name** via `JITBuilder::symbol()` in `compile_and_run`. Swapping the implementation is purely a matter of pointing those symbol registrations at the new `malus-runtime` function pointers.

**Metal dep to add** to `crates/malus-runtime/Cargo.toml`:
```toml
metal = "0.29"
```
Gate the entire crate: `#[cfg(target_os = "macos")]`.

## Coding conventions

- No comments unless the why is non-obvious
- No docstrings
- Rust 2021 edition throughout
- Tests live in `src/tests.rs` (a separate file, not inline) — see `malus-sema` for the pattern
- `cargo test --workspace` must pass before committing
- File extension for malus source files: `.ml`
