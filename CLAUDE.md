# malus — Agent Guide

## What this is

malus is a compiled ML DSL for Apple Silicon. Python-like syntax, dual compilation pipeline: `fn` bodies → Cranelift JIT (CPU), `kernel` bodies → Metal Shading Language (GPU). The CTMM memory model inserts static `free`/barrier calls at compile time; no GC, no RC on the fast path.

## Current state: M2 done, **M3 is next**

| Milestone | Status | Crate |
|---|---|---|
| M1 — Syntax (lexer, parser, AST, loader) | ✅ done | `malus-syntax`, `malus-loader` |
| M2 — Semantics (type checker + CTMM last-use) | ✅ done | `malus-sema` |
| M3 — CPU Codegen (Cranelift JIT for `fn` bodies) | **← start here** | `malus-codegen-cpu` |
| M4 — Metal Runtime | not started | `malus-runtime` |
| M5 — GPU Codegen (MSL for `kernel` bodies) | not started | `malus-codegen-gpu` |
| M6 — Integration (end-to-end CLI) | not started | `malus-cli` |

Full milestone specs: `docs/milestones/`. Architecture decisions: `docs/adr/`. Domain vocabulary: `CONTEXT.md`.

## Codebase map

```
crates/
  malus-syntax/        # lexer, parser, AST  (src/lexer.rs, src/parser.rs, src/ast.rs)
  malus-loader/        # module resolution + flattening  (src/lib.rs)
  malus-sema/          # type checker + CTMM  (src/{check,ctmm,env,builtins,ty,typed_ir,error}.rs)
  malus-codegen-cpu/   # Cranelift JIT  — STUB, start here for M3
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

## The pipeline (M1 + M2 complete)

```
.ml source file
  │
  ▼  malus_loader::ModuleLoader::new().load(path)
LoadedProgram { program: Program, module_aliases, sources }
  │
  ▼  malus_sema::check(&program, &module_aliases)
TypedProgram { fns: Vec<TypedFn>, kernels: Vec<TypedKernel> }
  │
  ▼  (M3) malus_codegen_cpu::compile_and_run(&typed_program)
     execute fn main()
```

`malus-cli/src/main.rs` already calls the first two stages and debug-prints the `TypedProgram`. M3's job is to replace that debug print with actual Cranelift compilation and execution.

## What M2 produces (what M3 consumes)

The typed IR lives in `crates/malus-sema/src/typed_ir.rs`. Key types:

```rust
TypedProgram { fns: Vec<TypedFn>, kernels: Vec<TypedKernel> }

TypedFn {
    name: String,
    params: Vec<TypedParam>,      // TypedParam { name, ty: ResolvedTy }
    return_ty: ResolvedTy,
    body: Vec<TypedStmt>,
    span: Span,
}

// ResolvedTy (crates/malus-sema/src/ty.rs):
//   Tensor { dtype: ScalarTy } | Scalar(ScalarTy) | Bool | Tuple(Vec<ResolvedTy>) | Unit

TypedStmt:
  Let { name: String, expr: TypedExpr }
  Return { expr: TypedExpr }
  Expr(TypedExpr)
  Drop { name: String }       // CTMM: free this tensor binding
  GpuBarrier                  // CTMM: wait for in-flight GPU work

TypedExpr { kind: TypedExprKind, ty: ResolvedTy, placement: Option<Placement>, span }

TypedExprKind:
  Lit(Lit)
  Ident(String)
  BinOp { op: BinOp, lhs: Box<TypedExpr>, rhs: Box<TypedExpr> }
  Unary { op: UnaryOp, operand: Box<TypedExpr> }
  Call { callee: String, args: Vec<TypedExpr> }          // fn or builtin
  KernelCall { callee: String, args: Vec<TypedExpr>,
               in_flight: Vec<String> }                  // GPU dispatch
  TensorLiteral { placement: Placement, dtype: ScalarTy, elements: Vec<TypedExpr> }
  Index { base, indices }
  FieldAccess { base, field }
```

CTMM guarantees that every non-returning tensor binding gets a `Drop` at its last-use point, preceded by a `GpuBarrier` if any binding in that group was passed to a `KernelCall`. M3 codegen pattern-matches these statements directly — no ownership analysis required.

## M3 implementation guide

See `docs/milestones/m3-cpu-codegen.md` for full spec. Key points:

**Entry point to implement:**
```rust
// crates/malus-codegen-cpu/src/lib.rs
pub fn compile_and_run(program: &TypedProgram) -> Result<(), CodegenError>
```

**Tensors are opaque `i64` handles** in Cranelift IR. The runtime owns the actual `MTLBuffer`. Codegen never touches Metal.

**Runtime C ABI** (define in `malus-runtime`, declare as Cranelift externals in `malus-codegen-cpu`):
```c
i64  tensor_alloc_gpu(i32 dtype_tag, i64 len, ptr data)   // data = null-terminated f32 array
i64  kernel_dispatch(ptr name, i64* handles, i32 nhandles) // returns output handle
void gpu_barrier()
void tensor_print(i64 handle)
void tensor_free(i64 handle)
```

**Cranelift deps to add** to `crates/malus-codegen-cpu/Cargo.toml`:
```toml
cranelift-codegen  = "0.113"
cranelift-frontend = "0.113"
cranelift-jit      = "0.113"
cranelift-native   = "0.113"
cranelift-module   = "0.113"
```

**Lowering table:**

| TypedStmt / TypedExprKind | Cranelift lowering |
|---|---|
| `TensorLiteral { placement: Gpu, dtype, elements }` | Emit data as static constant; call `tensor_alloc_gpu` |
| `Call { callee: "print", .. }` | Call `tensor_print(handle)` |
| `KernelCall { callee, args, .. }` | Call `kernel_dispatch(name_ptr, handles_array, len)` |
| `GpuBarrier` | Call `gpu_barrier()` |
| `Drop { name }` | Call `tensor_free(lookup(name))` |
| `BinOp` on scalars | Cranelift `iadd` / `fadd` / etc. |
| `Return` | Cranelift `return` |

**For M3, `kernel_dispatch` and `gpu_barrier` can be no-op stubs** — the goal is to prove the Cranelift pipeline end-to-end with `tensor_alloc_gpu`, `tensor_print`, and `tensor_free` wired to real (or mock) implementations.

## Coding conventions

- No comments unless the why is non-obvious
- No docstrings
- Rust 2021 edition throughout
- Tests live in `src/tests.rs` (a separate file, not inline) — see `malus-sema` for the pattern
- `cargo test --workspace` must pass before committing
- File extension for malus source files: `.ml`
