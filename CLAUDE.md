# malus — Agent Guide

## What this is

malus is a compiled ML DSL for Apple Silicon. Python-like syntax, dual compilation pipeline: `fn` bodies → Cranelift JIT (CPU), `kernel` bodies → Metal Shading Language (GPU). The CTMM memory model inserts static `free`/barrier calls at compile time, falling back to reference counting only when lifetimes are structurally ambiguous.

## Current state: M10 done, **M11 is next**

| Milestone | Status | Crate |
|---|---|---|
| M1 — Syntax (lexer, parser, AST, loader) | ✅ done | `malus-syntax`, `malus-loader` |
| M2 — Semantics (type checker + CTMM last-use) | ✅ done | `malus-sema` |
| M3 — CPU Codegen (Cranelift JIT for `fn` bodies) | ✅ done | `malus-codegen-cpu` |
| M4 — Metal Runtime | ✅ done | `malus-runtime` |
| M5 — GPU Codegen (MSL for `kernel` bodies) | ✅ done | `malus-codegen-gpu` |
| M5.1 — Built-in element-wise kernels for fn-body BinOp | ✅ done | `malus-codegen-gpu`, `malus-codegen-cpu` |
| M6 — Integration (end-to-end CLI) | ✅ done | `malus-cli` |
| M7 — Kernel Thickening (multi-stmt kernels, `let mut`, scalar broadcasting) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-*` |
| M8 — Core Stdlib (matmul, relu/sigmoid/tanh, transpose, zeros/ones, sum) | ✅ done | `malus-runtime`, `malus-codegen-*` |
| M9 — Control Flow (if/else, for, while, hierarchical CTMM) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime` |
| M10 — Structs + Enums (structs, data-carrying enums, match) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu` |
| **M11 — The 2-Layer MLP** (fixed arrays, diagnostics, done-when) | **← next** | all crates |

Full milestone specs: `docs/milestones/`. V1 plan overview: `docs/milestones/v1-plan.md`. Architecture decisions: `docs/adr/`. Domain vocabulary: `CONTEXT.md`.

## V1 Design Decisions

These decisions were made during V1 planning. Do not re-litigate them without user input.

| Decision | Choice | Rationale |
|---|---|---|
| CTMM for conditional paths | Hierarchical Drop (M9); RC fallback deferred | ADR-0014 supersedes ADR-0002: hierarchical analysis places Drop after the control-flow node in the outer scope, which is always correct regardless of branch taken / iteration count. RC nodes added to ABI + IR for M10 readiness but M9 emits zero of them. Dataflow liveness is a V2 optimization. |
| Mutation | `let mut` + reassignment | Shadowing breaks in loops (shadow dies at loop-body scope end). `let mut` CTMM = drop-old + bind-new. No aliasing risk under move semantics. |
| Kernel body expressiveness | Let bindings, comparisons, ternary | Enough for all gradient kernels. Loops inside kernels need threadgroup controls — deferred post-V1. |
| Enum scope | Data-carrying variants + match, no generics | No `Option<T>` in V1. |
| Arrays | Fixed-length with iteration | Growable `Vec<T>` deferred. CTMM can reason statically about fixed arrays. |
| Stdlib scope | Core math (~12 functions) | Near-zero marginal cost once M5.1 infrastructure exists. |
| V1 done-when | Manual forward+backward 2-layer MLP | North Star: "could someone build micrograd on this?" V1 proves expressiveness; tape comes later. |
| matmul/transpose/sum execution | Eager CPU loops (M8) | V1 proves expressiveness not throughput; MPS migration deferred post-V1. See ADR-0012. |
| Tensor shape checking | Runtime-only panics (M8) | Static shape inference is cross-cutting; zeros/ones accept runtime dim args defeating static shapes at entry. See ADR-0013. |

## Codebase map

```
crates/
  malus-syntax/        # lexer, parser, AST  (src/lexer.rs, src/parser.rs, src/ast.rs)
  malus-loader/        # module resolution + flattening  (src/lib.rs)
  malus-sema/          # type checker + CTMM  (src/{check,ctmm,env,builtins,ty,typed_ir,error}.rs)
  malus-codegen-cpu/   # Cranelift JIT  — M10 complete (src/lib.rs, src/tests.rs)
  malus-codegen-gpu/   # MSL codegen    — M10 complete (src/lib.rs, src/tests.rs)
  malus-runtime/       # Metal API      — M10 complete (src/lib.rs, src/metal.rs, src/tests.rs)
  malus-cli/           # entry point    (src/main.rs)
examples/
  add_tensors.ml       # MVP golden example
  mlp_forward.ml       # M8 done-when: 2-layer forward pass with relu/matmul/sum/transpose
  control_flow.ml      # M9 done-when: for loop + nested if with tensor ops
  structs_enums.ml     # M10 done-when: struct + data-carrying enum + match
  import_demo/         # multi-file import demo (main.ml, ops.ml)
docs/milestones/       # m1–m11 specs, v1-plan.md
docs/spec/             # language spec (01-overview … 09-modules)
docs/adr/              # architecture decision records
```

## The pipeline (M1–M10 complete)

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

`malus-cli/src/main.rs` runs all stages. `compile_and_run` is fully implemented; `fn main()` is JIT-compiled and executed via Cranelift. The `RuntimeSymbols` struct of thirteen `extern "C" fn` pointers is injected by the CLI (real Metal fns from `malus-runtime` on macOS); tests inject mock fns. The `kernel_ids` map (`&HashMap<String, u64>`) is produced by `compile_kernels` and passed to `compile_and_run` so the JIT'd code can bake `u64` kernel ids at `KernelCall` and unary-builtin-dispatch sites.

## Runtime C ABI (M10 state)

The thirteen runtime functions are real Metal implementations in `malus-runtime/src/metal.rs`, injected into the JIT via a `RuntimeSymbols` struct. codegen-cpu stays platform-agnostic and Metal-unaware (ADR-0008). Structs and enums are heap-allocated via libc `malloc`/`free` registered directly as JIT symbols (not in `RuntimeSymbols`).

```c
i64  tensor_alloc_gpu(i32 dtype_tag, const usize* shape_ptr, usize ndims, const float* data)
i64  tensor_alloc_zeros_gpu(const usize* shape_ptr, usize ndims)
i64  tensor_alloc_ones_gpu(const usize* shape_ptr, usize ndims)
i64  tensor_len(i64 handle)
i64  tensor_matmul(i64 handle_a, i64 handle_b)
i64  tensor_transpose(i64 handle)
i64  tensor_sum(i64 handle)
i64  kernel_dispatch(u64 kernel_id, const i64* handles, usize count)
void gpu_barrier()
void tensor_print(i64 handle)
void tensor_free(i64 handle)
// RC ABI — tensor_retain/release used by DropStruct to release tensor fields.
void tensor_retain(i64 handle)
void tensor_release(i64 handle)
```

The `i64` handle is a raw pointer to a heap-allocated `TensorBuffer { buffer: metal::Buffer, dtype: Dtype, len: usize, shape: Vec<usize>, ref_count: AtomicUsize }` wrapping a real `MTLBuffer` (`StorageModeShared`). The runtime owns it; `tensor_free` delegates to `tensor_release` which drops the box when the refcount hits zero. Invariant: `len == shape.iter().product()`.

`tensor_matmul`, `tensor_transpose`, and `tensor_sum` are eager CPU ops that call `gpu_barrier()` internally before reading buffers. Their results are ready tensors (not pending). See ADR-0012.

**dtype_tag** uses `ScalarTy` enum discriminant order: F32=0, F16=1, Bf16=2, I8=3, I16=4, I32=5, I64=6, U8=7, U16=8, U32=9, U64=10. `malus-runtime` defines an independent `Dtype` enum with `from_tag(i32)`/`to_tag() -> i32`; a drift-detection test asserts all 11 mappings. **V1 supports f32 only** — non-f32 panics per ADR-0006.

**Device/queue:** lazy `OnceLock<MetalContext { device, command_queue, current_command_buffer, pipelines }>`; first Metal fn call triggers `Device::system_default()` (panics if absent). `runtime_init` must be called before any `kernel_dispatch` to compile MSL kernels.

**gpu_barrier:** if a `current_command_buffer` exists, commits it and waits for completion; otherwise no-op.

**kernel_dispatch:** looks up the `MTLComputePipelineState` by `kernel_id`, allocates an output buffer matching the first input's dtype and shape, encodes a compute pass (`setComputePipelineState`, `setBuffer` per input + output, `dispatchThreads:threadsPerThreadgroup:`), does NOT commit (commit happens in `gpu_barrier`).

## Known Limitations and Which Milestone Fixes Them

| Limitation | Fix in |
|---|---|
| Intermediate temporaries from nested BinOps leak | M11 |
| No fixed-length arrays | M11 |
| Plain-string error messages (no spans or source context) | M11 |
| Enum-payload tensor RC deferred (tag-only enum payloads don't leak; f32 payload leaks its 16-byte malloc) | M11 |
| Cross-module structs/enums unsupported (loader `exported_names` gap) | M11 |
| `break`/`continue` statements | Post-V1 |
| Early `return` inside a control-flow body | Post-V1 |
| matmul/transpose/sum are CPU loops, not MPS | Post-V1 (ADR-0012) |
| Non-f32 dtypes panic | Post-V1 |
| Zero-length tensors crash Metal | Post-V1 |
| CTMM barrier coalescing is conservative | Post-V1 |

## Coding conventions

- No comments unless the why is non-obvious
- No docstrings
- Rust 2021 edition throughout
- Tests live in `src/tests.rs` (a separate file, not inline) — see `malus-sema` for the pattern
- `cargo test --workspace` must pass before committing
- File extension for malus source files: `.ml`
