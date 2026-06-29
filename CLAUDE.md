# malus — Agent Guide

## What this is

malus is a compiled ML DSL for Apple Silicon. Python-like syntax, dual compilation pipeline: `fn` bodies → Cranelift JIT (CPU), `kernel` bodies → Metal Shading Language (GPU). The CTMM memory model inserts static `free`/barrier calls at compile time, falling back to reference counting only when lifetimes are structurally ambiguous.

## Current state: **M20 done — lvalue assignment (`a[i]=e`, `s.f=e`), `mut` params, `**` power operator, AdamW example; M21 (MPS Migration) next**

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
| M11 — The 2-Layer MLP (fixed arrays, 2-D literals, diagnostics, XOR) | ✅ done | all crates |
| **V2 — Autograd** | | |
| M12 — Hardening (enum-payload retain-on-bind, zero-length guard, break/continue) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime` |
| M13 — The `Variable` Type (type-directed RC, dormant retain/release ABI activated) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime` |
| M13.5 — Tuples (anonymous product types, positional access, `let` destructuring, fn return types) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu` |
| M14 — The Tape + `backward()` (global tape, VJPs for all V1 ops, `no_grad`) | ✅ done | `malus-runtime`, `malus-sema`, `malus-codegen-cpu` |
| M15 — Differentiable Stdlib + Capstone (`zero_grad`, V2 XOR capstone) | ✅ done | all crates |
| **V3 — nanoGPT** | | |
| M16 — Broadcasting + Axis Reductions (NumPy broadcast, `sum`/`mean`/`max`/`var` over axis, differentiable) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime` |
| M17 — Shapes + Batched Matmul (`reshape`, `transpose`/`permute`, 3-D/batched matmul) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime` |
| M18 — Transformer Stdlib (`softmax`, `layernorm`, `gelu`, `cross_entropy`, causal mask) | ✅ done | `malus-sema`, `malus-codegen-cpu`, `malus-runtime` |
| M19 — Embeddings + Index Tensors (i32/i64 tensors, `embedding`, `randn`/Philox) | ✅ done | all crates |
| M20 — Lvalue Assignment + AdamW (`a[i]=e`, `s.f=e`, `mut` params, `**` op, AdamW example) | ✅ done | `malus-syntax`, `malus-sema`, `malus-codegen-cpu` |
| M21 — MPS Migration (`matmul` + reductions → MPS, pending tensors) | 🔲 planned | `malus-runtime` |
| M22 — Data I/O + nanoGPT Capstone (`read_file`, char tokenization, transformer) | 🔲 planned | all crates |

Full milestone specs: `docs/milestones/`. V1 plan: `docs/milestones/v1-plan.md`. V2 plan: `docs/milestones/v2-plan.md`. V3 plan: `docs/milestones/v3-plan.md`. Architecture decisions: `docs/adr/`. Domain vocabulary: `CONTEXT.md`.

## V2 Design Decisions

These decisions were made during V2 planning. Do not re-litigate them without user input.

| Decision | Choice | Rationale |
|---|---|---|
| Autograd architecture | Define-by-run runtime tape | Literal micrograd north-star; one VJP per op; activates RC machinery already in the ABI/IR. See ADR-0015. |
| Grad typing | Distinct `Variable` type | CTMM needs a compile-time type-directed signal to choose static `free` vs RC `release`. `Tensor` keeps static Drop everywhere. See ADR-0016. |
| Tape control | Global thread-local tape + scoped `no_grad` | micrograd/PyTorch model; no viral tape threading; fits `OnceLock<MetalContext>` pattern. |
| VJP authorship | Built-in Rust VJPs; `custom_grad` deferred | All ops to nanoGPT have analytic VJPs; nothing on the critical path needs user-defined grads. |
| CTMM/RC scope | Type-directed RC on `Variable` only | General dataflow-liveness RC fallback stays deferred; correctness comes from the type. |
| `.grad` type | Plain `Tensor<f32>` | No double-backward in V2; gradient tensors stay outside the tape and eligible for static Drop. |
| Tape clearing | Auto-clear after `backward` | PyTorch `retain_graph=False` default prevents unbounded tape growth across steps. |
| `Variable` name | `Variable` | Most recognizable autodiff term; capital-V type reads distinctly from lowercase program variables. |

## V3 Design Decisions

These decisions were made during V3 planning. Do not re-litigate them without user input.

| Decision | Choice | Rationale |
|---|---|---|
| V3 capstone | Char GPT on tiny Shakespeare via real file I/O | Faithful nanoGPT demo (train + sample); file I/O accepted (ADR-0018). |
| MPS migration scope | `matmul` + transformer reductions (M21) | Eager-CPU matmul makes the transformer capstone unrunnably slow; amends ADR-0012. See ADR-0017. |
| Optimizer | Lvalue assignment + stdlib AdamW (M20) | Retires V1 language gap; proves language composes into a real optimizer. |
| Index tensor dtype | i32/i64 only | Narrow carve-out for embedding lookup; full f16/bf16 compute generality stays deferred. |
| Broadcasting | NumPy right-aligned (M16) | Retires the `ones41 @ b` bias-broadcast trick. |
| Axis reductions | `keepdim` parameter (M16) | Required for layernorm and softmax normalization. |
| reshape buffer model | Clone `MTLBuffer` handle into independent `TensorBuffer` (M17) | Zero-copy; no CTMM special-casing; safe under immutability. Overrules spec text. See ADR-0023. |
| reshape vs view | `reshape` only; `view` reserved (M17) | Zero-copy M17 reshape *is* view semantics. Shipping both names would advertise a non-existent distinction. See ADR-0022. |
| transpose vs permute | Separate builtins, shared runtime engine (M17) | PyTorch `torch.transpose` swaps two axes; `torch.permute` reorders all. Overloading one name would be a future breaking change. See ADR-0022. |
| Batched matmul scope | Both-3-D identical-batch + 3-D⊗2-D broadcast (M17/M18) | 3-D⊗2-D (`(B,M,K) @ (K,N) → (B,M,N)`) added in M18 to support `x @ weight` where x is [B,T,C]. 2-D⊗3-D broadcast deferred as additive. |
| API parity principle | API surface tracks PyTorch's actual contracts (M17) | New names must match their PyTorch counterpart contract; deferral is additive, never breaking. See ADR-0022. |
| File I/O scope | Minimal `read_file` + 3 string primitives | Capstone needs real data; I/O scope strictly fenced. See ADR-0018. |
| `gather` vs `embedding` | Ship `embedding` only; reserve `gather` (M19) | `torch.gather(input, dim, index)` is a different contract (general axis gather). Naming row-lookup `gather` would violate ADR-0022. |
| randn RNG | Philox4x32-10 + Box-Muller, CPU-side, no user seed (M19) | Counter-based; GPU-portable; reproducible per call index. See ADR-0024. |
| cross_entropy target dtype | Integer (`Tensor<i32|i64>`) (M19) | Retires f32 placeholder; aligns with index tensor carve-out. |

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
  malus-codegen-cpu/   # Cranelift JIT  — V1 complete (src/lib.rs, src/tests.rs)
  malus-codegen-gpu/   # MSL codegen    — V1 complete (src/lib.rs, src/tests.rs)
  malus-runtime/       # Metal API      — V1 complete (src/lib.rs, src/metal.rs, src/tests.rs)
  malus-cli/           # entry point with ariadne diagnostics (src/main.rs)
examples/
  add_tensors.ml       # MVP golden example
  mlp_forward.ml       # M8 done-when: 2-layer forward pass with relu/matmul/sum/transpose
  control_flow.ml      # M9 done-when: for loop + nested if with tensor ops
  structs_enums.ml     # M10 done-when: struct + data-carrying enum + match
  arrays.ml            # M11: Array<T,N>, ForIn, indexing
  nested_tensor.ml     # M11: 2-D tensor literal [[r0],[r1]]
  xor.ml               # M11 done-when: 2→8→1 sigmoid MLP that learns XOR (V1 capstone)
  hardening.ml         # M12 done-when: break/continue, zeros(0), enum-payload escape
  tuples.ml            # M13.5 done-when: tuple construction, positional access, let destructuring, fn return
  gradient_check.ml    # M14 done-when: variable(), backward(), .grad, with no_grad:, autograd gradient check
  import_demo/         # multi-file import demo (main.ml, ops.ml)
  adamw.ml             # M20 done-when: self-contained AdamW optimizer + linear regression training
docs/milestones/       # m1–m12 specs, v1-plan.md, v2-plan.md
docs/spec/             # language spec (01-overview … 09-modules)
docs/adr/              # architecture decision records
```

## The pipeline (M12 complete — M1–M12)

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

## Runtime C ABI (M11 state)

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

**dtype_tag** uses `ScalarTy` enum discriminant order: F32=0, F16=1, Bf16=2, I8=3, I16=4, I32=5, I64=6, U8=7, U16=8, U32=9, U64=10. `malus-runtime` defines an independent `Dtype` enum with `from_tag(i32)`/`to_tag() -> i32`; a drift-detection test asserts all 11 mappings. **M19+: i32 (tag 5) and i64 (tag 6) are supported for index tensors** (`embedding`, `cross_entropy` targets). All other non-f32 dtypes still panic per ADR-0006.

**Device/queue:** lazy `OnceLock<MetalContext { device, command_queue, current_command_buffer, pipelines }>`; first Metal fn call triggers `Device::system_default()` (panics if absent). `runtime_init` must be called before any `kernel_dispatch` to compile MSL kernels.

**gpu_barrier:** if a `current_command_buffer` exists, commits it and waits for completion; otherwise no-op.

**kernel_dispatch:** looks up the `MTLComputePipelineState` by `kernel_id`, allocates an output buffer matching the first input's dtype and shape, encodes a compute pass (`setComputePipelineState`, `setBuffer` per input + output, `dispatchThreads:threadsPerThreadgroup:`), does NOT commit (commit happens in `gpu_barrier`).

## Known Limitations

| Limitation | Status | Notes |
|---|---|---|
| Escaping struct/enum match-arm payload (non-tensor) | M13 | Compile error today (ADR-0019); aggregate boxes have no refcount until M13 adds one |
| Tuple elements as struct fields or array elements | Post-M13.5 | Sema rejects these positions; requires recursive drop in DropStruct/RC path (ADR-0020) |
| `match` on tuples | Post-M13.5 | Destructure with `let (a, b) = x` instead; match arm patterns deferred |
| Nested tuples (`((a, b), c)`) | Post-M13.5 | Flat-only in M13.5; element types may not themselves be tuples (ADR-0020) |
| NumPy-style broadcasting | ✅ M16 | Done |
| Axis reductions (`mean`, `var`, `max` with keepdim) | ✅ M16 | Done |
| `reshape`/`transpose`/`permute`, batched/3-D matmul | ✅ M17 | Done; `view` reserved for strided non-contiguous post-V3 |
| Transformer stdlib (softmax, layernorm, GELU, cross-entropy) | ✅ M18 | Done; `gelu` uses tanh approx; `layernorm` has no affine (additive post-V3) |
| Index tensors, `embedding`, `randn` (Philox4x32-10) | ✅ M19 | Done; `gather` reserved (different PyTorch contract); user seed post-V3 |
| Lvalue assignment (`a[i]=e`, `s.f=e`) | ✅ M20 | Done; `mut` params for interior-only borrows; `**` power op; `Variable` field assign post-V3 (ADR-0016/ADR-0025) |
| matmul/transpose/sum are CPU loops, not MPS | M21 (ADR-0017 amends ADR-0012) | Correct but slow for transformer scale |
| File I/O / data loading | M22 (ADR-0018) | No `read_file` yet |
| Non-f32 compute dtypes (f16, bf16) | Post-V3 | Only i32/i64 for index tensors added in M19 |
| Cross-module structs/enums unsupported (loader `exported_names` gap) | Required post-V3 milestone | See `docs/milestones/cross-module-types.md`; fix needed before M22 import story works |
| ScalarBroadcast IR node | Post-V3 | Inline scalar-broadcast BinOps work; dedicated IR node deferred |
| CTMM barrier coalescing is conservative | Post-V3 | ADR-0009 "Consequences" |
| General dataflow-liveness RC fallback | Post-V3 | Type-directed RC on `Variable` supersedes this for autograd; true dataflow RC remains deferred |

## Coding conventions

- No comments unless the why is non-obvious
- No docstrings
- Rust 2021 edition throughout
- Tests live in `src/tests.rs` (a separate file, not inline) — see `malus-sema` for the pattern
- `cargo test --workspace` must pass before committing
- File extension for malus source files: `.ml`
