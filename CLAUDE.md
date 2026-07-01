# malus — Agent Guide

## What this is

malus is a compiled ML DSL for Apple Silicon. Python-like syntax, dual compilation pipeline: `fn` bodies → Cranelift JIT (CPU), `kernel` bodies → Metal Shading Language (GPU). North star: train models like PyTorch without the Python slowness — the V5 gate is ≤2x f32 PyTorch-MPS at the real Karpathy nanoGPT config. The CTMM memory model uses escape analysis + Lobster-style borrow-inference to insert static `free` calls at compile time, falling back to reference counting only where ownership is genuinely structurally ambiguous (a `List<T>` that may alias across a call boundary, or a struct field with no provable single owner) — never for the autograd tape, which retains its own copy of anything it saves (M29, ADR-0026). There is one tensor type: `Tensor<dtype>`.

## Current state: **V5 in progress — M30 done (2026-07-01), M31 next**

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
| M21 — MPS Migration (objc2-metal port + `matmul` → MPS, eager) | ✅ done | `malus-runtime` |
| M22 — Data I/O + nanoGPT Capstone (`read_file`, char tokenization, transformer) | ✅ done | all crates |
| **V4 — Reclaiming the Vision** (roadmap approved 2026-06-29; see `docs/adr/0026–0031`, plan file) | | |
| M23 — De-risk spike (extended `kernel_dispatch` ABI + CPU-compute counter CI gate) | ✅ done | `malus-runtime` |
| M24 — Kernel language v2 (thread hierarchy, flat indexing, `let shared`, `barrier()`, control flow, scalar uniforms) | ✅ done | `malus-codegen-gpu`, `malus-syntax`, `malus-sema`, `malus-runtime` |
| M25 — Stdlib forward kernels (all CPU-loop ops → malus `.ml` kernels; forward-hot-path CPU-counter==0) | ✅ done | all crates |
| M26 — Backward kernels (GPU autograd; full-step CPU-counter==0 canonical gate) | ✅ done | `malus-runtime`, `malus-codegen-cpu`, `malus-stdlib` |
| M27 — Kill `Variable` (static grad-inference; one `Tensor` type) | ✅ done | `malus-sema`, `malus-codegen-cpu`, `malus-codegen-gpu`, `malus-syntax` |
| M28 — Module trait + generic optimizer (generics, `impl`, `List<T>`; no-unroll lint gate) | ✅ done | all crates |
| M29 — Borrow-inference RC + benchmark (Lobster single-owner/borrow pass; ≤5% compile-time RC-reduction-ratio gate; both sides measured 2026-07-01: coarse whole-process ratio ≈ 60x vs f32 PyTorch-MPS at the toy config — the founding motivation for V5; M30's matched-methodology warm median corrected the steady-state gap to **26.187 ms/step ≈ 9.6x**) | ✅ done | `malus-sema`, `malus-runtime` |
| **V5 — Earning the Claim** (roadmap approved 2026-07-01; see `docs/milestones/v5-plan.md`, ADRs 0035–0037) | | |
| M30 — Honest timing baseline (`--bench` warm per-step median timer via dormant `bench_step_begin`/`bench_step_end` builtins; measured **26.187 ms/step ≈ 9.6x** vs f32 PyTorch-MPS at the toy config — the 60x coarse figure was ~5/6ths one-time startup; docs hygiene; ADR-0038) | ✅ done | `malus-runtime`, `malus-cli` |
| M31 — Async dispatch substrate (MPS matmul joins shared command buffer; per-buffer pending flags; auto-flush on host read; `__flush()` deleted) | planned | `malus-runtime`, `malus-sema` |
| M32 — Buffer pooling + memory budget (size-class MTLBuffer free-list) | planned | `malus-runtime` |
| M33 — N-D permute backward + multi-head attention (rank-generic permute VJP; head-folding) | planned | `malus-runtime`, `malus-stdlib` |
| M34 — Named submodules (`List<Struct>` recursive drop; optimizer recursion; ADR-0036) | planned | `malus-sema`, `malus-codegen-cpu`, `malus-cli` |
| M35 — Capstone + benchmark gate (Karpathy config 6L/6H/384d/T=256/B=64; **≤2x PyTorch-MPS f32 hard gate**; README rewrite) | planned | all crates |
| M36 — Mixed precision, bf16-first (autocast-style; post-gate; ADR-0037) | planned | `malus-codegen-gpu`, `malus-runtime`, `malus-sema`, `malus-stdlib` |

Full milestone specs: `docs/milestones/`. V1 plan: `docs/milestones/v1-plan.md`. V2 plan: `docs/milestones/v2-plan.md`. V3 plan: `docs/milestones/v3-plan.md`. V4 plan: `docs/milestones/v4-plan.md` + individual specs `m23` through `m29`. V5 plan: `docs/milestones/v5-plan.md` + individual specs `m30` through `m36`. Architecture decisions: `docs/adr/` (V4 ADRs: 0026–0034; V5 ADRs: 0035–0037). Domain vocabulary: `CONTEXT.md`.

## V5 Design Decisions

These decisions were made during V5 planning (2026-07-01). Do not re-litigate them without user input.

| Decision | Choice | Rationale |
|---|---|---|
| V5 north star | Performance-first: earn "without the Python slowness" | The vision's one falsifiable claim measured false at ~60x coarse (M30 matched: ~9.6x steady-state — still ~5x from the gate); causes are dispatch-architectural, untouchable by language/tooling work. |
| Capstone scale | Karpathy char-Shakespeare config (6L/6H/384d/T=256/B=64) | Toy-config Nx measures only dispatch overhead; the claim must hold at a config PyTorch users recognize. Toy kept as regression benchmark. |
| Perf bar | ≤2x f32 PyTorch-MPS, **hard gate**; parity stretch | V4's soft bar went unmeasured until after closing. Matmul dominates at 384d and both sides use MPS matmul, so ≤2x is achievable without fusion. |
| Execution model | Async runtime substrate (V5) → compile-time graph (V6); lazy runtime capture rejected | malus is compiled — the typed IR already IS the graph; lazy capture is the dynamic-language workaround and would be thrown away. See ADR-0035. |
| Read safety | Runtime per-buffer pending tracking + auto-flush | Fixes ADR-0032 barrier-before-read as a guarantee, not per-call-site `__flush()`. Static barriers demote to optimization (CTMM shape: static fast path, dynamic fallback). |
| Multi-head | Head-folding via rank-generic permute VJP + existing 3-D matmul | Whole forward path already works; only permute backward (hardcoded rank ≤3) blocks it. No 4-D matmul needed. |
| Submodules | Named, via optimizer recursion; `parameters()` concat rejected | Concat returns a snapshot — optimizer would update it while weights freeze silently (ADR-0034 hazard). See ADR-0036. |
| Mixed precision | bf16-first autocast, in V5 (M36) but strictly post-gate | Honors v4-plan promise; bf16 needs no loss scaling; gate stays f32-vs-f32 so it measures architecture, not precision. See ADR-0037. |
| Persistence | Save/load/SafeTensors deferred to V6 | Nothing in V5's done-when needs it; deserves its own designed milestone. |
| Tooling | Docs hygiene only (README rewrite at M35) | None of it moves the perf claim; the tooling arc is V6+. |

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
  malus-sema/          # type checker + CTMM  (src/{check,ctmm,borrow_inference,grad_inference,env,builtins,ty,typed_ir,error}.rs)
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
| Lvalue assignment (`a[i]=e`, `s.f=e`) | ✅ M20 | Done; `mut` params for interior-only borrows; `**` power op |
| `transpose`/`sum`/axis reductions are CPU loops | ✅ M25/M26 | All replaced by malus `.ml` kernels (ADR-0027/0028); CPU-counter CI gate enforces this |
| File I/O / data loading | ✅ M22 | `read_file`, `str_len`, `str_char_at`, `str_from_char`, `Buffer<i32>`, `freeze`, `rand_int`, `rand_uniform`, `tensor.data[i]` — all done |
| Non-f32 compute dtypes (f16, bf16) | **V5/M36 planned** | Only i32/i64 for index tensors today; bf16 autocast-style mixed precision lands post-gate in M36 (ADR-0037); f16 + loss scaling stays deferred |
| Cross-module structs/enums unsupported (loader `exported_names` gap) | Post-V4 | See `docs/milestones/cross-module-types.md` |
| ScalarBroadcast IR node | Post-V4 | Inline scalar-broadcast BinOps work; dedicated IR node deferred |
| CTMM barrier coalescing is conservative | Post-V4 | ADR-0009 "Consequences" |
| `MetalContext` is single-consumer, not thread-safe under concurrent host access | By design | Metal-touching test files (`malus-runtime/src/tests.rs`, `malus-codegen-cpu/tests/metal_integration.rs`) must serialize via a per-file `Mutex<()>` test lock, held for the whole test body — not reentrant, don't nest acquisitions. See ADR-0033. |
| `Variable` type (type-directed RC) | ✅ M27 | Eliminated; replaced by single `Tensor` type + whole-program static grad-inference (`malus-sema/src/grad_inference.rs`, ADR-0030) |
| Generics / `impl` / `Module` trait / `List<T>` | ✅ M28 | Generic `fn`s only (structs deferred, ADR-0034); `Module`/`impl Module for GPT`/one generic `fn adamw<M: Module>`; `List<T>` is a reference-counted aggregate, not Array-style static-drop (ADR-0034) |
| Lobster borrow-inference RC elimination | ✅ M29 | The founding CTMM differentiator; single owner + zero-cost borrow for scalar `Tensor`s (params, same-scope aliases, struct-init field transfers); RC survives only for `List<T>` and unprovable struct fields; ≤5% compile-time RC-reduction-ratio gate (`malus-sema/src/borrow_inference.rs`); ADR-0026 |
| Kernel language beyond elementwise maps | ✅ M24 | Thread hierarchy, flat indexing, `let shared`, `barrier()`, control flow, scalar uniforms (ADR-0027) |
| Multi-dim `a[i,j]` indexing + `TensorMeta` strides | **M25** | M24 uses flat 1-D indexing; rank/stride infra deferred with launch-config |
| Stdlib ops as malus kernels (dogfooding) | ✅ M25/M26 | Forward (M25) and backward (M26) ops are malus kernels; only `matmul` stays a Rust/MPS vendor builtin (ADR-0028) |
| Backward kernels on GPU | ✅ M26 | Every VJP is a malus kernel + host fn (ADR-0032); `tape.rs` keeps only the tape-walk orchestration; canonical `count()==0` full-train-step gate passes |
| Embedding backward scatter-add uses per-row gather, not atomics | Post-V4 | Deterministic and exact at nanoGPT's char-level vocab scale; `atomic<f32>`/`atomic_fetch_add_explicit` deferred as a real kernel-language feature for large-vocab efficiency (ADR-0032) |
| Backward reductions inherit forward's ≤1024 reduced-axis cap | Post-V4 | Single-threadgroup `Array<f32,1024>` scratch (M25); grid-stride reduction to lift it deferred — never hit at nanoGPT's gate config (ADR-0032) |
| `gpu_barrier()` is barrier-before-*drop*, not barrier-before-*read* | **V5/M31 planned** | CTMM's `insert_barriers` only flushes before a pending Tensor's static drop; RC-managed reads can see stale GPU state if nothing else triggers a flush first. Worked around per-call-site (e.g. `examples/gradient_check.ml`'s `__flush()`). M31 fixes it as a runtime guarantee: per-buffer pending tracking + auto-flush on host read (ADR-0035) |
| Flash attention | V6 | Composed attention ships V4 (ADR-0029); flash requires simdgroup_matrix + mixed precision (bf16 lands M36) |
| Dispatch architecture is sync-per-matmul eager | **V5/M31–M32 planned** | Measured 2026-07-01 (M30 warm median): toy nanoGPT 26.187 ms/step ≈ 9.6x slower than f32 PyTorch-MPS; `tensor_matmul` does commit+waitUntilCompleted per call, fresh `MTLBuffer` per op, global-flush barriers. V5 async substrate + pooling is the response (ADR-0035, `docs/milestones/m29-benchmark-results.md` M30 addendum) |
| `permute` backward is hardcoded to rank ≤3 — 4-D permute has no working gradient | **V5/M33 planned** | `tape.rs:591-599` passes exactly 3 inverse indices; blocks multi-head attention (head-folding needs differentiable 4-D permute). Forward permute is already rank-generic |
| `List<Struct>` elements leak on drop | **V5/M34 planned** | `DropList` only drops tensor elements; struct/list elements are silently skipped (`malus-codegen-cpu/src/lib.rs:1035-1037`). Only `List<Tensor<f32>>` is sound today; M34 makes drop type-directed and recursive |
| Model save/load / checkpointing (SafeTensors) | V6 | Consciously sequenced after the V5 perf claim; deserves its own designed milestone |

## Coding conventions

- No comments unless the why is non-obvious
- No docstrings
- Rust 2021 edition throughout
- Tests live in `src/tests.rs` (a separate file, not inline) — see `malus-sema` for the pattern
- `cargo test --workspace` must pass before committing
- File extension for malus source files: `.ml`
