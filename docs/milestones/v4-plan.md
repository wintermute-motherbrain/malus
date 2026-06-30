# malus V4 Plan — Reclaiming the Vision

## What V4 Is For

V4 reverses three categories of accumulated debt from V1–V3:

**Memory model.** The founding promise — "lifetime without annotations, static-free on the hot path, RC only where structurally ambiguous" — was never built. V3 ships conservative lexical last-use drop insertion and type-directed RC on `Variable`. The Lobster-style borrow-inference pass (single owner per allocation, all other uses are zero-cost borrows) is the actual founding differentiator. V4 builds it.

**GPU compute.** The `kernel` language expresses only `out[tid] = f(inputs[tid])`. Every transformer op — softmax, layernorm, gelu, cross-entropy, all backward passes — runs as Rust CPU loops. The V3 nanoGPT capstone uses zero `kernel` functions beyond elementwise arithmetic. V4 rewrites the kernel language into a real GPU programming model and rewrites the stdlib as malus kernels.

**Ergonomics.** No generics, no traits, no `Module` abstraction. The nanoGPT capstone hand-unrolls the optimizer (82 lines of explicit AdamW parameter updates), hand-specializes every function, and uses a `Variable`/`Tensor` split that forces constant boilerplate. V4 lands generics, `impl` blocks, one trait, `List<T>`, eliminates `Variable`, and writes a generic `adamw<M: Module>`.

## V4 Done-When Program

`examples/nanogpt.ml` runs on an M-series Mac where:

1. **Zero CPU compute on the hot path.** `malus_cpu_compute_count() == 0` over a full train step (forward + backward + optimizer). Every tensor arithmetic op dispatches to Metal.
2. **Real malus kernels.** `softmax`, `layernorm`, `gelu`, `cross_entropy`, and the attention mask kernel are authored in the malus kernel language, in the stdlib. The nanoGPT forward pass contains zero CPU-loop builtin calls.
3. **Generic abstraction.** The model is defined with `impl Module for GPT`; the optimizer is a single generic `fn adamw<M: Module>(...)`. No hand-unrolled parameter loops.
4. **Within Nx of f32 PyTorch-MPS.** Benchmark baseline established at M23/M25; final Nx target set empirically after the baseline measurement.

## Milestone Sequence

V4 has two parallel tracks converging at M4.

**GPU track (sequential):** M23 → M24 → M25 → M26

**Frontend track (parallel, targeting M28):** generics + impl + List, developed alongside M24–M26.

| Milestone | Theme | Key Deliverable | CI Gate |
|---|---|---|---|
| [M23](./m23-de-risk-spike.md) | De-risk spike | Extended `kernel_dispatch` ABI + CPU-compute counter | `count()==0` over one softmax dispatch |
| [M24](./m24-kernel-language-v2.md) | Kernel language v2 | Thread hierarchy, shared mem, barrier, arbitrary indexing, control flow in kernels | softmax/layernorm/gelu as `.ml` kernels, CPU-counter==0 |
| [M25](./m25-stdlib-forward-kernels.md) | Stdlib forward kernels | All CPU-loop forward ops replaced by malus `.ml` kernels | nanoGPT forward CPU-counter==0 |
| [M26](./m26-backward-kernels.md) | Backward kernels | GPU autograd — VJPs dispatch backward kernels | **Full-step CPU-counter==0** (canonical gate) + gradient_check |
| [M27](./m27-kill-variable.md) | Kill `Variable` | One `Tensor` type; static grad-inference | Zero `ResolvedTy::Variable` in IR; nanoGPT still passes M26 gate |
| [M28](./m28-module-trait.md) | Module trait + generic optimizer | Generics, `impl`, `List<T>`, `Module` trait, generic AdamW | No-unroll lint passes; nanoGPT trains with generic optimizer |
| [M29](./m29-borrow-inference.md) | Borrow-inference RC + benchmark | Lobster single-owner/borrow pass | RC-op-count ≤ ~5%; nanoGPT within Nx of PyTorch-MPS |

## Design Decisions

All decisions were locked during the V4 planning session (2026-06-29). Do not re-litigate them without user input. See the plan file at `/Users/jboldiga/.claude/plans/rippling-wandering-dusk.md` for full decision rationale.

| Decision | Choice | ADR |
|---|---|---|
| North star | Idiomatic GPU nanoGPT (4 hard constraints above) | — |
| GPU programming model | Explicit kernels (Mojo/Triton-like); transformer stdlib dogfooded in kernel language | ADR-0027 |
| matmul | MPS-backed blessed builtin (AMX); not in kernel language | ADR-0028 |
| Memory model | Real Lobster borrow-inference; one `Tensor` type; Variable eliminated | ADR-0026, ADR-0030 |
| Autograd | Hand-written GPU backward kernels per op; define-by-run tape kept; grad-tracking = static sema property | — |
| Abstraction | Generics + methods + one trait (`Module`) + `List<T>`; no Dict, no inheritance | ADR-0007 |
| Shapes | Runtime-only (defer static) | ADR-0013 |
| Milestone gates | Demo-gated CI asserts (CPU compute counter) | ADR-0031 |
| Attention | Composed (MPS-matmul → softmax kernel → MPS-matmul); flash reserved post-V4 | ADR-0029 |
| Precision | f32-only (plus existing i32/i64 index tensors); benchmark vs f32 PyTorch-MPS | — |

## What V4 Does NOT Include

Deferred post-V4:

- f16/bf16 mixed-precision compute (first post-V4 perf milestone)
- Fused flash attention (`simdgroup_matrix`, requires mixed precision)
- Static/gradual shape inference
- Auto-differentiation through user-written kernels (`custom_grad`)
- `Option<T>`, `Dict`, inheritance, named-submodule/state_dict
- Multi-GPU / distributed training
- Model checkpoint save/load (SafeTensors)
- Cross-module struct/enum types (loader `exported_names` gap)
- User-settable RNG seed
- `view` (non-contiguous strided tensor), `gather` (different PyTorch contract)
- Source-to-source autodiff through kernel IR
