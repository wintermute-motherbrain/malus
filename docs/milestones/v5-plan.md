# malus V5 Plan — Earning the Claim

## What V5 Is For

V4 reclaimed the vision's *differentiators*: real borrow-inference CTMM, a real GPU kernel language with a dogfooded stdlib, one `Tensor` type, a generic `Module` optimizer. What V4 did not do is earn the vision's *claim* — "train models like PyTorch, without the Python slowness." When both sides of the M29 benchmark were finally run (2026-07-01), the toy nanoGPT config measured **~60x slower than f32 PyTorch-MPS** (~163.9 ms/step coarse average vs 2.729 ms/step median). The V4 plan's soft "investigate" trigger was >3x.

The gap is architectural, not compute: every `tensor_matmul` performs a full blocking `gpu_barrier()` + `commit()` + `waitUntilCompleted()` round-trip (`metal.rs:495-531`), every op allocates a fresh `MTLBuffer`, and the only barrier primitive is a global flush. At toy scale the step is nearly all synchronization stalls. Cranelift-compiled host code buys nothing while it blocks on the GPU dozens of times per step.

V5 exists to make the claim true, at a scale that means something:

**Dispatch architecture.** Replace sync-per-matmul eager dispatch with an async substrate: all ops (including MPS matmul) encode into a shared command buffer; commits happen rarely; host reads are made safe by runtime pending-tracking. This is phase 1 of the long-term execution model — a compile-time graph — recorded in ADR-0035.

**A capstone that is actually nanoGPT.** The V4 capstone is 1 block, 1 head, C=32, T=16. V5's capstone is the Karpathy char-Shakespeare config — 6 layers, 6 heads, n_embd=384, block_size=256, batch 64 — written idiomatically (named submodules, no index-arithmetic unrolling) and trained until samples are recognizably Shakespeare-ish.

**A hard gate.** V4's benchmark bar was soft and went unmeasured until after the milestone closed. V5's is hard: ≤2x of f32 PyTorch-MPS at the capstone config, matched methodology, or V5 is not done.

## V5 Done-When Program

`examples/nanogpt.ml` is the Karpathy config above and runs on an M-series Mac where:

1. **≤2x of f32 PyTorch-MPS.** Steady-state median step time ≤ 2× the PyTorch-MPS f32 median at the identical config, same machine, matched methodology (both warm, both median). **Hard gate.** Parity (≤1x) is the stretch goal, not the gate.
2. **Idiomatic model code.** `GPT { blocks: List<Block>, ... }` with `impl Module for Block`; the optimizer recurses over submodules; no flat-list index arithmetic, no hand-unrolling.
3. **Prior gates hold.** Full-step `malus_cpu_compute_count() == 0`; borrow-inference RC-reduction ratio ≤ 5%; no-unroll lint; gradient checks within tolerance.
4. **No stale reads, no workarounds.** Barrier-before-read is a runtime guarantee (per-buffer pending tracking + auto-flush); every `__flush()` call site is deleted.

## Milestone Sequence

Sequential: M30 → M31 → M32 → M33 → M34 → M35 → M36. M33 and M34 are independent of each other and may land in either order; both depend on M31/M32 only for final performance numbers, not correctness.

| Milestone | Theme | Key Deliverable | Gate |
|---|---|---|---|
| [M30](./m30-honest-timing.md) | Honest timing baseline | Per-step steady-state timer; publish the 60x baseline; docs hygiene | Timer reports warm median; baseline documented |
| [M31](./m31-async-dispatch.md) | Async dispatch substrate | MPS matmul joins the shared command buffer; per-buffer pending flags; auto-flush on host read; `__flush()` deleted | Full test suite + gradient checks green under async dispatch; toy step time published |
| [M32](./m32-buffer-pooling.md) | Buffer pooling + memory budget | Size-class MTLBuffer free-list; peak-memory measurement | Pool hit-rate + peak-memory reported; capstone-config budget fits |
| [M33](./m33-nd-permute-multihead.md) | N-D permute backward + multi-head | Rank-generic permute VJP; head-folded 6-head attention | 4-D permute + folded-attention gradient checks pass |
| [M34](./m34-named-submodules.md) | Named submodules | `List<Struct>` recursive drop; optimizer recursion over submodules | Submodule nanoGPT trains; no leaks; lint updated + green |
| [M35](./m35-capstone-benchmark.md) | Capstone + benchmark gate | Karpathy-config nanoGPT trains to Shakespeare-ish samples; both-sides benchmark | **≤2x hard gate**; README rewritten |
| [M36](./m36-bf16-mixed-precision.md) | Mixed precision (bf16-first) | bf16 compute dtype + autocast-style training | Convergence parity with f32; speedup published (soft ≥1.3x) |

## Design Decisions

All decisions were locked during the V5 planning session (2026-07-01). Do not re-litigate them without user input.

| Decision | Choice | ADR |
|---|---|---|
| North star | Performance-first: earn "without the Python slowness" before language surface/tooling | — |
| Capstone scale | Karpathy char-Shakespeare config (6L/6H/384d/T=256/B=64); toy config kept as dispatch-overhead regression benchmark | — |
| Perf bar | ≤2x f32 PyTorch-MPS, **hard gate**; parity stretch. V4's soft bar was the mistake | — |
| Execution model | Compile-time graph is the endgame; V5 builds the async runtime substrate; runtime lazy graph capture rejected | ADR-0035 |
| Read safety | Runtime per-buffer pending tracking + auto-flush on host read; static CTMM barriers demoted to optimization | ADR-0035 |
| Multi-head | Head-folding via N-D permute + existing 3-D batched matmul; no 4-D matmul | — |
| Submodules | Named submodules; optimizer recurses per-submodule; `parameters()` concat rejected (breaks identity write-back) | ADR-0036 |
| Mixed precision | bf16-first, autocast-style, **in V5** (M36, after the f32 gate); f16 + loss-scaling deferred | ADR-0037 |
| Persistence | Save/load/SafeTensors deferred to V6 | — |
| Tooling | Docs hygiene only (README rewrite, benchmark doc); REPL/LSP/fmt/CLI verbs deferred | — |

## What V5 Does NOT Include

Deferred to V6+:

- Static scheduling + kernel fusion over the typed IR (phase 2 of ADR-0035's ladder; rides the M31 substrate)
- Fused flash attention (`simdgroup_matrix`; natural companion to bf16)
- Model checkpoint save/load (SafeTensors) — acknowledged as "a trainer you can't save from isn't a trainer," consciously sequenced after the perf claim
- Tooling arc: formatter, LSP, REPL, CLI subcommands, `.ml` test runner, package management
- Language surface: closures/higher-order fns, `Option<T>`/`Result`, generic structs/enums, growable `List` (`push`)/`concat`, cross-module struct/enum types, `Dict`
- Multi-GPU / distributed training
- Static backward (compiling the tape away) — V7 option per ADR-0035
- f16 with loss scaling; user-settable RNG seed; `view`/`gather`; atomics scatter-add; grid-stride reductions
