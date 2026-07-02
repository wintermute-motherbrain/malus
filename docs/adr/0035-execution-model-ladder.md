# Execution model: async runtime substrate now, compile-time graph as the endgame; runtime lazy capture rejected

## Decision

The V4 benchmark (completed 2026-07-01) measured the toy nanoGPT at **~60x slower than f32 PyTorch-MPS** (~163.9 ms/step vs 2.729 ms/step median). The dominant cause is the eager execution model: every `tensor_matmul` performs `gpu_barrier()` + `commit()` + `waitUntilCompleted()` — a full blocking CPU↔GPU round-trip per matmul (`metal.rs:495-531`) — and the only barrier primitive is a global flush inserted by CTMM before *drops* (not reads; the ADR-0032 gap).

We fix this on a three-rung ladder, and we explicitly reject the industry-standard alternative (runtime lazy graph capture) at every rung:

1. **V5 (M31): async runtime substrate.** All GPU work, including MPS matmul via `encodeToCommandBuffer:`, encodes into the shared command buffer; nothing commits inside an op. Read safety becomes a *runtime guarantee*: every `TensorBuffer` tracks whether uncommitted GPU work has written it (a commit-generation stamp), and every host-side read (`tensor_print`, `.data[i]`, scalar extraction) flushes iff pending. Per-call-site `__flush()` workarounds are deleted. CTMM's static barrier insertion is demoted from correctness mechanism to optimization.
2. **V6: compile-time scheduling and fusion.** A sema pass over the typed IR plans commit points, buffer reuse, and kernel fusion statically — turning the runtime pending-checks into a fallback the same way RC is CTMM's fallback for memory.
3. **V7 (option): static backward.** With M27's whole-program grad-inference and every VJP now a known malus kernel (M26), the define-by-run tape (ADR-0015) is statically knowable in principle; compiling it away is a future decision, not a commitment.

## Why this is surprising

Every successful ML framework on this problem went the other way: PyTorch (torch.compile), JAX, tinygrad, and Apple's own MPSGraph all capture a graph *at runtime* and optimize it there. A future reader will ask why malus doesn't just do that. The answer is that runtime capture is the workaround dynamic languages pay for opacity: Python frameworks cannot see the program, so tracing is the only way to recover a dataflow graph. malus is a compiled language — the typed IR of a `fn` body *already is* the dataflow graph, with CTMM lifetime information attached. Bolting runtime capture onto it would build machinery the compiler is designed to obsolete, and that machinery would be thrown away at rung 2. The async substrate, by contrast, is permanent: any graph execution — static or captured — needs exactly this runtime (encode many ops, commit rarely, guarantee host reads flush).

It is also surprising that read-safety moves *from* the compiler *to* the runtime in a project whose founding philosophy is "static on the hot path." The trade: a wrong static barrier is a silent stale read (the ADR-0032 gap made this real — `__flush()` call sites exist because the static analysis missed reads); a runtime pending-check is one branch per host read, and host reads are rare by construction (once per step for loss printing). Static analysis returns at rung 2 as an *optimization* over a substrate that is already correct — exactly the CTMM shape: static free on the hot path, RC where ambiguous; here, static commit-planning on the hot path, runtime flush where reads happen.

## Considered alternatives

**Runtime lazy graph capture (tinygrad/MPSGraph style).** Fastest route to fusion today. Rejected: duplicates what the compiler already knows, gets ripped out at rung 2, and moves malus's identity from "compiled language" toward "interpreted framework with a JIT." The one thing runtime capture handles that static IR cannot — data-dependent op sequences — is served adequately by the substrate itself (those ops simply encode as they execute).

**Keep eager-with-sync and chase the gate with fusion only.** Rejected: at toy scale the measured time is nearly all synchronization stalls, not unfused kernels; no amount of fusion pays for a blocking round-trip per matmul.

**Fix read-safety statically (extend CTMM barrier insertion to reads).** Purer, but every new host-read builtin becomes a compiler change, and a missed site is a silent wrong answer. Rejected as the *correctness* mechanism; retained as the rung-2 *optimization*.

## Consequences

- `tensor_matmul`'s contract changes: its result is a **pending** tensor, not ready. CONTEXT.md's "Ready tensor" entry (which lists MPS ops as ready-after-wait) must be amended in M31.
- `gpu_barrier()` stops being the ambient safety mechanism; correctness lives in `flush_if_pending` on host reads. Tests must pass with static barrier insertion disabled (M31 done-when #4) to prove the runtime guarantee stands alone.
- (M31 implementation, 2026-07-01) `insert_barriers` is **off by default**, not merely optional: under async every static `GpuBarrier` is a full commit+wait fired before pending drops, which would re-create sync-per-drop and protects nothing (drops are memory-safe — command buffers retain resources). It survives behind an opt-in flag (`CheckOptions::insert_static_barriers` / hidden `--static-barriers`) as an A/B lever, to be deleted when V6's static commit-planner lands. GPU errors panic at the flush point with the Metal error description; `MALUS_SYNC_DISPATCH=1` restores per-op fault attribution.
- Loss printing (or any per-step host read) forces one flush per step — the same cadence PyTorch pays for `.item()`. This is the natural pipeline boundary, not a defect.
- Buffer pooling (M32) depends on the pending machinery: a buffer is recyclable only when its last writing generation has completed.
- The ladder is recorded so V6 work starts from "add the static planner," not from re-debating capture.
