# Benchmark methodology: synced-step warm median

## Decision

The canonical malus performance number is the **warm per-step median** (M30): the median wall-clock time of a full training step — batch construction through optimizer update — with a **GPU flush inside the timed region**, measured after skipping 3 warmup steps. It is produced by a dormant builtin pair, `bench_step_begin()`/`bench_step_end()`, called in the training loop of `examples/nanogpt.ml` and activated only by the CLI's `--bench` flag; `bench_step_end()` calls `gpu_barrier()` before recording. The CLI reports `median (min/max)` over the warm steps after the run. Every V5 perf milestone (M31–M35) reports its delta against this number via `bench/nanogpt_step.sh`.

This methodology is fixed for the duration of V5. Changing it invalidates every before/after comparison the V5 plan is built on.

## Why this is surprising

Once M31 lands, the substrate pipelines GPU work across ops — and this benchmark deliberately defeats that across step boundaries by flushing inside every timed step. A serialized step looks like it undersells the async substrate. It doesn't: `bench/nanogpt_pytorch.py` calls `torch.mps.synchronize()` inside its timed step too, so both runtimes are measured as synced-step latency. The Nx ratio only means something if both sides pay the same serialization tax. The flush lives in `bench_step_end()` — bench runs serialize, normal runs (and the M35 capstone demo) pipeline freely.

The M30 measurement itself was surprising enough to justify recording: the M29 coarse number (whole-process 49.2s / 300 steps = 164ms/step) overstated steady-state per-step cost by ~6x. The warm median is 26.2ms/step — the other ~41s of the process is one-time cost (startup, MSL compile, data load/tokenize) plus post-training generation. The honest toy-config baseline is **~9.6x** PyTorch-MPS, not the ~60x recorded at M29 close (which the M29 doc itself flagged as methodology-mismatched, predicting — wrongly — that the matched number would land at 45–55x). The V5 motivation is unchanged in kind (9.6x is far from the ≤2x gate, and the causes are the same dispatch architecture), but every V5 delta is measured against 26.2ms, not 164ms.

## Considered alternatives

**Pipelined-interval measurement** (time between successive step completions, no per-step flush). Arguably the truer throughput number post-M31, but it doesn't match the PyTorch script's methodology, so the ratio would compare malus throughput against PyTorch latency. Rejected while the Nx ratio is the headline; may be added *alongside* (never instead) if M31+ wants to show pipelining wins.

**A `--bench` CLI mode with no language surface** (runtime hooks on `backward()` completion). No new builtins, but the measurement point lands mid-step, the flush semantics are murky once dispatch is async, and it silently breaks if the program structure changes. Rejected: two dormant builtins are a smaller cost than a fragile implicit contract.

**A separate bench copy of nanogpt.ml.** Keeps the example pristine, but cross-module structs are unsupported, so the copy duplicates ~200 lines around the `GPT` struct and drifts — the benchmark stops measuring the program users run. Rejected.

**Always-on timing in the example.** Forces a per-step barrier into the flagship demo forever, inhibiting exactly the cross-step pipelining M31 exists to demonstrate. Rejected.

## Consequences

- `bench_step_begin`/`bench_step_end` are permanent language builtins with a trivial contract: no-ops unless the host process enabled bench mode. They are not general-purpose timers; a user-facing `time_ns()` remains future work if ever needed.
- The warmup count (3) and the flush-inside-region rule are part of the number's definition. A result reported without them is not a "warm per-step median" (see CONTEXT.md).
- The toy-config benchmark (`bench/nanogpt_step.sh`) is the dispatch-overhead regression check — manual, not a CI assert (wall-clock gates flake). The M35 capstone gate is a separate, harder measurement at the Karpathy config.

## Amendment (M33, 2026-07-02): benchmark-architecture changes require lockstep + re-baseline

"This methodology is fixed" above governs *how* the number is measured.
M33 changed *what* is measured: `examples/nanogpt.ml` — which is the
benchmark harness — became true multi-head attention. The rule this sets:

- A change to the benchmarked program's architecture must land in
  `examples/nanogpt.ml` and `bench/nanogpt_pytorch.py` **in the same
  commit**, dims and op structure matched. A one-sided change silently
  turns the Nx ratio apples-to-oranges (M33 found the pair had in fact
  already drifted: the PyTorch side scaled attention by 1/√8 while malus
  used 1/√32).
- Each such change gets an explicit re-baseline addendum in
  `m29-benchmark-results.md`, with same-session interleaved measurements
  of old-HEAD and new. **The Nx ratio carries across the change; absolute
  step-time history does not** — post-M33 step times (MHA) are not
  comparable to M30–M32 step times (single-head) and must not be quoted
  against each other.
