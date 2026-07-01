# M30 — Honest Timing Baseline

**Crates:** `malus-runtime`, `malus-cli` (bench harness), docs
**Track:** perf
**Depends on:** M29 (V4 complete)
**Status:** done (2026-07-01)

**Outcome:** `bench_step_begin()`/`bench_step_end()` dormant builtins + `malus --bench` (ADR-0038). Measured warm per-step median: **26.187 ms (min 24.242, max 30.983, 297 warm steps)** → matched Nx ≈ **9.6x** vs PyTorch-MPS's 2.729 ms — not the predicted 45–55x. The coarse 164 ms/step figure was ~5/6ths one-time startup (~40.6s before step 1: MSL compile of the full stdlib kernel set, data load/tokenize, JIT) plus <1s generation. See the M30 addendum in `m29-benchmark-results.md`.

Give malus a steady-state per-step timer (the M29 spec §4.1 item that was skipped), record the V5 starting line honestly, and clean up the stale docs. No performance fixes in this milestone — just the truth.

## Done-When

1. `bench/nanogpt_step.sh` (or a `--bench` mode in `malus-cli`) reports a **warm per-step median**: skip ≥3 warmup steps, time each subsequent step with `std::time::Instant`, report median/min/max. Whole-process wall-clock is no longer the reported number.
2. `docs/milestones/m29-benchmark-results.md` is amended with the completed comparison: malus ~163.9 ms/step (coarse) vs PyTorch-MPS 2.729 ms/step (median), **Nx ≈ 60x**, measured 2026-07-01 on M4 Max — plus the new warm-median malus number once the timer exists. This closes V4 constraint 4 with an honest answer.
3. The toy-config benchmark is wired as the **dispatch-overhead regression benchmark** — a fast, always-runnable check that V5's substrate work moves the number and nothing regresses it. Not a CI assert (wall-clock gates flake); a documented manual/bench-script check.
4. Docs hygiene: `docs/milestones/m29-benchmark-results.md` lr corrected to `lr=0.001` (matches `examples/nanogpt.ml:157`). README updated only for factual errors (it still documents `Variable<f32>` and says "V2 in progress"); the full README rewrite lands in M35 when there is a result worth leading with.
5. `cargo test --workspace` passes.

## Scope

### 1. Per-step timer

Instrument the training loop timing at the *host* level: the bench harness runs `malus examples/nanogpt.ml` variant with per-step timestamps emitted (simplest: a `bench_step_begin()`/`bench_step_end()` builtin pair or a `--bench` CLI flag that wraps `main()`'s loop — decide at implementation; the requirement is warm median, not a specific mechanism). PyTorch's `bench/nanogpt_pytorch.py` already reports a warm median; malus must match that methodology or the ratio is meaningless.

### 2. Baseline documentation

Record in `m29-benchmark-results.md` and reference from the V5 plan. The 60x number is the "before" photo; every V5 perf milestone reports its delta against it.

## Out of Scope

- Any dispatch/allocation/barrier changes (M31/M32).
- README rewrite beyond factual corrections (M35).
