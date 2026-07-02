# M29 Benchmark Results

ADR-0026 / D7: the V4 performance baseline. Informational — the V4 plan sets
no hard pass/fail threshold at this milestone; the Nx ratio (malus /
PyTorch-MPS) is the number V4 hands off, with `>3x` a soft
"investigate before declaring V4 done" trigger, not a gate.

## Machine

- Chip: Apple M4 Max, 48 GB unified memory
- macOS 26.5.1 (build 25F80)
- rustc 1.96.0
- `cargo build --release` (malus-cli)

## Config

Both sides run the exact architecture and dims `examples/nanogpt.ml`'s
`fn main()` uses: single-block causal self-attention + GELU-MLP char-GPT,
`C=32` (embedding), `T=16` (context), `B=4` (batch), `V=128` (vocab),
`C4=128` (MLP hidden), f32, AdamW (`lr=0.001, beta1=0.9, beta2=0.999,
eps=1e-8, wd=0.01` — an earlier revision of this doc said `lr=0.01`,
contradicting `examples/nanogpt.ml:157`; `0.001` is what the code runs),
trained on `data/tiny_shakespeare.txt` with char-level tokenization.

## Results

### malus (measured)

```
$ bash bench/nanogpt_step.sh
malus nanoGPT: full run (300 steps) = 49.407861000s, avg/step = 0.1647s
```

**300 steps, 49.41s total, ~164.7ms/step** (coarse whole-process average,
`bench/nanogpt_step.sh` — includes one-time cost: process startup, MSL
kernel compilation, `data/tiny_shakespeare.txt` load/tokenize amortized
over 300 steps; not a true per-step median).

### PyTorch-MPS (measured 2026-07-01)

Both sides run by the user on the same M4 Max machine:

```
malus nanoGPT:       full run (300 steps) = 49.177331s, avg/step = 163.92ms
PyTorch-MPS nanoGPT: 20 steps, median step = 2.729ms (min=2.550ms, max=3.389ms)
```

(The malus re-run reproduced the original measurement within noise:
49.18s vs 49.41s total.)

### Nx ratio

**Nx ≈ 60x** (163.9 ms / 2.729 ms ≈ 60.1). Even discounting a generous
share of the malus number as one-time startup/MSL-compile/data-load cost,
steady state would remain ~45–50x. The V4 soft "investigate" trigger was
>3x; this result is the founding motivation for V5 (see
`docs/milestones/v5-plan.md` and ADR-0035).

**Diagnosis:** at this toy scale both runtimes are dispatch-bound, not
compute-bound, so the ratio measures dispatch architecture. malus performs
a full blocking `gpu_barrier()` + `commit()` + `waitUntilCompleted()`
round-trip inside every `tensor_matmul` (~20+ per step across
forward+backward), allocates a fresh `MTLBuffer` per op, and its only
barrier is a global flush; PyTorch-MPS pipelines the whole step. The gap
is architectural, not a borrow-inference/CTMM artifact and not kernel
quality. V5/M31 (async dispatch substrate) is the response.

## M30 addendum — warm per-step median (measured 2026-07-01)

M30 added the matched-methodology timer (ADR-0038): `malus --bench` skips 3
warmup steps and reports the per-step median with a `gpu_barrier()` flush
inside the timed region, mirroring `bench/nanogpt_pytorch.py`'s
`torch.mps.synchronize()`-inside-the-step median. Same machine, same config,
same day as the coarse measurement above:

```
$ bash bench/nanogpt_step.sh
malus bench: 297 warm steps, median step = 26.187ms (min=24.242ms, max=30.983ms)
(whole-process wall-clock incl. startup/MSL-compile/data-load/generation: 49.208530000s)
```

**Matched Nx ≈ 9.6x** (26.187 ms / 2.729 ms). The coarse 164ms/step figure —
and this doc's earlier prediction that the matched number would land at
45–55x — overstated the steady-state gap by ~6x: a timestamped run shows the
300-step training loop spans only ~7.9s and post-training generation <1s of
the ~49.2s process; the remaining ~40.6s is one-time startup before step 1
(MSL compilation of the full M25/M26 stdlib kernel set, tiny_shakespeare
load/char-tokenize, Cranelift JIT). That startup cost is real UX but is not
per-step dispatch overhead, and PyTorch's median never counted its own
equivalents.

**What this changes:** every V5 milestone reports its delta against
**26.187 ms/step (9.6x)**, not 164 ms (60x). **What it doesn't change:** the
V5 motivation and diagnosis. 9.6x is still ~5x short of the M35 ≤2x gate, and
the causes are the same architecture measured here — sync-per-matmul
`commit()+waitUntilCompleted()` round-trips, a fresh `MTLBuffer` per op,
global-flush barriers (M31/M32). The 60x figure should no longer be quoted as
the steady-state gap; it was the coarse whole-process ratio.

## Known caveats affecting comparability

- The malus number is a coarse whole-run average (300-step total / 300),
  not a steady-state per-step median; it includes one-time MSL
  kernel-compilation and data-load overhead that a `--bench`-style
  per-step timer (noted as a follow-up in `bench/nanogpt_step.sh`) would
  exclude. The true steady-state per-step cost is likely somewhat lower
  than 164.7ms.
- The two methodologies are therefore not strictly comparable (coarse
  whole-run average vs warm steady-state median). V5's M30 adds the
  per-step warm-median timer to malus so subsequent comparisons are
  matched. Because the malus average includes one-time startup cost, the
  ~60x figure slightly overstates the steady-state gap — the matched
  number will likely land in the ~45–55x range, which changes nothing
  about the conclusion. *(M30 correction: this prediction was wrong — the
  matched number is ~9.6x; the overstatement was ~6x, not "slight". See
  the M30 addendum above.)*
- malus dispatches per-op Metal kernels with CTMM-inserted barriers
  (composed attention, ADR-0029); PyTorch's MPS backend uses fused
  scaled-dot-product-attention and heavily kernel-fused ops. Some gap is
  structural, not a borrow-inference/CTMM artifact — this is exactly why
  V4 sets no hard threshold at this milestone (V4 plan: "final Nx target
  set empirically after the baseline measurement").

## M31 addendum — async dispatch substrate (measured 2026-07-01)

M31 killed sync-per-matmul: MPS matmul now encodes into the shared command
buffer like every other op (zero `commit`/`waitUntilCompleted` inside any op
function), read safety is a runtime guarantee (per-buffer commit-generation
pending tracking + auto-flush on host read), and CTMM static barrier
insertion is off by default (ADR-0035). Same machine, same toy config, same
methodology as the M30 addendum:

```
$ malus examples/nanogpt.ml --bench
malus bench: 297 warm steps, median step = 6.065ms (min=5.291ms, max=7.436ms)
```

**26.187 ms → 6.065 ms/step (4.3x faster); matched Nx ≈ 2.2x**
(6.065 ms / 2.729 ms PyTorch-MPS f32). Not a gate — the V5 gate is M35's
≤2x at the Karpathy config — but the toy config is now within sight of it
before buffer pooling (M32) has landed.

Supporting measurements, same day:

- **A/B, static barriers re-enabled** (`--static-barriers`): 24.015 ms/step —
  ≈ the M30 baseline. This confirms the M31 design call empirically: each
  CTMM `GpuBarrier` is a full commit+wait fired before pending drops, so
  leaving the pass on would have nullified the async substrate almost
  entirely. Read-safety cannot live in static barriers under this
  execution model.
- **Numerics unchanged**: the 300-step loss curve is bit-identical to a
  pre-M31 build (deterministic Philox RNG, unchanged op order) — async
  encoding reordered nothing observable.
- **Remaining gap attribution**: with dispatch syncs gone, the ~3.3 ms/step
  over PyTorch is dominated by per-op `MTLBuffer` allocation (M32 buffer
  pooling), per-call MPS object churn (`MPSMatrix`/`MPSMatrixMultiplication`
  created per matmul — an M32 companion candidate), and per-op encoder
  overhead (V6 fusion territory).
