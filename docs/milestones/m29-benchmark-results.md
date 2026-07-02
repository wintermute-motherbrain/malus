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

## M32 addendum — buffer pooling + memory budget (measured 2026-07-01)

M32 recycles `MTLBuffer`s through a generation-gated exact-size pool
(ADR-0039), fills pooled `zeros` via GPU blit, replaces the per-dispatch
uniforms/TensorMeta `MTLBuffer`s with `setBytes`, and caches
`MPSMatrixMultiplication` kernels by shape. Same machine, same toy config,
same methodology as the M30/M31 addenda:

```
$ malus examples/nanogpt.ml --bench          (5 runs)
medians: 2.422 / 2.477 / 2.604 / 3.651 / 3.729 ms   (min 2.03, max 6.60)
malus pool: 113731 hits / 29217 misses (79.6% hit rate), peak device 204.7 MB
```

**6.065 ms → ~2.6 ms/step (median-of-5 runs 2.604 ms); matched Nx ≈ 0.95x —
parity with PyTorch-MPS f32 (2.729 ms) at the toy config.** Run-to-run
medians are bimodal (2.4–2.6 vs 3.6–3.7, likely thermal/scheduler state), so
the honest statement is "0.9–1.4x, ≈ parity", not the best run. Not the
gate — M35's ≤2x at the Karpathy config is — but the toy config now sits at
the level the V5 plan hoped to reach only after fusion.

Supporting measurements, same day:

- **Numerics unchanged**: 300-step loss curve bit-identical to the M31
  build (fully-overwritten outputs make dirty pooled buffers invisible;
  blit-filled zeros are exact).
- **Allocation histogram** (`MALUS_ALLOC_HISTOGRAM=1`, the M32 spec
  deliverable): exactly 10 distinct sizes over the whole run (4 B–32 KB,
  all already powers of two). Size-class rounding would be a no-op —
  exact-size buckets confirmed. The 20% misses are gen-gated (pending
  entries between flush points), not size scatter; only V6's commit
  planner can convert those.
- **Memory-budget proxy at capstone dims** (single-head 6L/384d/T=256/B=64,
  12 steps): peak device **12.9 GB** (of which ~8.8 GB is end-state pooled
  temporaries), 541 ms/step, 77.7% hit rate. Peak is set during step 1
  (first-touch misses for one step's whole temp footprint). Head-folded
  multi-head multiplies only the `[B,T,T]`-shaped attention tensors by
  H=6 (≈ +3 GB analytic) → expected real-capstone peak ≈ 16 GB —
  comfortable headroom on the 48 GB target. A naive B=384 (=B·H)
  everything-×6 upper bound measured 49.9 GB and swapped (76 s/step); it
  over-bounds activations 6x and is not representative.
- **Memory-budget valve**: read-free loops (no host read → no flush → pool
  never cycles → unbounded growth, a pre-existing M31 behavior) are now
  bounded by a soft valve — on an over-budget miss whose bucket holds
  pending entries, flush once and retry (default 8 GiB,
  `MALUS_MEM_BUDGET_MB`). Unit-tested; realistic loops never hit it.
- **Pre-existing bugs discovered** (not M32 regressions; reproduced on the
  M31 build): (a) a loop-carried `let mut x` tensor reassigned across
  `for` iterations under the tape trips the M29 over-release detector in
  `backward()`; (b) a 6-deep chain of block-fn calls frees the loss tensor
  early — `loss.data` after `backward()`+`adamw()` reads a freed buffer
  (returns 0 pre-M32, recycled garbage post-M32). Both need a sema/tape
  root-cause pass — candidates for M33/M34 hardening scope.

## M33 addendum — N-D permute + multi-head attention (measured 2026-07-02)

M33 is a capability milestone (rank-generic permute VJP + head-folded
multi-head attention), but it changes the benchmark harness itself:
`examples/nanogpt.ml` and `bench/nanogpt_pytorch.py` are now **true
multi-head (H=4, hs=8)** in lockstep, per the ADR-0038 amendment (benchmark
architecture changes must land on both sides simultaneously, with an
explicit re-baseline — the Nx ratio carries across the change; absolute
step-time history does not).

Same machine, same methodology. The machine ran hot this session: the
unmodified M32 HEAD re-measured **3.662 ms** (vs 2.4–3.7 bimodal recorded at
M32), so all comparisons below are same-session, interleaved.

**Single-head (old architecture), A/B for the specialized-kernel decision:**

```
HEAD (M32 build):                        3.662 ms
M33, 2-D/3-D fast paths kept (A):        3.736 / 3.752 / 3.748 ms
M33, all permutes via generic kernel(B): 4.011 / 3.901 / 4.034 ms
```

B is a consistent ~6.5% regression outside run-to-run noise → **the
`__transpose_2d_kernel`/`__permute_3d_kernel` fast paths stay** (A ships).
The rank-generic `__permute_nd_kernel` serves rank ≥ 4, the
`transpose(t,i,j)` axis-swap form, and **every** tape-side permute VJP —
the tape path is a single rank-generic call (done-when #1); the specialized
kernels survive purely as an eager-forward codegen fast path. A (3.75) vs
HEAD (3.66) ≈ 2%: the rank-generic backward costs nothing measurable.

**New MHA baseline pair (back-to-back, 300 steps):**

```
malus  examples/nanogpt.ml  --bench:   5.762 ms/step  (min 5.508, max 6.400)
PyTorch-MPS nanogpt_pytorch.py:        2.693 ms/step  (min 2.556, max 3.075)
matched Nx ≈ 2.14x
```

The step went 3.75 → 5.76 ms on the malus side: six extra permute
dispatches per step (Q/K/V fold + unfold, forward and backward) at
~0.25 ms/dispatch of per-op encoder overhead — the known V6-fusion gap,
now visible because MHA is dispatch-heavier. PyTorch absorbs the same head
split for ~0 ms (its transpose is a lazy view; materialization fuses into
the following op). This is the honest toy number going forward; parity
(0.95x) remains the recorded result *for the single-head architecture* at
M32.

Supporting notes, same day:

- **Pre-existing scale mismatch found and retired**: pre-M33,
  `nanogpt_pytorch.py` scaled scores by 0.35355 (=1/√8) while
  `examples/nanogpt.ml` used 0.17678 (=1/√32) — the "exactly matched"
  benchmark pair disagreed on attention scale. Both now compute
  1/√hs = 0.35355 by construction.
- **Loss trajectory** (MHA, 300 steps): 4.86 → ~2.6, healthy; end-to-end
  MHA gradient check (`check_mha` in `examples/gradient_check.ml`) passes
  at 1e-3 alongside 4-D permute checks for `(0,2,1,3)` and `(1,2,3,0)`.
- **Full-step CPU-counter gate** (`test_v4_m28_full_step_zero_cpu_compute`,
  now head-folded MHA): 0 — the rank-4 forward permute went GPU. Pre-M33
  a 4-arg `permute` silently fell back to a CPU loop (`permute_by_perm`,
  now `cpu_fallback`-only), contradicting the M33 spec's premise that the
  4-D forward "already worked" as a GPU path.

### Startup-cost correction (2026-07-02, post-M33)

M30 attributed the ~40 s one-time process cost to "startup, MSL compile,
data load/tokenize, JIT". Measured attribution: **~40.2 s of it was the
char-tokenize loop alone** — `str_char_at` re-validated the entire 1.1 MB
string as UTF-8 *and* scanned to the i-th char on every call (O(n²) over
the loop; its doc even said "suitable for small vocabularies"). The whole
compile pipeline (parse → sema → MSL compile of the full stdlib → Metal
pipeline creation → JIT) measures 0.044 s. Fixed by precomputing an
`ascii` flag per `StrBox` at construction: ASCII strings take an O(1)
byte-index path (multi-byte strings keep the char-indexed scan). Tokenize:
40.4 s → 0.32 s; whole-process `nanogpt.ml` run: ~42.5 s → **2.15 s**.
Warm per-step median unaffected (tokenize is outside the timed region) —
this changes no benchmark number, only the wall-clock sanity line.

## M34 addendum — named submodules + three lifetime-bug fixes (measured 2026-07-02)

M34 is a capability milestone (recursive drop, named submodules, optimizer
recursion — see the spec and ADR-0036/0040); its perf obligation is only
"don't regress the harness". Same machine, same methodology:

```
$ malus examples/nanogpt.ml --bench          (flat harness, unchanged file*)
medians: 5.822 / 5.828 / 5.862 / 5.878 ms    (M33 baseline: 5.762 ms)
$ malus examples/nanogpt_modular.ml --bench  (new 2-block modular form, informal)
medians: 10.311 / 10.339 / 10.339 ms
```

- **Flat harness: ~5.85 ms ≈ +1% vs M33** — within this machine's recorded
  session noise. Pool hit rate moved 80% → 74% (peak device 205 → 256 MB):
  the ungated alias retains (below) hold tensors to their full, correct
  lifetimes; the old number was partly early-free unsoundness.
- **Modular 2-block form: ~10.3 ms**, i.e. ~1.77x the 1-block flat step —
  sublinear in block count (embeddings/lm_head/data prep amortize). No
  pathological cost from List<Struct> access, method dispatch, or
  per-submodule optimizer calls. NOT a like-for-like architecture; recorded
  as the structural preview of the M35 capstone form, not a baseline.
- **Modular RC ratio (reported, not gated)**: 7 RC ops / 74 tensor bindings
  = 9.5% across the example's user fns (`test_m34_modular_example_rc_ratio_
  reported`). The M29 ≤5% gate on the existing corpus stays green.

Done-when #0 outcome — the two M32-addendum bugs plus one more found by
systematic probing, all three pre-existing (reproduced on pre-M34 HEAD),
all fixed with Metal regression tests:

- **(a) loop-carried `let mut` over-release**: CTMM's hoist-temp counter was
  per-*body*; outer body and loop body each minted `__malus_tmp_0`, the
  outer last-use scan attributed the inner temp's uses to the outer name,
  and the resulting stray post-loop Drop resolved (via codegen's flat
  variable map) to the inner temp — double-releasing the last iteration's
  value. Fixed: one function-unique counter threaded through all scopes.
- **(b) "loss.data reads a freed buffer"**: binding a container-element read
  (`let w = model.params[k]`) emitted a Drop with no balancing retain — the
  bind stole the container's reference. The tape's saved-operand retains
  masked each theft until backward()'s auto-clear, then corruption (0s,
  recycled garbage, up to SIGSEGV with two binds per element per step).
  Fixed: retain-on-bind for container-element reads (ADR-0040); the
  nanogpt.ml "inline reads only" rule is no longer load-bearing.
- **(c) non-grad alias double-release** (found during M34 verification):
  tensor alias retains were gated on `grad_tracked` (ADR-0026 D6) but the
  alias's static Drop never was — `let b = a; sum(b); sum(a)` with plain
  tensors, or the natural forward-loop shape `let mut x = x0;
  for blk in blocks: x = blk.forward(x)`, double-released. Surfaced at
  step 4+ of modular training as a matmul on a freed (shape-[]) tensor.
  Fixed: alias retains unconditional on tensor type;
  `demote_safe_borrows` strips the provably-redundant pairs.

*the flat harness file changed only in one comment (the retired inline-read
rule); code identical.
