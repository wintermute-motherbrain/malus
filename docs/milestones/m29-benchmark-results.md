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
`C4=128` (MLP hidden), f32, AdamW (`lr=0.01, beta1=0.9, beta2=0.999,
eps=1e-8, wd=0.01`), trained on `data/tiny_shakespeare.txt` with char-level
tokenization.

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

### PyTorch-MPS (not measured in this environment)

`bench/nanogpt_pytorch.py` is written and ready (matches the malus
architecture/dims exactly — see the script's own `Block`/`GPT` classes) but
**could not be run**: this session's environment has no `torch` installed
(`ModuleNotFoundError: No module named 'torch'`). To complete the
comparison on a machine with PyTorch + MPS:

```
pip install torch
python3 bench/nanogpt_pytorch.py --steps 20
```

### Nx ratio

**Not yet computed** — pending the PyTorch-MPS run above. Once both
numbers exist, ratio = malus per-step time / PyTorch per-step median time.

## Known caveats affecting comparability

- The malus number is a coarse whole-run average (300-step total / 300),
  not a steady-state per-step median; it includes one-time MSL
  kernel-compilation and data-load overhead that a `--bench`-style
  per-step timer (noted as a follow-up in `bench/nanogpt_step.sh`) would
  exclude. The true steady-state per-step cost is likely somewhat lower
  than 164.7ms.
- malus dispatches per-op Metal kernels with CTMM-inserted barriers
  (composed attention, ADR-0029); PyTorch's MPS backend uses fused
  scaled-dot-product-attention and heavily kernel-fused ops. Some gap is
  structural, not a borrow-inference/CTMM artifact — this is exactly why
  V4 sets no hard threshold at this milestone (V4 plan: "final Nx target
  set empirically after the baseline measurement").
