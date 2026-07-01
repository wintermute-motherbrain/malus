# M35 — Capstone + Benchmark Gate

**Crates:** `examples/`, `bench/`, docs
**Track:** convergence of all V5 tracks
**Depends on:** M31, M32, M33, M34
**Status:** planned

Assemble the real capstone, train it, and pass the gate that defines V5.

## Done-When

1. `examples/nanogpt.ml` is the Karpathy char-Shakespeare config: **6 layers, 6 heads, n_embd=384, block_size=256, batch 64, V=vocab of tiny Shakespeare charset**, written with named submodules (M34) and head-folded multi-head attention (M33). The toy config moves to `examples/nanogpt_toy.ml` (or a config switch) and remains the dispatch-overhead regression benchmark.
2. It trains on `data/tiny_shakespeare.txt` until generated samples are **recognizably Shakespeare-ish** — real words, line structure, speaker headers (the standard nanoGPT smoke bar; document the loss reached and a sample in the results doc). Non-monotonic step noise is fine; gibberish is not.
3. **The V5 gate: malus warm-median step time ≤ 2× f32 PyTorch-MPS warm-median step time** at the identical config, same machine, same tokenizer, matched methodology (M30 timer vs `bench/nanogpt_pytorch.py` updated to the same config). Parity (≤1x) is stretch. If the gate fails, V5 is not done — profile, fix (double-buffering, targeted fusion of the worst offender, layernorm-affine fusion, whatever the profile says), re-measure.
4. Results published in `docs/milestones/m35-benchmark-results.md`: machine, config, both medians, the Nx ratio, deltas vs the M30 60x baseline, peak memory (M32 stats), and honest caveats.
5. README rewritten to V5 reality: current pitch, the capstone, the measured number, quickstart, current limitations. The V2-era `Variable<f32>` content is gone.
6. All standing gates green: full-step `cpu_compute_count()==0`, RC ratio ≤5%, no-unroll lint, gradient checks, `cargo test --workspace`.

## Scope

- Assembly + tuning only; all capabilities land in M31–M34. Expected integration work: batch-building at B=64/T=256 (Buffer<i32> path at larger sizes), lr/schedule-free hyperparameters that converge (Karpathy's char nanoGPT settings are the reference), sampling loop reuse.
- PyTorch comparison script updated to the same 6L/6H/384d config, f32, MPS, AdamW, same data — kept architecture-matched line-for-line as `bench/nanogpt_pytorch.py` already is for the toy.

## Out of Scope

- bf16 (M36 — the gate here is f32 vs f32).
- LR schedulers, dropout — the reference nanoGPT uses dropout, but at char-Shakespeare scale convergence to the smoke bar does not require it; capstone fidelity is structural (attention/MLP/residual/layernorm), not training-recipe-complete. Document the divergence.
- Checkpoint save/load (V6).
