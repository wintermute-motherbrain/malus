# M33 — N-D Permute Backward + Multi-Head Attention

**Crates:** `malus-runtime` (tape), `malus-stdlib` (kernels), `malus-sema` + `malus-codegen-cpu` (rank-8 `out=` launch config), `examples/`, `bench/`
**Track:** capability
**Depends on:** M30 (timing); independent of M31/M32 for correctness
**Status:** done (2026-07-02) — rank-generic `__permute_nd_kernel` + `__permute_nd_fwd` (rank ≤ 8); tape VJP is one rank-generic call through the same host fn (inverse perm); `out=` launch config extended to 8 dims; toy nanogpt + PyTorch reference rewritten to head-folded MHA (H=4) in lockstep, new baseline pair **5.762 vs 2.693 ms/step ≈ 2.14x** (see M33 addendum in `m29-benchmark-results.md`); 4-D permute + end-to-end MHA gradient checks pass at 1e-3; full-step CPU-counter gate holds under MHA.

Make true multi-head attention differentiable.

> **Premise correction (found at implementation, 2026-07-02):** this spec
> originally claimed the 4-D forward permute "already works — rank-agnostic,
> materializing" as a GPU path. False: the GPU forward was *also*
> rank-hardcoded (`__transpose_2d_kernel`/`__permute_3d_kernel` selected by
> codegen fast paths), and every other form — including the attention
> permute `(0,2,1,3)` and `transpose(t,i,j)` — fell back to
> `permute_by_perm`, a CPU loop that increments the CPU-compute counter. A
> head-folded forward pass alone would have failed done-when #4. The
> "host-side composition" option below was therefore wrong as written
> (it composed on a CPU loop); the shipped design makes the *forward*
> rank-generic on GPU and the VJP composes on that. A second blocker found
> at the same time: the `.ml` kernel-launch `out=[...]` config was
> hardcoded to 3 elements in sema/codegen-cpu, so a rank-4 output could
> not be declared from a `.ml` host fn; extended to rank ≤ 8 (TensorMeta's
> existing ceiling).

The remaining forward pieces did hold: 4-D `reshape` (rank-agnostic zero-copy), 3-D batched matmul, axis-generic softmax, trailing-dim mask broadcast. The VJP blocker was as described: `tape.rs` selected `PermuteBwd2D`/`PermuteBwd3D` and passed exactly three inverse indices; a 4-D permute had no working gradient.

## Done-When

1. Permute's VJP is rank-generic: one backward path that applies the full rank-N inverse permutation. The `PermuteBwd2D`/`PermuteBwd3D` pair is replaced (or generalized); no hardcoded index counts remain in the permute tape path.
2. Gradient check for 4-D permute (all 24 permutations of a small [2,3,4,5] tensor is overkill; a representative set including the attention permutation `(0,2,1,3)` and its inverse) passes at the existing 1e-3 tolerance.
3. `examples/nanogpt.ml` attention is head-folded multi-head: `[B,T,C] → reshape [B,T,H,hs] → permute (0,2,1,3) → reshape [B*H,T,hs] → scores/softmax/attn·V → unfold back to [B,T,C]`, with H=6 at the capstone config (may land at reduced dims until M34/M35 assemble the full capstone). End-to-end gradient check on a small multi-head attention block passes.
4. Full-step `cpu_compute_count()==0` still holds (the new backward must be a malus kernel + host fn per ADR-0032, not a Rust loop).
5. `cargo test --workspace` passes.

## Scope (as shipped)

### 1. Rank-generic forward GPU permute + composed VJP

One `__permute_nd_kernel` (rank ≤ 8): perm as 8 scalar uniforms staged into
a `let shared` scratch array (tg=[1,1,1] → thread-private, no barrier),
output flat index peeled per-dim against `x.shape[perm[dd]]`, gather via
`x.strides[perm[dd]]` — the `broadcast_binop.ml` rank-generic loop pattern.
Host fn `__permute_nd_fwd(x, p0..p7)` branches per rank for the static
`out=` literal (the `__reduce_sum_fwd` pattern). It is registered in the
backward-slot table (`BwdSlot::PermuteNdFwd`, the forward-fn-as-VJP
convention of ExpBwd/NegBwd/GradAcc): `tensor_permute` (Rust) normalizes +
validates every form (`normalize_perm`) and calls it; the tape's Transpose
arm computes the inverse perm (orchestration, ADR-0031) and calls the same
fn. `PermuteBwd2D`/`PermuteBwd3D` and `stdlib/backward/permute_bwd.ml` are
deleted; `permute_by_perm` survives only under the `cpu_fallback` feature
as the mock-wiring reference.

**A/B outcome:** the specialized `__transpose_2d_kernel`/`__permute_3d_kernel`
eager-forward fast paths stay — deleting them measured a consistent ~6.5%
toy-config regression (outside noise; interleaved runs, see the M33
addendum). The tape path itself is a single rank-generic call — done-when
#1's "no hardcoded index counts in the permute tape path" holds.

### 2. `out=` launch config extended to rank ≤ 8

`Array<i64,N≤8>` accepted for `out=` (sema derives N from the literal;
codegen sizes the slot from the typed length); `grid`/`tg` stay 3;
trailing-literal-0 ndim stripping unchanged.

### 3. Reshape-after-permute contiguity

Permute materializes a fresh contiguous buffer, so the subsequent zero-copy
`reshape` is valid (ADR-0023's trust-the-caller model holds). Documented in
the attention example comments — the one place the zero-copy reshape's
"caller guarantees contiguity" contract is load-bearing.

### 4. Benchmark lockstep (ADR-0038 amendment)

`examples/nanogpt.ml` is the benchmark harness, so its architecture change
to MHA required updating `bench/nanogpt_pytorch.py` in the same commit
(H=4/hs=8 — H must divide C=32; the spec's H=6 is realizable only at
capstone dims and lands with M35) and an explicit re-baseline. Lockstep
also retired a latent mismatch: the PyTorch side already scaled by 1/√8
while malus used 1/√32.

## Out of Scope

- 4-D (or N-D) batched matmul — head-folding makes it unnecessary; reserve for a future PyTorch-parity milestone (ADR-0022: additive).
- Flash/fused attention (V6; ADR-0029 still governs — composed attention remains the shipped form).
- `view` (strided, non-materializing permute) — post-V5 as before.
