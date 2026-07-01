# M33 — N-D Permute Backward + Multi-Head Attention

**Crates:** `malus-runtime` (tape), `malus-stdlib` (backward kernel), `examples/`
**Track:** capability
**Depends on:** M30 (timing); independent of M31/M32 for correctness
**Status:** planned

Make true multi-head attention differentiable. The entire head-folding forward path already works — 4-D `reshape` (rank-agnostic zero-copy), 4-D `permute` (rank-agnostic, materializing), 3-D batched matmul, axis-generic softmax, trailing-dim mask broadcast. The single blocker is that permute's VJP is hardcoded to rank ≤ 3: `tape.rs:591-599` selects `PermuteBwd2D`/`PermuteBwd3D` and passes exactly three inverse indices, and `__permute_bwd_3d` (`stdlib/backward/permute_bwd.ml:13`) calls a 3-arg `permute` that asserts `perm.len() == rank`. A 4-D permute currently has no working gradient.

## Done-When

1. Permute's VJP is rank-generic: one backward path that applies the full rank-N inverse permutation. The `PermuteBwd2D`/`PermuteBwd3D` pair is replaced (or generalized); no hardcoded index counts remain in the permute tape path.
2. Gradient check for 4-D permute (all 24 permutations of a small [2,3,4,5] tensor is overkill; a representative set including the attention permutation `(0,2,1,3)` and its inverse) passes at the existing 1e-3 tolerance.
3. `examples/nanogpt.ml` attention is head-folded multi-head: `[B,T,C] → reshape [B,T,H,hs] → permute (0,2,1,3) → reshape [B*H,T,hs] → scores/softmax/attn·V → unfold back to [B,T,C]`, with H=6 at the capstone config (may land at reduced dims until M34/M35 assemble the full capstone). End-to-end gradient check on a small multi-head attention block passes.
4. Full-step `cpu_compute_count()==0` still holds (the new backward must be a malus kernel + host fn per ADR-0032, not a Rust loop).
5. `cargo test --workspace` passes.

## Scope

### 1. Rank-generic permute VJP

Two acceptable implementations — decide at implementation time:
- **Variadic uniforms:** one `__permute_bwd` kernel taking the inverse permutation as scalar uniforms up to a max rank (rank ≤ 8 covers everything PyTorch supports in practice), with rank passed as a uniform.
- **Host-side composition:** the VJP host fn calls the existing rank-agnostic *forward* permute machinery (`permute_by_perm`) with the inverse permutation — the forward is already a GPU path post-M25, so this stays CPU-counter clean and needs no new kernel.

The second is likely smaller; the first is more uniform with other backward kernels. Either satisfies the gate.

### 2. Reshape-after-permute contiguity

`permute_by_perm` materializes a fresh contiguous buffer, so the subsequent zero-copy `reshape` is valid (ADR-0023's trust-the-caller model holds). Document this invariant in the attention example comments — it is the one place the zero-copy reshape's "caller guarantees contiguity" contract is load-bearing.

## Out of Scope

- 4-D (or N-D) batched matmul — head-folding makes it unnecessary; reserve for a future PyTorch-parity milestone (ADR-0022: additive).
- Flash/fused attention (V6; ADR-0029 still governs — composed attention remains the shipped form).
- `view` (strided, non-materializing permute) — post-V5 as before.
