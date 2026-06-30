# ADR-0029 — Composed Attention Now; Flash Attention Reserved Post-V4

**Status:** Accepted (V4)

## Context

The nanoGPT north star requires a causal multi-head attention implementation. Two approaches are in scope:

**Composed attention:** `attention(Q,K,V) = softmax(Q @ Kᵀ / √d + mask) @ V`. Uses MPS-matmul for Q@Kᵀ and V-projection, a malus softmax+mask kernel in between. Three separate GPU passes; materializes the full `[B, H, T, T]` scores matrix.

**Fused flash attention:** Online softmax (Dao et al.), tiling Q/K/V in shared memory to avoid materializing `[B,H,T,T]`. Requires the softmax+matmul to be fused in a single kernel — which would require in-kernel matmul on AMX/tensor cores, a deliberate exception to the vendor-primitives rule (ADR-0028).

## Decision

**V4 ships composed attention. Fused flash attention is reserved post-V4.**

**Rationale for composed:**
- Correctly implements the attention mechanism for nanoGPT at the context lengths in the capstone (≤ 256 tokens; the T² memory scaling is immaterial at this scale).
- Reuses the MPS-matmul builtin (ADR-0028) and the malus softmax kernel (ADR-0027) — both already required for V4 M2/M3.
- Differentiable: matmul VJP (MPS) + softmax VJP kernel + matmul VJP (MPS). No new backward machinery beyond M3.
- V4 is a "reclaim the vision" release where correctness and completeness beat performance ceiling. Composed attention is correct and compositional.

**Why flash attention is reserved:**
- A fused kernel needs in-kernel matmul. On Metal, this currently requires `simdgroup_matrix` intrinsics (Apple's WMMA analog) which are not yet in the malus kernel IR.
- Adding `simdgroup_matrix` to the kernel language is additive (post-V4), does not break any existing kernel, and is a natural companion to mixed-precision support (the other major post-V4 perf milestone).
- The composed path's benchmark ceiling (Nx vs PyTorch-MPS with composed attention) sets the V4 performance target. Flash attention can only improve on that.

## Consequences

- `attention` in the nanoGPT capstone = MPS-matmul(QKᵀ) → causal mask + softmax kernel → MPS-matmul(scores·V).
- No new `attention` builtin is needed: this is written as user-level `.ml` using the kernel-language softmax + MPS matmul.
- The `[B,H,T,T]` scores buffer is materialized → O(T²) memory. Acceptable for the capstone scale.
- Post-V4 flash attention is additive: it replaces the composed attention in the stdlib without breaking user code that calls `softmax`/`matmul` directly.
