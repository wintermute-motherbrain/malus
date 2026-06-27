# malus V3 Plan — nanoGPT

## What V3 Is For

V3 proves malus can express a real transformer, trained end-to-end with real data. The north star is **nanoGPT on Apple Silicon**: a decoder-only language model that reads a text file, trains on it, and generates plausible continuations.

V3 builds on V2's autograd by filling the stdlib gap — broadcasting, axis reductions, reshape, batched matmul, transformer ops, embeddings, and random init — and adding the language features needed for a clean model definition: lvalue assignment targets and a reusable AdamW optimizer. It also migrates `matmul` and heavy reductions to MPS so the capstone trains in reasonable wall-clock time on M-series hardware.

## V3 Done-When Program

`examples/nanogpt.ml` runs on an M-series Mac:

- Reads `data/tiny_shakespeare.txt` from disk
- Char-tokenizes it in-language, builds vocab, batches training data
- Trains a decoder-only transformer (token + positional embeddings, ≥2 causal self-attention + MLP + layernorm blocks, cross-entropy LM loss, AdamW optimizer) showing **decreasing loss** over the training run
- Samples and prints a plausible text continuation after training

The transformer architecture must be structurally faithful (real causal masking, real layernorm, real multi-head attention) — not a toy approximation. Model scale is tuned to train to visible improvement on an M-series Mac in a reasonable wall-clock time.

## Milestone Sequence

V3 is seven sequential milestones. Each has a standalone done-when, ordered correctness-first (gradient-check before MPS, MPS before capstone).

| Milestone | Theme | Key Features |
|---|---|---|
| [M16](./m16-broadcasting-axis-reductions.md) | Broadcasting + Axis Reductions | NumPy right-aligned broadcasting, `sum/mean/max/var` over axis with `keepdim`, VJPs |
| [M17](./m17-shapes-batched-matmul.md) | Shapes + Batched Matmul | `reshape`/`view`, `transpose(dims)`, 3-D/batched matmul, VJPs |
| [M18](./m18-transformer-stdlib.md) | Transformer Stdlib | `softmax`, `layernorm`, `gelu`, `cross_entropy`, causal mask, VJPs |
| [M19](./m19-embeddings-index-tensors.md) | Embeddings + Index Tensors | i32/i64 index tensors, `gather`/embedding lookup, scatter-add VJP, `randn`/Philox |
| [M20](./m20-lvalue-assignment-adamw.md) | Lvalue Assignment + AdamW | `s.field = e`, `a[i] = e` assignment targets, AdamW stdlib construct |
| [M21](./m21-mps-migration.md) | MPS Migration | `matmul` and reductions → `MPSMatrixMultiplication`/Metal, pending tensors |
| [M22](./m22-data-io-nanogpt.md) | Data I/O + nanoGPT Capstone | File read, in-language char tokenization, batching, nanoGPT capstone |

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| V3 capstone | Char GPT on tiny Shakespeare via real file I/O | Faithful nanoGPT demo (train + sample). Accepts a minimal I/O surface to make the capstone recognizable. See ADR-0018. |
| MPS migration scope | matmul + reductions (M21) | Eager-CPU matmul makes a transformer capstone unrunnably slow. Migrating matmul to `MPSMatrixMultiplication` and reductions to Metal unlocks reasonable wall-clock. See ADR-0017. |
| Optimizer | lvalue assignment + stdlib AdamW | Indexed/field assignment is a general language gap that blocks clean model param management. AdamW as a stdlib construct proves the language composes into a real optimizer. |
| Index tensor dtype | i32 / i64 only | Embedding lookup requires integer index tensors. Full f16/bf16 *compute* dtype generality stays deferred — this is a narrow carve-out for indices only. |
| Broadcasting | NumPy right-aligned | Replaces the `ones41 @ b` bias-broadcast trick used since M8/V2. Eliminates an ergonomic gap that is obvious to any NumPy user. |
| Axis reductions | `keepdim` parameter | Layernorm and softmax require reductions over a specific axis with shape preservation. Whole-tensor `sum` (V1) is insufficient. |
| File I/O | Minimal byte/text read + in-language char tokenization | The capstone needs real training data. I/O scope is fenced to the minimum required: read bytes from a path, iterate chars, build a vocab map. No networking, no binary formats. See ADR-0018. |

## What V3 Does NOT Include

Deferred to post-V3:

- User-definable custom gradient hooks (`custom_grad`)
- Second-order gradients / double-backward
- Gradient checkpointing
- Full non-f32 dtype compute (f16, bf16) — beyond i32/i64 index tensors
- MPS for all stdlib ops (M21 covers matmul and the reductions needed by the transformer)
- SafeTensors / NumPy file I/O (model checkpoint save/load)
- Multi-GPU / distributed training
- Kernel-body control flow (if/else, loops inside `kernel` — needs threadgroup intrinsics)
- `inout` kernel parameters
- GPU RNG intrinsics beyond Philox `randn`
- Growable `Vec<T>`
- Generics / `Option<T>`
- `import as` aliasing
- Cross-module struct/enum types
- CTMM barrier coalescing optimization (still conservative — correctness over performance)
- General dataflow-liveness RC fallback
