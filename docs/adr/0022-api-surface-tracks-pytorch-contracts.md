# ADR-0022 — API surface tracks PyTorch's actual contracts

**Status**: Accepted (M17)

## Context

malus is heading toward a nanoGPT capstone. Its users will transfer intuitions from PyTorch. Every API decision about tensor ops creates either a smooth path ("it works like PyTorch") or a hidden trap ("it looks like PyTorch but diverges in ways I'll discover at runtime").

During M17 planning, the question of `transpose` and `reshape` naming surfaced a deeper issue: a name that collapses two distinct PyTorch functions (e.g. one `transpose` overloaded to also do full-axis reorder) is not merely inelegant — it burns the name on a broader contract that was never intended, and a future addition to match PyTorch's actual contract would be breaking.

## Decision

**API surface tracks PyTorch's actual contracts.** Concretely:

1. Do not collapse two distinct PyTorch functions onto one malus name.
2. Do not burn a PyTorch name on a mere synonym of something PyTorch calls differently.
3. Capability deferral must be **additive**, never **breaking**: ship a strict subset of PyTorch's contract now; the remaining subset is added later without changing existing call sites.
4. When malus must reserve a name for a future true contract (e.g. `view` for strided non-contiguous views), leave it undefined rather than repurposing it.

## Consequences

- `transpose(t)` and `transpose(t, i, j)` match `torch.transpose` — two-axis swap only. Full reorder requires `permute(t, p0..prank)`, matching `torch.permute`. These are separate builtins sharing one runtime engine.
- `reshape(t, d0..dn)` matches `torch.reshape` (may copy). In M17 it is zero-copy because all M17 tensors are contiguous, which is strictly a subset of PyTorch's contract.
- `view` is reserved — not a synonym for `reshape`. When malus eventually adds non-contiguous strided views, `view` carries the PyTorch "contiguous or error" contract.
- Future ops (softmax, layernorm, cross_entropy, gather, etc.) should be named to match their PyTorch counterparts without semantic overloading.

## Alternatives rejected

- **Collapse transpose + permute under one name**: saves one builtin registration today, but the overloaded `transpose` would diverge from `torch.transpose` and require a breaking deprecation to add full-rank permute later.
- **Use `view` as an alias for `reshape`**: erases the distinction between `view` (contiguous-or-error) and `reshape` (always succeeds via copy). Future users who write `.view()` expecting contiguous-or-error semantics would get silent wrong behavior.
