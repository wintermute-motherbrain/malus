# Tensor shapes are runtime-only in V1; no static shape checking

`TensorBuffer` carries `shape: Vec<usize>` at runtime (added in M8), but the type
system remains dtype-only: `ResolvedTy::Tensor { dtype }` has no rank or shape
component. Shape errors (matmul inner-dimension mismatch, non-2D transpose) are caught
at runtime with clear panic messages, consistent with ADR-0006 (panic-only error model)
and the "dynamic shape" Tensor definition in CONTEXT.md.

Adding static shape checking was ruled out for V1 because it is a cross-cutting change:
every tensor-producing operation would need an output-shape inference rule, and
`zeros`/`ones` take runtime `i64` dim args that cannot be statically sized regardless.
The most common tensor-construction path defeats static shapes from day one. Static
shape checking is an additive post-V1 refinement; punting now costs nothing structurally
later.
