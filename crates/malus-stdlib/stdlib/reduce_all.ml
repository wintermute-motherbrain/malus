# Full-tensor reduction: sum of every element, regardless of rank.
# Replaces the legacy CPU `sum(t)` builtin (M26 gate hardening, ADR-0031/0032).
# Reuses __reduce_sum_kernel by treating the whole flat buffer as one axis
# (outer=1, inner=1) — works for any rank since the kernel only cares about
# flat index arithmetic, not the tensor's declared shape.
# Capped at 1024 elements (the shared-scratch threadgroup limit shared by
# every other stdlib reduce kernel) — not on the nanoGPT hot path; see
# CLAUDE.md Known Limitations for the deferred grid-stride lift.
fn __reduce_all_sum_fwd(x: Tensor<f32>) -> Tensor<f32>:
    let n = x.len
    let inner = 1
    return __reduce_sum_kernel[grid=[1, 1, 1], tg=[n, 1, 1], out=[1, 0, 0]](x, n, inner)
