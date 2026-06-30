# Backward for the general transpose(x) / permute(x, p0, p1, p2) builtins
# (OpTag::Transpose).  dx = permute(dout, inverse_perm).  inverse_perm is
# computed in Rust (tape.rs) from the forward perm recorded on the tape —
# pure scalar index arithmetic over a length-<=3 list, not tensor compute,
# so it stays orchestration (ADR-0031).
#
# rank==2's only meaningful perm is the full swap [1,0], which is its own
# inverse, so transpose(dout) (0-arg) covers it without needing p0/p1.

fn __permute_bwd_2d(dout: Tensor<f32>) -> Tensor<f32>:
    return transpose(dout)

fn __permute_bwd_3d(dout: Tensor<f32>, inv0: i64, inv1: i64, inv2: i64) -> Tensor<f32>:
    return permute(dout, inv0, inv1, inv2)
