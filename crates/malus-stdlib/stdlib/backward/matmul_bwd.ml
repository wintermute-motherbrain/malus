# Backward for matmul.  matmul itself stays the MPS vendor primitive
# (ADR-0028); only the transpose/permute + reduce around it are kernels.
# Covers all three forward rank cases: 2x2, 3x3 (batched), and 3x2
# (broadcast: A=(B,M,K), B=(K,N)).
#
# C = A @ B
#   2x2:        dA = dC @ Bt            dB = At @ dC
#   3x3 batch:  dA = dC @ permute(B)    dB = permute(A) @ dC
#   3x2 broad:  dA = dC @ Bt            dB = reduce_sum(permute(A) @ dC, axis=0)
#
# When B is 2-D, `dC @ transpose(B)` works directly via the existing 3D(x)2D
# matmul broadcast (M17/M18) whether A — and therefore dC — is 2-D or 3-D,
# so the 2x2 and 3x2 cases for dA collapse into one branch.

fn __matmul_bwd_a(a: Tensor<f32>, b: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    if b.ndim == 3:
        return dout @ permute(b, 0, 2, 1)
    else:
        return dout @ transpose(b)

fn __matmul_bwd_b(a: Tensor<f32>, b: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    if a.ndim == 2:
        return transpose(a) @ dout
    else:
        if b.ndim == 3:
            return permute(a, 0, 2, 1) @ dout
        else:
            let prod = permute(a, 0, 2, 1) @ dout
            return __reduce_sum_fwd(prod, 0, 0)
