# General-axis layer normalisation, no affine transform.
# Returns (normed: Tensor<f32>, variance: Tensor<f32>).
# variance is the population variance keepdim at the reduced axis (for backward VJP).
# The VJP in tape.rs computes inv_std = 1/sqrt(var+eps) itself; do NOT return inv_std here.
#
# No % operator: r = n - (n/d)*d
# Grid: [outer*inner, 1, 1], tg: [axis_size, 1, 1]
kernel __layernorm_kernel(x: Tensor<f32>, axis_size: i32, inner_u: i32, eps: f32) -> Tensor<f32>:
    let tg = threadgroup_id()
    let col = thread_in_threadgroup()
    let o = tg / inner_u
    let iv = tg - (tg / inner_u) * inner_u
    let flat = o * axis_size * inner_u + col * inner_u + iv
    let shared scratch: Array<f32, 1024>
    scratch[col] = x[flat]
    barrier()
    let mut s = 0.0
    let mut n_f = 0.0
    for k in range(0, axis_size):
        s = s + scratch[k]
        n_f = n_f + 1.0
    let mean = s / n_f
    barrier()
    let mut v = 0.0
    for k in range(0, axis_size):
        let d = scratch[k] - mean
        v = v + d * d
    let var_val = v / n_f
    let inv_std = rsqrt(var_val + eps)
    barrier()
    out[flat] = (scratch[col] - mean) * inv_std

# Variance kernel: same loop structure, writes population variance per group (keepdim).
# Output shape same as x — variance broadcast across axis dimension for the VJP.
kernel __layernorm_var_kernel(x: Tensor<f32>, axis_size: i32, inner_u: i32, eps: f32) -> Tensor<f32>:
    let tg = threadgroup_id()
    let col = thread_in_threadgroup()
    let o = tg / inner_u
    let iv = tg - (tg / inner_u) * inner_u
    let flat = o * axis_size * inner_u + col * inner_u + iv
    let shared scratch: Array<f32, 1024>
    scratch[col] = x[flat]
    barrier()
    let mut s = 0.0
    let mut n_f = 0.0
    for k in range(0, axis_size):
        s = s + scratch[k]
        n_f = n_f + 1.0
    let mean = s / n_f
    barrier()
    let mut v = 0.0
    for k in range(0, axis_size):
        let d = scratch[k] - mean
        v = v + d * d
    let var_val = v / n_f
    barrier()
    out[flat] = var_val

fn __layernorm_fwd(x: Tensor<f32>, axis: i64) -> (Tensor<f32>, Tensor<f32>):
    let ndim = x.ndim
    let axis_sz = x.shape[axis]
    let mut outer = 1
    for k in range(0, axis):
        outer = outer * x.shape[k]
    let mut inner = 1
    for k in range(axis + 1, ndim):
        inner = inner * x.shape[k]
    let ngroups = outer * inner
    let eps = 0.00001
    let normed = __layernorm_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1]](x, axis_sz, inner, eps)
    let variance = __layernorm_var_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1]](x, axis_sz, inner, eps)
    return (normed, variance)
