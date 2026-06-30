# General-axis softmax using shared memory.
# Decomposes tensor into outer/axis/inner slabs:
#   flat_idx = outer_idx * axis_size * inner + axis_idx * inner + inner_idx
# One threadgroup per (outer, inner) pair; tg_size = axis_size threads.
# No % operator: r = n - (n/d)*d
#
# Uniforms: axis_size: i32, inner_u: i32
# Grid: [outer*inner, 1, 1], tg: [axis_size, 1, 1]
kernel __softmax_kernel(x: Tensor<f32>, axis_size: i32, inner_u: i32) -> Tensor<f32>:
    let tg = threadgroup_id()
    let col = thread_in_threadgroup()
    let o = tg / inner_u
    let iv = tg - (tg / inner_u) * inner_u
    let flat = o * axis_size * inner_u + col * inner_u + iv
    let shared scratch: Array<f32, 1024>
    scratch[col] = x[flat]
    barrier()
    let mut m = scratch[0]
    for k in range(1, axis_size):
        m = fmax(m, scratch[k])
    barrier()
    scratch[col] = exp(scratch[col] - m)
    barrier()
    let mut s = 0.0
    for k in range(0, axis_size):
        s = s + scratch[k]
    barrier()
    out[flat] = scratch[col] / s

fn __softmax_fwd(x: Tensor<f32>, axis: i64) -> Tensor<f32>:
    let ndim = x.ndim
    let axis_sz = x.shape[axis]
    let mut outer = 1
    for k in range(0, axis):
        outer = outer * x.shape[k]
    let mut inner = 1
    for k in range(axis + 1, ndim):
        inner = inner * x.shape[k]
    let ngroups = outer * inner
    return __softmax_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1]](x, axis_sz, inner)
