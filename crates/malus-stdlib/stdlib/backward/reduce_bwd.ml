# Backward for axis reductions (sum/mean/max/var) and full-tensor sum.
#
# __expand_axis_kernel is the dual of __reduce_sum_kernel's outer/axis/inner
# decomposition: it replicates a [outer, inner]-flat tensor (the reduced
# axis collapsed, whether the forward call used keepdim or not — a size-1
# axis contributes a stride-0 factor either way, so the flat layout is
# identical) back out to [outer, axis_size, inner].
# No % operator: r = n - (n/d)*d.

kernel __expand_axis_kernel(small: Tensor<f32>, axis_size: i32, inner_u: i32) -> Tensor<f32>:
    let flat = thread_id()
    let stride1 = axis_size * inner_u
    let o = flat / stride1
    let rem = flat - o * stride1
    let a = rem / inner_u
    let iv = rem - a * inner_u
    let small_flat = o * inner_u + iv
    out[flat] = small[small_flat]

# x_template supplies the desired output shape (the original pre-reduction
# tensor saved on the tape); only its .shape/.ndim are read, never its data.
fn __expand_axis_fwd(small: Tensor<f32>, x_template: Tensor<f32>, axis: i64) -> Tensor<f32>:
    let ndim = x_template.ndim
    let axis_sz = x_template.shape[axis]
    let mut outer = 1
    for k in range(0, axis):
        outer = outer * x_template.shape[k]
    let mut inner = 1
    for k in range(axis + 1, ndim):
        inner = inner * x_template.shape[k]
    let n = outer * axis_sz * inner
    if ndim == 1:
        return __expand_axis_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x_template.shape[0], 0, 0]](small, axis_sz, inner)
    else:
        if ndim == 2:
            return __expand_axis_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x_template.shape[0], x_template.shape[1], 0]](small, axis_sz, inner)
        else:
            return __expand_axis_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x_template.shape[0], x_template.shape[1], x_template.shape[2]]](small, axis_sz, inner)

# y = sum(x, axis, keepdim)  ->  dx = expand_axis(dout, x, axis)
fn __reduce_sum_axis_bwd(x: Tensor<f32>, dout: Tensor<f32>, axis: i64) -> Tensor<f32>:
    return __expand_axis_fwd(dout, x, axis)

# y = mean(x, axis, keepdim)  ->  dx = expand_axis(dout / N, x, axis)
fn __reduce_mean_axis_bwd(x: Tensor<f32>, dout: Tensor<f32>, axis: i64) -> Tensor<f32>:
    let n = x.shape[axis]
    let mut n_f = 0.0
    for k in range(0, n):
        n_f = n_f + 1.0
    let scaled = __scale_fwd(dout, 1.0 / n_f)
    return __expand_axis_fwd(scaled, x, axis)

# y = max(x, axis, keepdim)  ->  dx = expand_axis(dout, x, axis) * (x == expand_axis(y, x, axis))
fn __reduce_max_axis_bwd(x: Tensor<f32>, y: Tensor<f32>, dout: Tensor<f32>, axis: i64) -> Tensor<f32>:
    let y_bc = __expand_axis_fwd(y, x, axis)
    let mask = __eq_fwd(x, y_bc)
    let dout_bc = __expand_axis_fwd(dout, x, axis)
    return __broadcast_mul_fwd(dout_bc, mask)

# y = var(x, axis, keepdim) population  ->  dx = expand_axis(dout, x, axis) * 2*(x - mean)/N
fn __reduce_var_axis_bwd(x: Tensor<f32>, dout: Tensor<f32>, axis: i64) -> Tensor<f32>:
    let mean_h = __reduce_mean_fwd(x, axis, 1)
    let mean_bc = __expand_axis_fwd(mean_h, x, axis)
    let x_minus_mean = __broadcast_sub_fwd(x, mean_bc)
    let n = x.shape[axis]
    let mut n_f = 0.0
    for k in range(0, n):
        n_f = n_f + 1.0
    let scaled = __scale_fwd(x_minus_mean, 2.0 / n_f)
    let dout_bc = __expand_axis_fwd(dout, x, axis)
    return __broadcast_mul_fwd(dout_bc, scaled)

# s = sum(x) (full-tensor reduction)  ->  dx = fill_like(x, dout[0])
fn __sum_bwd(x: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let c = dout[0]
    return __fill_like_fwd(x, c)
