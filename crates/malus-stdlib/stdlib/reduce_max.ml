# General-axis reduction: max.
# Thread 0 computes the maximum over shared scratch.
# fmax(a, b) is available in MSL and in Malus kernel bodies.
kernel __reduce_max_kernel(x: Tensor<f32>, axis_sz: i32, inner_u: i32) -> Tensor<f32>:
    let tg = threadgroup_id()
    let col = thread_in_threadgroup()
    let o = tg / inner_u
    let iv = tg - (tg / inner_u) * inner_u
    let shared scratch: Array<f32, 1024>
    scratch[col] = x[o * axis_sz * inner_u + col * inner_u + iv]
    barrier()
    if col == 0:
        let mut m = scratch[0]
        for k in range(1, axis_sz):
            m = fmax(m, scratch[k])
        out[tg] = m

fn __reduce_max_fwd(x: Tensor<f32>, axis: i64, keepdim: i64) -> Tensor<f32>:
    let ndim = x.ndim
    let axis_sz = x.shape[axis]
    let mut outer = 1
    for k in range(0, axis):
        outer = outer * x.shape[k]
    let mut inner = 1
    for k in range(axis + 1, ndim):
        inner = inner * x.shape[k]
    let ngroups = outer * inner
    if ndim == 1:
        return __reduce_max_kernel[grid=[1, 1, 1], tg=[axis_sz, 1, 1], out=[1, 0, 0]](x, axis_sz, inner)
    else:
        if ndim == 2:
            if keepdim == 0:
                if axis == 0:
                    return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[1], 0, 0]](x, axis_sz, inner)
                else:
                    return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[0], 0, 0]](x, axis_sz, inner)
            else:
                if axis == 0:
                    return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[1, x.shape[1], 0]](x, axis_sz, inner)
                else:
                    return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[0], 1, 0]](x, axis_sz, inner)
        else:
            if keepdim == 0:
                if axis == 0:
                    return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[1], x.shape[2], 0]](x, axis_sz, inner)
                else:
                    if axis == 1:
                        return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[0], x.shape[2], 0]](x, axis_sz, inner)
                    else:
                        return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[0], x.shape[1], 0]](x, axis_sz, inner)
            else:
                if axis == 0:
                    return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[1, x.shape[1], x.shape[2]]](x, axis_sz, inner)
                else:
                    if axis == 1:
                        return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[0], 1, x.shape[2]]](x, axis_sz, inner)
                    else:
                        return __reduce_max_kernel[grid=[ngroups, 1, 1], tg=[axis_sz, 1, 1], out=[x.shape[0], x.shape[1], 1]](x, axis_sz, inner)
