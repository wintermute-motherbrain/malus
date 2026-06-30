# General NumPy-style broadcast elementwise kernels.
# Convention: A is the tensor with the output shape (a.ndim >= b.ndim).
# Each thread computes one output element.
# B's flat index is computed by projecting the N-D output index through
# B's TensorMeta with broadcast rules (size-1 dims always index 0).
# No % operator: r = n - (n/d)*d  (avoid % in Malus; OK to use in MSL via strides)
#
# Output shape = A's shape (kernel_dispatch_v2 out_ndim=0 inherits first input's shape).
# Grid: [out.len, 1, 1], tg: [1, 1, 1]
kernel __broadcast_add_kernel(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let flat = thread_id()
    let ndim = a.ndim
    let b_ndim = b.ndim
    let b_off = ndim - b_ndim
    let mut b_flat = 0
    let mut rem = flat
    for d in range(0, ndim):
        let stride = a.strides[d]
        let idx_d = rem / stride
        rem = rem - idx_d * stride
        let db = d - b_off
        if db >= 0:
            let b_dim = b.shape[db]
            if b_dim > 1:
                b_flat = b_flat + idx_d * b.strides[db]
    out[flat] = a[flat] + b[b_flat]

kernel __broadcast_sub_kernel(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let flat = thread_id()
    let ndim = a.ndim
    let b_ndim = b.ndim
    let b_off = ndim - b_ndim
    let mut b_flat = 0
    let mut rem = flat
    for d in range(0, ndim):
        let stride = a.strides[d]
        let idx_d = rem / stride
        rem = rem - idx_d * stride
        let db = d - b_off
        if db >= 0:
            let b_dim = b.shape[db]
            if b_dim > 1:
                b_flat = b_flat + idx_d * b.strides[db]
    out[flat] = a[flat] - b[b_flat]

kernel __broadcast_mul_kernel(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let flat = thread_id()
    let ndim = a.ndim
    let b_ndim = b.ndim
    let b_off = ndim - b_ndim
    let mut b_flat = 0
    let mut rem = flat
    for d in range(0, ndim):
        let stride = a.strides[d]
        let idx_d = rem / stride
        rem = rem - idx_d * stride
        let db = d - b_off
        if db >= 0:
            let b_dim = b.shape[db]
            if b_dim > 1:
                b_flat = b_flat + idx_d * b.strides[db]
    out[flat] = a[flat] * b[b_flat]

kernel __broadcast_div_kernel(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let flat = thread_id()
    let ndim = a.ndim
    let b_ndim = b.ndim
    let b_off = ndim - b_ndim
    let mut b_flat = 0
    let mut rem = flat
    for d in range(0, ndim):
        let stride = a.strides[d]
        let idx_d = rem / stride
        rem = rem - idx_d * stride
        let db = d - b_off
        if db >= 0:
            let b_dim = b.shape[db]
            if b_dim > 1:
                b_flat = b_flat + idx_d * b.strides[db]
    out[flat] = a[flat] / b[b_flat]

fn __broadcast_add_fwd(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    if a.ndim >= b.ndim:
        let n = a.len
        return __broadcast_add_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](a, b)
    else:
        let n = b.len
        return __broadcast_add_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](b, a)

fn __broadcast_sub_fwd(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    if a.ndim >= b.ndim:
        let n = a.len
        return __broadcast_sub_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](a, b)
    else:
        let n = b.len
        return __broadcast_sub_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](b, a)

fn __broadcast_mul_fwd(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    if a.ndim >= b.ndim:
        let n = a.len
        return __broadcast_mul_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](a, b)
    else:
        let n = b.len
        return __broadcast_mul_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](b, a)

fn __broadcast_div_fwd(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    if a.ndim >= b.ndim:
        let n = a.len
        return __broadcast_div_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](a, b)
    else:
        let n = b.len
        return __broadcast_div_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](b, a)
