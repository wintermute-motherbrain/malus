# Backward for the four NumPy-broadcast binops (Add/Sub/Mul/Div).
# __sum_to_shape_bwd is the general "reduce-back-to-a-smaller-broadcastable-
# shape" primitive (the dual of broadcast): it first collapses any leading
# dims `dout` has beyond `target`'s rank (full removal, NumPy right-align),
# then axis-reduces (keepdim) any dim where target is size-1 but dout isn't.
# Composed entirely from existing forward kernels — no new GPU kernel here.

fn __sum_to_shape_bwd(dout: Tensor<f32>, target: Tensor<f32>) -> Tensor<f32>:
    let extra = dout.ndim - target.ndim
    # cur must never alias a parameter directly (CTMM owns reassigned `let
    # mut` bindings); __scale_fwd(dout, 1.0) materialises a fresh, owned copy.
    let mut cur = __scale_fwd(dout, 1.0)
    for k in range(0, extra):
        cur = __reduce_sum_fwd(cur, 0, 0)
    let t_ndim = target.ndim
    let mut axis = 0
    while axis < t_ndim:
        if target.shape[axis] == 1:
            if cur.shape[axis] > 1:
                cur = __reduce_sum_fwd(cur, axis, 1)
        axis = axis + 1
    return cur

# C = A + B  ->  dA = sum_to_shape(dC, A.shape),  dB = sum_to_shape(dC, B.shape)
fn __add_bwd_a(dout: Tensor<f32>, a: Tensor<f32>) -> Tensor<f32>:
    return __sum_to_shape_bwd(dout, a)

fn __add_bwd_b(dout: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return __sum_to_shape_bwd(dout, b)

# C = A - B  ->  dA = sum_to_shape(dC, A.shape),  dB = -sum_to_shape(dC, B.shape)
fn __sub_bwd_a(dout: Tensor<f32>, a: Tensor<f32>) -> Tensor<f32>:
    return __sum_to_shape_bwd(dout, a)

fn __sub_bwd_b(dout: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let neg_dout = __scale_fwd(dout, -1.0)
    return __sum_to_shape_bwd(neg_dout, b)

# C = A * B  ->  dA = sum_to_shape(dC * broadcast(B), A.shape)
#                dB = sum_to_shape(broadcast(A) * dC, B.shape)
fn __mul_bwd_a(dout: Tensor<f32>, a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let full = __broadcast_mul_fwd(dout, b)
    return __sum_to_shape_bwd(full, a)

fn __mul_bwd_b(dout: Tensor<f32>, a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let full = __broadcast_mul_fwd(dout, a)
    return __sum_to_shape_bwd(full, b)

# C = A / B  ->  dA = sum_to_shape(dC / broadcast(B), A.shape)
#                dB = sum_to_shape(-dC * broadcast(A) / broadcast(B)^2, B.shape)
fn __div_bwd_a(dout: Tensor<f32>, a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let full = __broadcast_div_fwd(dout, b)
    return __sum_to_shape_bwd(full, a)

fn __div_bwd_b(dout: Tensor<f32>, a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let t1 = __broadcast_mul_fwd(dout, a)
    let neg_t1 = __scale_fwd(t1, -1.0)
    let b_sq = __broadcast_mul_fwd(b, b)
    let full = __broadcast_div_fwd(neg_t1, b_sq)
    return __sum_to_shape_bwd(full, b)
