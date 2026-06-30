# Backward kernels for unary elementwise ops, plus two general-purpose
# elementwise primitives (scale-by-constant, same-shape equality) reused by
# several other backward files.  All operate one thread per element;
# input/output/dout always share the same shape (no broadcast).

kernel __scale_kernel(x: Tensor<f32>, c: f32) -> Tensor<f32>:
    let i = thread_id()
    out[i] = x[i] * c

fn __scale_fwd(x: Tensor<f32>, c: f32) -> Tensor<f32>:
    let n = x.len
    return __scale_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](x, c)

# Fill a tensor shaped like `x` with the constant `c`.  Used by __sum_bwd
# (full-tensor sum) to broadcast a single gradient scalar back to x's shape.
kernel __fill_like_kernel(x: Tensor<f32>, c: f32) -> Tensor<f32>:
    let i = thread_id()
    out[i] = c

fn __fill_like_fwd(x: Tensor<f32>, c: f32) -> Tensor<f32>:
    let n = x.len
    return __fill_like_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](x, c)

# Same-shape elementwise equality mask (1.0 / 0.0).  Used by reduce_max's VJP.
kernel __eq_kernel(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    if a[i] == b[i]:
        out[i] = 1.0
    else:
        out[i] = 0.0

fn __eq_fwd(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let n = a.len
    return __eq_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](a, b)

# sigmoid: s = sigma(x)  ->  dx = dout * s * (1 - s)
kernel __sigmoid_bwd_kernel(s: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    let sv = s[i]
    out[i] = dout[i] * sv * (1.0 - sv)

fn __sigmoid_bwd(s: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let n = s.len
    return __sigmoid_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](s, dout)

# relu: r = max(x, 0)  ->  dx = dout * (x > 0)
kernel __relu_bwd_kernel(x: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    if x[i] > 0.0:
        out[i] = dout[i]
    else:
        out[i] = 0.0

fn __relu_bwd(x: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let n = x.len
    return __relu_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](x, dout)

# tanh: t = tanh(x)  ->  dx = dout * (1 - t*t)
kernel __tanh_bwd_kernel(t: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    let tv = t[i]
    out[i] = dout[i] * (1.0 - tv * tv)

fn __tanh_bwd(t: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let n = t.len
    return __tanh_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](t, dout)

# sqrt: s = sqrt(x)  ->  dx = dout / (2*s)
kernel __sqrt_bwd_kernel(s: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    out[i] = dout[i] / (2.0 * s[i])

fn __sqrt_bwd(s: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let n = s.len
    return __sqrt_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](s, dout)

# abs: a = |x|  ->  dx = dout * sign(x)
kernel __abs_bwd_kernel(x: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    let xv = x[i]
    if xv > 0.0:
        out[i] = dout[i]
    else:
        if xv < 0.0:
            out[i] = -dout[i]
        else:
            out[i] = 0.0

fn __abs_bwd(x: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let n = x.len
    return __abs_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](x, dout)
