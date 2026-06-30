# GELU elementwise kernel (tanh approximation).
# gelu(x) = 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x*x*x)))
# One thread per element; no shared memory.
# Grid: [N, 1, 1], tg: [1, 1, 1]
kernel __gelu_kernel(x: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    let xv = x[i]
    let x3 = xv * xv * xv
    let inner = 0.7978845608028654 * (xv + 0.044715 * x3)
    out[i] = 0.5 * xv * (1.0 + tanh(inner))

fn __gelu_fwd(x: Tensor<f32>) -> Tensor<f32>:
    let n = x.len
    return __gelu_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](x)
