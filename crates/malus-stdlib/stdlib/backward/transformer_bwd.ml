# Backward kernels for the M18/M19 transformer stdlib: softmax, layernorm,
# gelu, cross_entropy, embedding.
# No % operator: r = n - (n/d)*d.

# 1/sqrt(v + eps), elementwise.  eps must match __layernorm_fwd's eps exactly.
kernel __rsqrt_eps_kernel(v: Tensor<f32>, eps: f32) -> Tensor<f32>:
    let i = thread_id()
    out[i] = rsqrt(v[i] + eps)

fn __rsqrt_eps_fwd(v: Tensor<f32>, eps: f32) -> Tensor<f32>:
    let n = v.len
    return __rsqrt_eps_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](v, eps)

# s = softmax(x, axis)  ->  dx = s * (dout - sum(dout*s, axis, keepdim=1))
fn __softmax_bwd(s: Tensor<f32>, dout: Tensor<f32>, axis: i64) -> Tensor<f32>:
    let prod = __broadcast_mul_fwd(dout, s)
    let sum_ds = __reduce_sum_fwd(prod, axis, 1)
    let sum_bc = __expand_axis_fwd(sum_ds, s, axis)
    let diff = __broadcast_sub_fwd(dout, sum_bc)
    return __broadcast_mul_fwd(s, diff)

# y = (x - mean) / sigma,  sigma = sqrt(var + eps)
# dx = (1/sigma) * (dy - mean(dy, axis, k) - y * mean(dy*y, axis, k))
fn __layernorm_bwd(y: Tensor<f32>, var_h: Tensor<f32>, dout: Tensor<f32>, axis: i64) -> Tensor<f32>:
    let eps = 0.00001
    let inv_sigma_h = __rsqrt_eps_fwd(var_h, eps)
    let inv_sigma_bc = __expand_axis_fwd(inv_sigma_h, y, axis)
    let dy_mean_h = __reduce_mean_fwd(dout, axis, 1)
    let dy_mean_bc = __expand_axis_fwd(dy_mean_h, y, axis)
    let dy_y = __broadcast_mul_fwd(dout, y)
    let dy_y_mean_h = __reduce_mean_fwd(dy_y, axis, 1)
    let dy_y_mean_bc = __expand_axis_fwd(dy_y_mean_h, y, axis)
    let y_term = __broadcast_mul_fwd(y, dy_y_mean_bc)
    let tmp = __broadcast_sub_fwd(dout, dy_mean_bc)
    let numer = __broadcast_sub_fwd(tmp, y_term)
    return __broadcast_mul_fwd(numer, inv_sigma_bc)

# y = 0.5*x*(1 + tanh(g)),  g = c0*(x + c1*x^3)
# dy/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*g',  t = tanh(g),  g' = c0*(1 + 3*c1*x^2)
kernel __gelu_bwd_kernel(x: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    let xv = x[i]
    let x3 = xv * xv * xv
    let g = 0.7978845608028654 * (xv + 0.044715 * x3)
    let t = tanh(g)
    let gp = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * xv * xv)
    let deriv = 0.5 * (1.0 + t) + 0.5 * xv * (1.0 - t * t) * gp
    out[i] = dout[i] * deriv

fn __gelu_bwd(x: Tensor<f32>, dout: Tensor<f32>) -> Tensor<f32>:
    let n = x.len
    return __gelu_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](x, dout)

# L = -mean(log(s[i, t[i]])),  s = softmax(logits, axis=1)
# d_logits[i,j] = dout[0]/N * (s[i,j] - 1{j == t[i]})
# Uniforms: vocab: i32, scale: f32 (= dout[0] / n_tokens)
kernel __cross_entropy_bwd_kernel(probs: Tensor<f32>, targets: Tensor<i32>, vocab: i32, scale: f32) -> Tensor<f32>:
    let flat = thread_id()
    let i = flat / vocab
    let j = flat - i * vocab
    let tgt = targets[i]
    let mut g = probs[flat]
    if j == tgt:
        g = g - 1.0
    out[flat] = g * scale

fn __cross_entropy_bwd(probs: Tensor<f32>, targets: Tensor<i32>, dout: Tensor<f32>) -> Tensor<f32>:
    let n_tokens = probs.shape[0]
    let vocab = probs.shape[1]
    let n = n_tokens * vocab
    let mut n_f = 0.0
    for k in range(0, n_tokens):
        n_f = n_f + 1.0
    let scale = dout[0] / n_f
    return __cross_entropy_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](probs, targets, vocab, scale)

# dweight[v, c] = sum over t where indices[t] == v of dout[t, c]
# Per-row gather (D3): one thread per (vocab_row, dim) sums its own
# contributions, avoiding the write collisions an indices-major scatter
# would hit.  No atomics needed; deterministic and exact.
# Atomics for large-vocab efficiency are a documented future deferral
# (see CLAUDE.md Known Limitations).
kernel __embedding_bwd_kernel(dout: Tensor<f32>, indices: Tensor<i32>, seq_len: i32, embed_dim: i32) -> Tensor<f32>:
    let flat = thread_id()
    let v = flat / embed_dim
    let c = flat - v * embed_dim
    let mut acc = 0.0
    for t in range(0, seq_len):
        if indices[t] == v:
            acc = acc + dout[t * embed_dim + c]
    out[flat] = acc

fn __embedding_bwd(weight: Tensor<f32>, indices: Tensor<i32>, dout: Tensor<f32>) -> Tensor<f32>:
    let vocab = weight.shape[0]
    let embed_dim = weight.shape[1]
    let seq_len = indices.len
    let n = vocab * embed_dim
    return __embedding_bwd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[vocab, embed_dim, 0]](dout, indices, seq_len, embed_dim)
