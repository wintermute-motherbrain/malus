# Embedding lookup: O[i, c] = W[indices[i], c]
# Weight shape: [vocab, embed_dim]
# indices shape: [seq_len]  (Tensor<i32>)
# Output shape: [seq_len, embed_dim]
#
# One thread per output element (seq_len * embed_dim total).
# No % operator: c = flat - (flat/embed_dim)*embed_dim
#
# Uniforms: embed_dim: i32
# Grid: [seq_len * embed_dim, 1, 1], tg: [1, 1, 1]
kernel __embedding_kernel(weight: Tensor<f32>, indices: Tensor<i32>, embed_dim: i32) -> Tensor<f32>:
    let flat = thread_id()
    let n = flat / embed_dim
    let c = flat - (flat / embed_dim) * embed_dim
    let idx = indices[n]
    out[flat] = weight[idx * embed_dim + c]

fn __embedding_fwd(weight: Tensor<f32>, indices: Tensor<i32>) -> Tensor<f32>:
    let vocab = weight.shape[0]
    let embed_dim = weight.shape[1]
    let seq_len = indices.len
    let n_threads = seq_len * embed_dim
    return __embedding_kernel[grid=[n_threads, 1, 1], tg=[1, 1, 1], out=[seq_len, embed_dim, 0]](weight, indices, embed_dim)
