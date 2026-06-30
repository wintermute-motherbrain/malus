# Layer normalisation over rows of a 2-D tensor.
# (x - mean) / sqrt(var + eps), no affine transform.
# One threadgroup per row; threads_per_threadgroup == cols.
# inv_cols = 1.0/cols pre-computed by the host (avoids int/float division in kernel).
#
# Uniforms: cols: i32, inv_cols: f32, eps: f32
# Grid:     [rows, 1, 1] threadgroups, each [cols, 1, 1] threads.

kernel layernorm(input: Tensor<f32>, cols: i32, inv_cols: f32, eps: f32) -> Tensor<f32>:
    let row = threadgroup_id()
    let col = thread_in_threadgroup()
    let start = row * cols

    # Load row into shared memory.
    let shared scratch: Array<f32, 1024>
    scratch[col] = input[start + col]
    barrier()

    # Compute mean.
    let mut s = 0.0
    for i in range(0, cols):
        s = s + scratch[i]
    let mean = s * inv_cols
    barrier()

    # Compute variance.
    let mut v = 0.0
    for j in range(0, cols):
        let d = scratch[j] - mean
        v = v + d * d
    let inv_std = rsqrt(v * inv_cols + eps)
    barrier()

    # Normalise and write.
    out[start + col] = (scratch[col] - mean) * inv_std
