# Softmax over rows of a 2-D tensor using threadgroup shared memory.
# One threadgroup per row; threads_per_threadgroup == cols.
# Flat indexing: input[row * cols + col], output via implicit `out`.
#
# Uniforms: cols: i32
# Grid:     [rows, 1, 1] threadgroups, each [cols, 1, 1] threads.

kernel softmax(input: Tensor<f32>, cols: i32) -> Tensor<f32>:
    let row = threadgroup_id()
    let col = thread_in_threadgroup()
    let start = row * cols

    # Load row into shared memory.
    let shared scratch: Array<f32, 1024>
    scratch[col] = input[start + col]
    barrier()

    # Find the row max (each thread independently reduces all elements).
    let mut m = scratch[0]
    for i in range(1, cols):
        m = fmax(m, scratch[i])
    barrier()

    # Subtract max and exponentiate.
    scratch[col] = exp(scratch[col] - m)
    barrier()

    # Sum the exponentiated values.
    let mut s = 0.0
    for j in range(0, cols):
        s = s + scratch[j]
    barrier()

    # Write normalised output.
    out[start + col] = scratch[col] / s
