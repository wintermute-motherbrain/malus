# 2-D transpose and 3-D permute GPU kernels.
# Both use strided-gather: each thread computes one output element,
# maps its N-D output index back to the input via the permutation.
#
# x.strides[k] in the kernel body → (int)x_meta.strides[k].
# For contiguous tensors these are row-major product strides.
# No % operator: subtract-trick (r = n - (n/d)*d).

# --- 2-D transpose: [rows, cols] → [cols, rows] ---
# One thread per output element.
# Output flat: flat = r_out*n_rows + c_out  where n_rows=in_cols, n_cols_in=in_rows
# x[c_out, r_out] via x.strides[0]=cols_in.
kernel __transpose_2d_kernel(x: Tensor<f32>) -> Tensor<f32>:
    let flat = thread_id()
    let n_rows = x.shape[0]
    let n_cols = x.strides[0]
    let r_out = flat / n_rows
    let c_out = flat - r_out * n_rows
    out[flat] = x[c_out * n_cols + r_out]

fn __transpose_2d_fwd(x: Tensor<f32>) -> Tensor<f32>:
    let n_rows = x.shape[0]
    let n_cols = x.shape[1]
    let n = n_rows * n_cols
    return __transpose_2d_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[n_cols, n_rows, 0]](x)

# --- 3-D permute: perm=[p0,p1,p2], output[i0,i1,i2] = x[in0,in1,in2] ---
# os1, os2 = output shape dims [1] and [2] (for decomposition).
# Uniforms: p0, p1, p2 (axis indices), os1, os2 (output shape dims 1,2).
# x.strides[k] supplies row-major strides for the input gather.
kernel __permute_3d_kernel(x: Tensor<f32>, p0: i32, p1: i32, p2: i32, os1: i32, os2: i32) -> Tensor<f32>:
    let flat = thread_id()
    let stride0 = os1 * os2
    let i0 = flat / stride0
    let rem0 = flat - i0 * stride0
    let i1 = rem0 / os2
    let i2 = rem0 - i1 * os2
    let mut in0 = 0
    let mut in1 = 0
    let mut in2 = 0
    if p0 == 0:
        in0 = i0
    if p0 == 1:
        in1 = i0
    if p0 == 2:
        in2 = i0
    if p1 == 0:
        in0 = i1
    if p1 == 1:
        in1 = i1
    if p1 == 2:
        in2 = i1
    if p2 == 0:
        in0 = i2
    if p2 == 1:
        in1 = i2
    if p2 == 2:
        in2 = i2
    let flat_in = in0 * x.strides[0] + in1 * x.strides[1] + in2 * x.strides[2]
    out[flat] = x[flat_in]

fn __permute_3d_fwd(x: Tensor<f32>, p0: i64, p1: i64, p2: i64) -> Tensor<f32>:
    let os0 = x.shape[p0]
    let os1 = x.shape[p1]
    let os2 = x.shape[p2]
    let n = os0 * os1 * os2
    return __permute_3d_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[os0, os1, os2]](x, p0, p1, p2, os1, os2)
