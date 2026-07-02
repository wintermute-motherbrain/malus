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

# --- N-D permute (rank 1..8): output[i0..i_{r-1}] = x[perm-gathered] ---
# One thread per output element, tg=[1,1,1]: every thread is its own
# threadgroup, so the `let shared pbuf` scratch is thread-private — no
# barrier needed. Callers pass the FULL normalized perm in p0..p7 (entries
# beyond x.ndim are ignored); validation happens in Rust normalize_perm
# before dispatch.
#
# Decomposition peels output dims from the last: out.shape[dd] ==
# x.shape[perm[dd]], and the gather stride for output dim dd is
# x.strides[perm[dd]]. No % operator: subtract-trick.
kernel __permute_nd_kernel(x: Tensor<f32>, p0: i32, p1: i32, p2: i32, p3: i32, p4: i32, p5: i32, p6: i32, p7: i32) -> Tensor<f32>:
    let flat = thread_id()
    let ndim = x.ndim
    let shared pbuf: Array<i32, 8>
    pbuf[0] = p0
    pbuf[1] = p1
    pbuf[2] = p2
    pbuf[3] = p3
    pbuf[4] = p4
    pbuf[5] = p5
    pbuf[6] = p6
    pbuf[7] = p7
    let mut rem = flat
    let mut in_flat = 0
    for d in range(0, ndim):
        let dd = ndim - 1 - d
        let sz = x.shape[pbuf[dd]]
        let q = rem / sz
        let i = rem - q * sz
        rem = q
        in_flat = in_flat + i * x.strides[pbuf[dd]]
    out[flat] = x[in_flat]

# Host fn: rank is runtime, but the out= literal's length must be static, so
# branch per rank (same pattern as __reduce_sum_fwd). Only the first `ndim`
# perm entries are read — x.shape[p_k] with k < ndim is in-range because the
# Rust caller (tensor_permute / the tape's Transpose arm) normalizes and
# validates the perm first.
fn __permute_nd_fwd(x: Tensor<f32>, p0: i64, p1: i64, p2: i64, p3: i64, p4: i64, p5: i64, p6: i64, p7: i64) -> Tensor<f32>:
    let ndim = x.ndim
    let mut n = 1
    for k in range(0, ndim):
        n = n * x.shape[k]
    if ndim == 1:
        return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], 0, 0, 0, 0, 0, 0, 0]](x, p0, p1, p2, p3, p4, p5, p6, p7)
    if ndim == 2:
        return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], x.shape[p1], 0, 0, 0, 0, 0, 0]](x, p0, p1, p2, p3, p4, p5, p6, p7)
    if ndim == 3:
        return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], x.shape[p1], x.shape[p2], 0, 0, 0, 0, 0]](x, p0, p1, p2, p3, p4, p5, p6, p7)
    if ndim == 4:
        return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], x.shape[p1], x.shape[p2], x.shape[p3], 0, 0, 0, 0]](x, p0, p1, p2, p3, p4, p5, p6, p7)
    if ndim == 5:
        return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], x.shape[p1], x.shape[p2], x.shape[p3], x.shape[p4], 0, 0, 0]](x, p0, p1, p2, p3, p4, p5, p6, p7)
    if ndim == 6:
        return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], x.shape[p1], x.shape[p2], x.shape[p3], x.shape[p4], x.shape[p5], 0, 0]](x, p0, p1, p2, p3, p4, p5, p6, p7)
    if ndim == 7:
        return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], x.shape[p1], x.shape[p2], x.shape[p3], x.shape[p4], x.shape[p5], x.shape[p6], 0]](x, p0, p1, p2, p3, p4, p5, p6, p7)
    return __permute_nd_kernel[grid=[n, 1, 1], tg=[1, 1, 1], out=[x.shape[p0], x.shape[p1], x.shape[p2], x.shape[p3], x.shape[p4], x.shape[p5], x.shape[p6], x.shape[p7]]](x, p0, p1, p2, p3, p4, p5, p6, p7)
