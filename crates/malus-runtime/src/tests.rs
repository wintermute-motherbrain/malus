use std::collections::HashMap;

use crate::{
    Dtype, TensorBuffer, runtime_init,
    tensor_alloc_gpu, tensor_alloc_zeros_gpu, tensor_alloc_ones_gpu,
    tensor_retain, tensor_release, tensor_free, tensor_print, tensor_len,
    tensor_matmul, tensor_transpose, tensor_sum,
    kernel_dispatch, gpu_barrier,
};

#[test]
fn test_dtype_from_tag_drift() {
    assert_eq!(Dtype::from_tag(0),  Dtype::F32);
    assert_eq!(Dtype::from_tag(1),  Dtype::F16);
    assert_eq!(Dtype::from_tag(2),  Dtype::Bf16);
    assert_eq!(Dtype::from_tag(3),  Dtype::I8);
    assert_eq!(Dtype::from_tag(4),  Dtype::I16);
    assert_eq!(Dtype::from_tag(5),  Dtype::I32);
    assert_eq!(Dtype::from_tag(6),  Dtype::I64);
    assert_eq!(Dtype::from_tag(7),  Dtype::U8);
    assert_eq!(Dtype::from_tag(8),  Dtype::U16);
    assert_eq!(Dtype::from_tag(9),  Dtype::U32);
    assert_eq!(Dtype::from_tag(10), Dtype::U64);
}

#[test]
fn test_dtype_to_tag_roundtrip() {
    for tag in 0..=10 {
        assert_eq!(Dtype::from_tag(tag).to_tag(), tag);
    }
}

#[test]
#[should_panic(expected = "unknown dtype")]
fn test_dtype_unknown_tag_panics() {
    Dtype::from_tag(99);
}

fn alloc_f32(data: &[f32]) -> i64 {
    let shape = [data.len()];
    tensor_alloc_gpu(0, shape.as_ptr(), 1, data.as_ptr())
}

fn alloc_2d(data: &[f32], rows: usize, cols: usize) -> i64 {
    assert_eq!(data.len(), rows * cols);
    let shape = [rows, cols];
    tensor_alloc_gpu(0, shape.as_ptr(), 2, data.as_ptr())
}

fn read_f32(handle: i64) -> Vec<f32> {
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let ptr = tb.buffer.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, tb.len).to_vec() }
}

#[test]
fn test_tensor_alloc_roundtrip() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let handle = alloc_f32(&data);
    assert!(handle != 0, "handle should be non-null");

    let tb = unsafe { &*(handle as *const TensorBuffer) };
    assert_eq!(tb.len, 4);
    assert_eq!(tb.shape, &[4]);
    assert_eq!(tb.dtype, Dtype::F32);

    let ptr = tb.buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, tb.len) };
    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);

    tensor_free(handle);
}

#[test]
fn test_tensor_alloc_null_data() {
    let shape = [4usize];
    let handle = tensor_alloc_gpu(0, shape.as_ptr(), 1, std::ptr::null());
    assert!(handle != 0);

    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let ptr = tb.buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, tb.len) };
    assert!(slice.iter().all(|v| *v == 0.0), "buffer should be zeroed");

    tensor_free(handle);
}

#[test]
fn test_tensor_alloc_2d_shape() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let handle = alloc_2d(&data, 2, 3);
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    assert_eq!(tb.len, 6);
    assert_eq!(tb.shape, &[2, 3]);
    tensor_free(handle);
}

#[test]
fn test_tensor_print_no_panic() {
    let handle = alloc_f32(&[1.0f32, 2.0, 3.0, 4.0]);
    tensor_print(handle);
    println!();
    tensor_free(handle);
}

#[test]
fn test_tensor_free_no_crash() {
    let handle = alloc_f32(&[1.0f32, 2.0, 3.0]);
    tensor_free(handle);
}

#[test]
fn test_tensor_len_returns_element_count() {
    let handle = alloc_f32(&[1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(tensor_len(handle), 4);
    tensor_free(handle);
}

#[test]
fn test_gpu_barrier_noop_when_no_work() {
    gpu_barrier();
}

#[test]
fn test_massive_alloc_free() {
    let mut handles = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        handles.push(alloc_f32(&[1.0f32, 2.0, 3.0, 4.0]));
    }
    for h in handles {
        tensor_free(h);
    }
}

// ── M8: zeros / ones ─────────────────────────────────────────────────────────

#[test]
fn test_tensor_alloc_zeros_is_zero() {
    let shape = [2usize, 3];
    let handle = tensor_alloc_zeros_gpu(shape.as_ptr(), 2);
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    assert_eq!(tb.len, 6);
    assert_eq!(tb.shape, &[2, 3]);
    let data = read_f32(handle);
    assert!(data.iter().all(|&v| v == 0.0));
    tensor_free(handle);
}

#[test]
fn test_tensor_alloc_ones_is_one() {
    let shape = [3usize, 4];
    let handle = tensor_alloc_ones_gpu(shape.as_ptr(), 2);
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    assert_eq!(tb.len, 12);
    assert_eq!(tb.shape, &[3, 4]);
    let data = read_f32(handle);
    assert!(data.iter().all(|&v| v == 1.0));
    tensor_free(handle);
}

// ── M8: matmul ───────────────────────────────────────────────────────────────

#[test]
fn test_tensor_matmul_2x3_3x2() {
    // [2,3] @ [3,2] -> [2,2]
    // a = [[1,2,3],[4,5,6]], b = [[1,0],[0,1],[1,0]]
    // out[0,0] = 1*1+2*0+3*1=4, out[0,1]=1*0+2*1+3*0=2
    // out[1,0] = 4*1+5*0+6*1=10, out[1,1]=4*0+5*1+6*0=5
    let a = alloc_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = alloc_2d(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], 3, 2);
    let out = tensor_matmul(a, b);
    let tb = unsafe { &*(out as *const TensorBuffer) };
    assert_eq!(tb.shape, &[2, 2]);
    assert_eq!(read_f32(out), &[4.0, 2.0, 10.0, 5.0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

#[test]
fn test_tensor_matmul_ones() {
    // ones([2,3]) @ ones([3,4]) -> [2,4] of all 3.0
    let shape_23 = [2usize, 3];
    let shape_34 = [3usize, 4];
    let a = tensor_alloc_ones_gpu(shape_23.as_ptr(), 2);
    let b = tensor_alloc_ones_gpu(shape_34.as_ptr(), 2);
    let out = tensor_matmul(a, b);
    let tb = unsafe { &*(out as *const TensorBuffer) };
    assert_eq!(tb.shape, &[2, 4]);
    assert!(read_f32(out).iter().all(|&v| v == 3.0));
    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

// Dim mismatch panics are not testable with #[should_panic] because tensor_matmul is
// extern "C" (panics can't unwind through C ABI). The runtime panics with a clear message
// at runtime; validated manually.

// ── M8: transpose ────────────────────────────────────────────────────────────

#[test]
fn test_tensor_transpose_2x3() {
    // [[1,2,3],[4,5,6]] transposed -> [[1,4],[2,5],[3,6]]
    let h = alloc_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = tensor_transpose(h);
    let tb = unsafe { &*(out as *const TensorBuffer) };
    assert_eq!(tb.shape, &[3, 2]);
    assert_eq!(read_f32(out), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    tensor_free(h);
    tensor_free(out);
}

#[test]
fn test_tensor_transpose_ones_shape() {
    let shape = [3usize, 4];
    let h = tensor_alloc_ones_gpu(shape.as_ptr(), 2);
    let out = tensor_transpose(h);
    let tb = unsafe { &*(out as *const TensorBuffer) };
    assert_eq!(tb.shape, &[4, 3]);
    assert!(read_f32(out).iter().all(|&v| v == 1.0));
    tensor_free(h);
    tensor_free(out);
}

// ── M8: sum ──────────────────────────────────────────────────────────────────

#[test]
fn test_tensor_sum_flat() {
    let h = alloc_f32(&[1.0, 2.0, 3.0, 4.0]);
    let out = tensor_sum(h);
    let tb = unsafe { &*(out as *const TensorBuffer) };
    assert_eq!(tb.shape, &[1]);
    assert_eq!(read_f32(out), &[10.0]);
    tensor_free(h);
    tensor_free(out);
}

#[test]
fn test_tensor_sum_ones_2x4() {
    let shape = [2usize, 4];
    let h = tensor_alloc_ones_gpu(shape.as_ptr(), 2);
    let out = tensor_sum(h);
    assert_eq!(read_f32(out), &[8.0]);
    tensor_free(h);
    tensor_free(out);
}

// ── M5: Real kernel dispatch ──────────────────────────────────────────────────

const ADD_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void malus_kernel_0(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = (a[tid] + b[tid]);
}
"#;

fn init_add_kernel() {
    let mut registry = HashMap::new();
    registry.insert(0u64, ADD_MSL.to_string());
    runtime_init(&registry);
}

#[test]
fn test_msl_compiles_without_error() {
    init_add_kernel();
}

#[test]
fn test_kernel_dispatch_add() {
    init_add_kernel();

    let a = alloc_f32(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = alloc_f32(&[5.0f32, 6.0, 7.0, 8.0]);
    let handles = [a, b];

    let output = kernel_dispatch(0, handles.as_ptr(), 2);
    assert!(output != 0);

    gpu_barrier();

    let tb = unsafe { &*(output as *const TensorBuffer) };
    assert_eq!(tb.len, 4);
    assert_eq!(tb.shape, &[4]);
    assert_eq!(tb.dtype, Dtype::F32);

    let ptr = tb.buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, tb.len) };
    assert_eq!(slice, &[6.0, 8.0, 10.0, 12.0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(output);
}

#[test]
fn test_kernel_dispatch_then_print() {
    init_add_kernel();

    let a = alloc_f32(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = alloc_f32(&[5.0f32, 6.0, 7.0, 8.0]);
    let handles = [a, b];

    let output = kernel_dispatch(0, handles.as_ptr(), 2);
    gpu_barrier();

    tensor_print(output);
    println!();

    tensor_free(a);
    tensor_free(b);
    tensor_free(output);
}

// ── M8: kernel_dispatch preserves 2D shape ───────────────────────────────────

#[test]
fn test_kernel_dispatch_preserves_shape() {
    init_add_kernel();

    let a = alloc_2d(&[1.0f32, 2.0, 3.0, 4.0], 2, 2);
    let b = alloc_2d(&[1.0f32, 1.0, 1.0, 1.0], 2, 2);
    let handles = [a, b];

    let output = kernel_dispatch(0, handles.as_ptr(), 2);
    gpu_barrier();

    let tb = unsafe { &*(output as *const TensorBuffer) };
    assert_eq!(tb.shape, &[2, 2], "kernel_dispatch output must preserve input shape");

    tensor_free(a);
    tensor_free(b);
    tensor_free(output);
}

// ── M9: retain / release ──────────────────────────────────────────────────────

#[test]
fn test_tensor_retain_keeps_alive() {
    let h = alloc_f32(&[1.0, 2.0, 3.0]);
    // ref_count = 1 after alloc; retain bumps to 2.
    tensor_retain(h);
    // First release: ref_count → 1.  Must NOT free (buffer still alive).
    tensor_release(h);
    // Verify the buffer is still readable.
    let data = read_f32(h);
    assert_eq!(data, &[1.0f32, 2.0, 3.0], "buffer must be readable after retain+release");
    // Second release: ref_count → 0.  Frees.
    tensor_release(h);
}

#[test]
fn test_tensor_free_still_works() {
    // tensor_free now delegates to tensor_release.  Verify it still frees a fresh tensor.
    let h = alloc_f32(&[9.0, 8.0, 7.0]);
    tensor_free(h); // ref_count 1 → 0 → freed; must not crash
}

#[test]
fn test_tensor_retain_null_no_crash() {
    tensor_retain(0); // guard: handle == 0 → no-op
}

#[test]
fn test_tensor_release_null_no_crash() {
    tensor_release(0); // guard: handle == 0 → no-op
}
