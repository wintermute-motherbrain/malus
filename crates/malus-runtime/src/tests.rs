use std::collections::HashMap;

use crate::{
    Dtype, TensorBuffer, runtime_init,
    tensor_alloc_gpu, tensor_alloc_zeros_gpu, tensor_alloc_ones_gpu,
    tensor_retain, tensor_release, tensor_free, tensor_print, tensor_len,
    tensor_matmul, tensor_transpose, tensor_sum,
    tensor_broadcast_add, tensor_broadcast_sub, tensor_broadcast_mul, tensor_broadcast_div,
    tensor_reduce_sum_axis, tensor_reduce_mean_axis, tensor_reduce_max_axis, tensor_reduce_var_axis,
    kernel_dispatch, gpu_barrier,
    tape_record_binop, tape_record_unary, tape_record_reduce, tape_register_leaf,
    tape_pause, tape_resume, tape_get_grad, tape_clear,
    backward, tape_zero_grad, OpTag, tape_reset,
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

// ── M14: tape + backward ──────────────────────────────────────────────────────

#[test]
fn test_optag_from_tag_drift() {
    assert_eq!(OpTag::from_tag(0),  OpTag::Matmul);
    assert_eq!(OpTag::from_tag(1),  OpTag::Add);
    assert_eq!(OpTag::from_tag(2),  OpTag::Sub);
    assert_eq!(OpTag::from_tag(3),  OpTag::Mul);
    assert_eq!(OpTag::from_tag(4),  OpTag::Div);
    assert_eq!(OpTag::from_tag(5),  OpTag::Sigmoid);
    assert_eq!(OpTag::from_tag(6),  OpTag::Relu);
    assert_eq!(OpTag::from_tag(7),  OpTag::Tanh);
    assert_eq!(OpTag::from_tag(8),  OpTag::Exp);
    assert_eq!(OpTag::from_tag(9),  OpTag::Log);
    assert_eq!(OpTag::from_tag(10), OpTag::Sqrt);
    assert_eq!(OpTag::from_tag(11), OpTag::Abs);
    assert_eq!(OpTag::from_tag(12), OpTag::Sum);
    assert_eq!(OpTag::from_tag(13), OpTag::Transpose);
    assert_eq!(OpTag::from_tag(14), OpTag::Neg);
    assert_eq!(OpTag::from_tag(15), OpTag::ReduceSumAxis);
    assert_eq!(OpTag::from_tag(16), OpTag::ReduceMeanAxis);
    assert_eq!(OpTag::from_tag(17), OpTag::ReduceMaxAxis);
    assert_eq!(OpTag::from_tag(18), OpTag::ReduceVarAxis);
}

#[test]
fn test_tape_clear_empty_no_crash() {
    tape_reset();
    tape_clear();
}

#[test]
fn test_no_grad_records_nothing() {
    tape_reset();
    let a = alloc_f32(&[1.0, 2.0]);
    let b = alloc_f32(&[3.0, 4.0]);
    tape_register_leaf(a);
    tape_register_leaf(b);
    tape_pause();
    let out = alloc_f32(&[4.0, 6.0]);
    tape_record_binop(OpTag::Add as i32, a, b, out);
    tape_resume();
    // Tape should still be empty because we were paused.
    backward(out);
    let ga = tape_get_grad(a);
    let gb = tape_get_grad(b);
    // No nodes recorded → grads were never accumulated → zeros returned.
    assert!(read_f32(ga).iter().all(|&v| v == 0.0), "paused record should produce zero grad");
    assert!(read_f32(gb).iter().all(|&v| v == 0.0));
    tensor_free(ga);
    tensor_free(gb);
    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

#[test]
fn test_backward_add() {
    tape_reset();
    let a = alloc_f32(&[1.0, 2.0]);
    let b = alloc_f32(&[3.0, 4.0]);
    tape_register_leaf(a);
    tape_register_leaf(b);
    // Forward: out = a + b  (simulated manually)
    let out = alloc_f32(&[4.0, 6.0]);
    tape_record_binop(OpTag::Add as i32, a, b, out);
    backward(out);
    let ga = tape_get_grad(a);
    let gb = tape_get_grad(b);
    // dA = dC = ones_like(out) = [1,1]; dB = dC = [1,1]
    assert_eq!(read_f32(ga), vec![1.0, 1.0]);
    assert_eq!(read_f32(gb), vec![1.0, 1.0]);
    tensor_free(ga);
    tensor_free(gb);
    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

#[test]
fn test_backward_sub() {
    tape_reset();
    let a = alloc_f32(&[5.0, 6.0]);
    let b = alloc_f32(&[2.0, 1.0]);
    tape_register_leaf(a);
    tape_register_leaf(b);
    let out = alloc_f32(&[3.0, 5.0]);
    tape_record_binop(OpTag::Sub as i32, a, b, out);
    backward(out);
    let ga = tape_get_grad(a);
    let gb = tape_get_grad(b);
    // dA = [1,1]; dB = -[1,1] = [-1,-1]
    assert_eq!(read_f32(ga), vec![1.0, 1.0]);
    assert_eq!(read_f32(gb), vec![-1.0, -1.0]);
    tensor_free(ga); tensor_free(gb);
    tensor_free(a); tensor_free(b); tensor_free(out);
}

#[test]
fn test_backward_mul() {
    tape_reset();
    let a = alloc_f32(&[2.0, 3.0]);
    let b = alloc_f32(&[4.0, 5.0]);
    tape_register_leaf(a);
    tape_register_leaf(b);
    let out = alloc_f32(&[8.0, 15.0]);
    tape_record_binop(OpTag::Mul as i32, a, b, out);
    backward(out);
    let ga = tape_get_grad(a);
    let gb = tape_get_grad(b);
    // dA = dC * B = [1,1]*[4,5] = [4,5]; dB = A * dC = [2,3]*[1,1] = [2,3]
    assert_eq!(read_f32(ga), vec![4.0, 5.0]);
    assert_eq!(read_f32(gb), vec![2.0, 3.0]);
    tensor_free(ga); tensor_free(gb);
    tensor_free(a); tensor_free(b); tensor_free(out);
}

#[test]
fn test_backward_neg() {
    tape_reset();
    let a = alloc_f32(&[1.0, -2.0, 3.0]);
    tape_register_leaf(a);
    let out = alloc_f32(&[-1.0, 2.0, -3.0]);
    tape_record_unary(OpTag::Neg as i32, a, out);
    backward(out);
    let ga = tape_get_grad(a);
    // dx = -dC = -[1,1,1] = [-1,-1,-1]
    assert_eq!(read_f32(ga), vec![-1.0, -1.0, -1.0]);
    tensor_free(ga); tensor_free(a); tensor_free(out);
}

#[test]
fn test_backward_relu() {
    tape_reset();
    // relu(x): mask = x>0
    let x = alloc_f32(&[-1.0, 0.0, 2.0, 3.0]);
    tape_register_leaf(x);
    let out = alloc_f32(&[0.0, 0.0, 2.0, 3.0]);
    tape_record_unary(OpTag::Relu as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    // dC = ones; mask = [0,0,1,1]; dx = [0,0,1,1]
    assert_eq!(read_f32(gx), vec![0.0, 0.0, 1.0, 1.0]);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_sigmoid() {
    tape_reset();
    // s = sigmoid(0) = 0.5; ds/dx = s*(1-s) = 0.25
    let x = alloc_f32(&[0.0]);
    tape_register_leaf(x);
    let s_val = 1.0_f32 / (1.0 + (-0.0_f32).exp());
    let out = alloc_f32(&[s_val]);
    tape_record_unary(OpTag::Sigmoid as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    let gx_data = read_f32(gx);
    let expected = s_val * (1.0 - s_val);
    assert!((gx_data[0] - expected).abs() < 1e-6, "sigmoid grad: got {}, want {}", gx_data[0], expected);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_tanh() {
    tape_reset();
    let x = alloc_f32(&[0.0]);
    tape_register_leaf(x);
    let t_val = 0.0_f32.tanh(); // = 0
    let out = alloc_f32(&[t_val]);
    tape_record_unary(OpTag::Tanh as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    let expected = 1.0 - t_val * t_val; // = 1.0
    assert!((read_f32(gx)[0] - expected).abs() < 1e-6);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_exp() {
    tape_reset();
    let x = alloc_f32(&[1.0]);
    tape_register_leaf(x);
    let e_val = 1.0_f32.exp();
    let out = alloc_f32(&[e_val]);
    tape_record_unary(OpTag::Exp as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    // dx = dC * e = 1 * e = e
    assert!((read_f32(gx)[0] - e_val).abs() < 1e-6);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_log() {
    tape_reset();
    let x = alloc_f32(&[2.0]);
    tape_register_leaf(x);
    let out = alloc_f32(&[2.0_f32.ln()]);
    tape_record_unary(OpTag::Log as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    // dx = dC / x = 1 / 2 = 0.5
    assert!((read_f32(gx)[0] - 0.5).abs() < 1e-6);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_sqrt() {
    tape_reset();
    let x = alloc_f32(&[4.0]);
    tape_register_leaf(x);
    let s = 4.0_f32.sqrt(); // = 2.0
    let out = alloc_f32(&[s]);
    tape_record_unary(OpTag::Sqrt as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    // dx = dC / (2*s) = 1 / 4 = 0.25
    assert!((read_f32(gx)[0] - 0.25).abs() < 1e-6);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_abs() {
    tape_reset();
    let x = alloc_f32(&[-3.0, 0.0, 2.0]);
    tape_register_leaf(x);
    let out = alloc_f32(&[3.0, 0.0, 2.0]);
    tape_record_unary(OpTag::Abs as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    // sign: -1, 0, 1
    assert_eq!(read_f32(gx), vec![-1.0, 0.0, 1.0]);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_div() {
    tape_reset();
    // C = A / B; A=[6,8], B=[2,4]
    let a = alloc_f32(&[6.0, 8.0]);
    let b = alloc_f32(&[2.0, 4.0]);
    tape_register_leaf(a);
    tape_register_leaf(b);
    let out = alloc_f32(&[3.0, 2.0]);
    tape_record_binop(OpTag::Div as i32, a, b, out);
    backward(out);
    let ga = tape_get_grad(a);
    let gb = tape_get_grad(b);
    // dA = dC / B = [1,1] / [2,4] = [0.5, 0.25]
    let ga_data = read_f32(ga);
    assert!((ga_data[0] - 0.5).abs() < 1e-6, "dA[0]: {}", ga_data[0]);
    assert!((ga_data[1] - 0.25).abs() < 1e-6, "dA[1]: {}", ga_data[1]);
    // dB = -dC * A / B² = -[1,1] * [6,8] / [4,16] = [-1.5, -0.5]
    let gb_data = read_f32(gb);
    assert!((gb_data[0] - (-1.5)).abs() < 1e-6, "dB[0]: {}", gb_data[0]);
    assert!((gb_data[1] - (-0.5)).abs() < 1e-6, "dB[1]: {}", gb_data[1]);
    tensor_free(ga); tensor_free(gb);
    tensor_free(a); tensor_free(b); tensor_free(out);
}

#[test]
fn test_backward_sum() {
    tape_reset();
    let x = alloc_f32(&[1.0, 2.0, 3.0]);
    tape_register_leaf(x);
    let out = alloc_f32(&[6.0]); // sum = 6
    tape_record_unary(OpTag::Sum as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    // dx = ones_like(x) * dC[0] = [1,1,1]*1 = [1,1,1]
    assert_eq!(read_f32(gx), vec![1.0, 1.0, 1.0]);
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_transpose() {
    tape_reset();
    // [[1,2],[3,4]] transposed = [[1,3],[2,4]]
    let x = alloc_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    tape_register_leaf(x);
    let out = alloc_2d(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    tape_record_unary(OpTag::Transpose as i32, x, out);
    backward(out);
    let gx = tape_get_grad(x);
    let gx_tb = unsafe { &*(gx as *const TensorBuffer) };
    // dA = dBᵀ.  dB = ones[2,2].  dBᵀ = ones[2,2] transposed = ones[2,2].
    assert_eq!(gx_tb.shape, &[2, 2]);
    assert!(read_f32(gx).iter().all(|&v| v == 1.0));
    tensor_free(gx); tensor_free(x); tensor_free(out);
}

#[test]
fn test_backward_matmul() {
    tape_reset();
    // A=[2,3] @ B=[3,2] -> C=[2,2]; finite-diff check
    let a_data = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x3
    let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
    let a = alloc_2d(&a_data, 2, 3);
    let b = alloc_2d(&b_data, 3, 2);
    tape_register_leaf(a);
    tape_register_leaf(b);
    let out = tensor_matmul(a, b);
    tape_record_binop(OpTag::Matmul as i32, a, b, out);
    backward(out); // seeds with ones[2,2]

    let ga = tape_get_grad(a);
    let gb = tape_get_grad(b);
    let ga_tb = unsafe { &*(ga as *const TensorBuffer) };
    let gb_tb = unsafe { &*(gb as *const TensorBuffer) };

    // dA = dC @ Bᵀ; dB = Aᵀ @ dC (dC = ones[2,2])
    assert_eq!(ga_tb.shape, &[2, 3]);
    assert_eq!(gb_tb.shape, &[3, 2]);
    // dA = ones[2,2] @ B^T; B^T = [[1,3,5],[2,4,6]]; row sums = [1+2,3+4,5+6] = [3,7,11]
    let ga_data = read_f32(ga);
    assert!((ga_data[0] - 3.0).abs() < 1e-5, "ga[0]: {}", ga_data[0]);
    assert!((ga_data[1] - 7.0).abs() < 1e-5, "ga[1]: {}", ga_data[1]);
    assert!((ga_data[2] - 11.0).abs() < 1e-5, "ga[2]: {}", ga_data[2]);

    tensor_free(ga); tensor_free(gb);
    tensor_free(a); tensor_free(b); tensor_free(out);
}

#[test]
fn test_tape_clears_after_backward() {
    tape_reset();
    let a = alloc_f32(&[1.0]);
    let b = alloc_f32(&[2.0]);
    tape_register_leaf(a);
    let out = alloc_f32(&[3.0]);
    tape_record_binop(OpTag::Add as i32, a, b, out);
    backward(out);
    // backward calls tape_clear; push another node and clear manually
    let g = tape_get_grad(a);
    tensor_free(g);
    // Another backward with empty tape should be a no-op
    let loss2 = alloc_f32(&[0.0]);
    backward(loss2);
    tensor_free(loss2);
    tensor_free(a); tensor_free(b); tensor_free(out);
}

#[test]
fn test_leaf_grad_accumulates_across_two_backward_calls() {
    tape_reset();
    let a = alloc_f32(&[2.0]);
    tape_register_leaf(a);

    // First backward: a + b1
    let b1 = alloc_f32(&[3.0]);
    let out1 = alloc_f32(&[5.0]);
    tape_record_binop(OpTag::Add as i32, a, b1, out1);
    backward(out1);
    // Second backward: a + b2
    let b2 = alloc_f32(&[7.0]);
    let out2 = alloc_f32(&[9.0]);
    tape_record_binop(OpTag::Add as i32, a, b2, out2);
    backward(out2);

    // Each backward seeds ones[1] → da=1 per call → accumulated = 2
    let ga = tape_get_grad(a);
    assert_eq!(read_f32(ga), vec![2.0], "should accumulate across two backward calls");
    tensor_free(ga);
    tensor_free(a); tensor_free(b1); tensor_free(b2); tensor_free(out1); tensor_free(out2);
}

#[test]
fn test_chain_sum_sigmoid_matmul() {
    // Smoke test: loss = sum(sigmoid(x @ w)); backward works without panic.
    tape_reset();
    let x_data = [1.0f32, 2.0, 3.0, 4.0];
    let w_data = [0.1f32, 0.2, 0.3, 0.4];
    let x = alloc_2d(&x_data, 2, 2);
    let w = alloc_2d(&w_data, 2, 2);
    tape_register_leaf(w);

    let mm = tensor_matmul(x, w);
    tape_record_binop(OpTag::Matmul as i32, x, w, mm);

    let sig_data: Vec<f32> = read_f32(mm).iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    let sig = alloc_like_vec(mm, &sig_data);
    tape_record_unary(OpTag::Sigmoid as i32, mm, sig);

    let s: f32 = sig_data.iter().sum();
    let loss = alloc_f32(&[s]);
    tape_record_unary(OpTag::Sum as i32, sig, loss);

    backward(loss);

    let gw = tape_get_grad(w);
    let gw_tb = unsafe { &*(gw as *const TensorBuffer) };
    assert_eq!(gw_tb.shape, &[2, 2], "grad w should be [2,2]");

    tensor_free(gw);
    tensor_free(x); tensor_free(w); tensor_free(mm); tensor_free(sig); tensor_free(loss);
}

fn alloc_like_vec(template: i64, data: &[f32]) -> i64 {
    let tb = unsafe { &*(template as *const TensorBuffer) };
    tensor_alloc_gpu(0, tb.shape.as_ptr(), tb.shape.len(), data.as_ptr())
}

// ── M15: zero_grad + leaf-registry lifecycle ──────────────────────────────────

#[test]
fn test_zero_grad_clears_leaf_grad() {
    tape_reset();
    let x = alloc_f32(&[3.0]);
    tape_register_leaf(x);
    let out = alloc_f32(&[3.0]);
    tape_record_unary(OpTag::Neg as i32, x, out);
    backward(out);

    // After backward, x has an accumulated grad.
    let g1 = tape_get_grad(x);
    let g1_val = read_f32(g1)[0];
    tensor_release(g1);
    assert!((g1_val - (-1.0)).abs() < 1e-6, "expected grad -1.0, got {g1_val}");

    // zero_grad should clear it; next tape_get_grad returns zeros.
    let handles = [x];
    tape_zero_grad(handles.as_ptr(), handles.len());
    let g2 = tape_get_grad(x);
    let g2_val = read_f32(g2)[0];
    tensor_release(g2);
    assert_eq!(g2_val, 0.0, "grad should be 0 after zero_grad");

    tensor_release(x);
    tensor_release(out);
}

#[test]
fn test_rewrap_registry_stays_bounded() {
    // Simulate the SGD re-wrap idiom across 50 iterations and assert that
    // LEAVES and LEAF_GRAD stay bounded — this is the core M15 leak check.
    // Without the tape_on_release hook, LEAVES grows by 1 per iteration.
    tape_reset();

    let mut w = alloc_f32(&[0.5, 0.5]);
    let lr = 0.01f32;

    for _ in 0..50 {
        tape_register_leaf(w);

        // Tiny forward: out = -w  (Neg VJP: dx = -dout = -ones)
        let out = alloc_f32(&[-0.5, -0.5]);
        tape_record_unary(OpTag::Neg as i32, w, out);
        backward(out);
        tensor_release(out);

        let g = tape_get_grad(w);

        // zero_grad
        let handles = [w];
        tape_zero_grad(handles.as_ptr(), handles.len());

        // SGD update: new_w = w_data - lr * g  (produce new tensor)
        let w_data = read_f32(w);
        let g_data = read_f32(g);
        let new_data: Vec<f32> = w_data.iter().zip(g_data.iter())
            .map(|(wi, gi)| wi - lr * gi).collect();
        let shape = [2usize];
        let new_w = tensor_alloc_gpu(0, shape.as_ptr(), 1, new_data.as_ptr());

        tensor_release(g);
        // Release old leaf — triggers tape_on_release, deregisters from LEAVES/LEAF_GRAD.
        tensor_release(w);
        w = new_w;
    }

    let (leaves, grads) = crate::tape::registry_lens();
    assert!(leaves <= 1, "LEAVES must stay bounded across re-wrap, got {leaves}");
    assert!(grads  == 0, "LEAF_GRAD must be empty after zero_grad + re-wrap, got {grads}");

    tensor_release(w);
    tape_reset();
}

// ── M16: broadcasting + axis reduction tests ──────────────────────────────────

fn read_shape(handle: i64) -> Vec<usize> {
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    tb.shape.clone()
}

fn alloc_nd(data: &[f32], shape: &[usize]) -> i64 {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    tensor_alloc_gpu(0, shape.as_ptr(), shape.len(), data.as_ptr())
}

#[test]
fn test_broadcast_add_equal_shapes_fast_path() {
    // Equal shapes: goes through GPU fast path (kernel_dispatch with the registered add kernel).
    init_add_kernel();
    let a = alloc_f32(&[1.0, 2.0, 3.0, 4.0]);
    let b = alloc_f32(&[10.0, 20.0, 30.0, 40.0]);
    let out = tensor_broadcast_add(0, a, b); // kernel_id=0 = registered add kernel
    gpu_barrier();
    let result = read_f32(out);
    assert_eq!(result, [11.0, 22.0, 33.0, 44.0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

#[test]
fn test_broadcast_add_rank_expansion() {
    // (8,) + (4,8) → (4,8) via CPU broadcast loop.
    let b_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let a_data: Vec<f32> = vec![1.0f32; 32];
    let b = alloc_nd(&b_data, &[8]);
    let a = alloc_nd(&a_data, &[4, 8]);
    let out = tensor_broadcast_add(0, a, b);
    let result = read_f32(out);
    let shape = read_shape(out);
    assert_eq!(shape, vec![4, 8]);
    // Each of the 4 rows should be [2,3,4,5,6,7,8,9].
    for row in 0..4 {
        for col in 0..8 {
            let expected = 1.0 + (col as f32 + 1.0);
            assert!((result[row * 8 + col] - expected).abs() < 1e-5,
                    "row={} col={} expected={} got={}", row, col, expected, result[row * 8 + col]);
        }
    }
    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

#[test]
fn test_broadcast_sub_scalar_row() {
    // (1,4) - (3,4) → (3,4)
    let a = alloc_nd(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let b = alloc_nd(&[1.0; 12], &[3, 4]);
    let out = tensor_broadcast_sub(0, a, b);
    let result = read_f32(out);
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().cycle().take(12)
        .zip(vec![1.0f32; 12].iter()).map(|(x, y)| x - y).collect();
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "index {} expected {} got {}", i, e, r);
    }
    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

#[test]
fn test_reduce_sum_axis0_no_keepdim() {
    // sum([[1,2,3],[4,5,6]], axis=0) → [5,7,9]
    let h = alloc_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let out = tensor_reduce_sum_axis(h, 0, 0);
    let result = read_f32(out);
    let shape = read_shape(out);
    assert_eq!(shape, vec![3]);
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
    tensor_free(h);
    tensor_free(out);
}

#[test]
fn test_reduce_sum_axis1_keepdim() {
    // sum([[1,2,3],[4,5,6]], axis=1, keepdim=1) → [[6],[15]]
    let h = alloc_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let out = tensor_reduce_sum_axis(h, 1, 1);
    let result = read_f32(out);
    let shape = read_shape(out);
    assert_eq!(shape, vec![2, 1]);
    assert_eq!(result, vec![6.0, 15.0]);
    tensor_free(h);
    tensor_free(out);
}

#[test]
fn test_reduce_mean_axis0() {
    let h = alloc_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let out = tensor_reduce_mean_axis(h, 0, 0);
    let result = read_f32(out);
    let shape = read_shape(out);
    assert_eq!(shape, vec![3]);
    let expected = vec![2.5, 3.5, 4.5];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5);
    }
    tensor_free(h);
    tensor_free(out);
}

#[test]
fn test_reduce_max_axis1() {
    let h = alloc_nd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let out = tensor_reduce_max_axis(h, 1, 0);
    let result = read_f32(out);
    let shape = read_shape(out);
    assert_eq!(shape, vec![2]);
    assert_eq!(result, vec![5.0, 6.0]);
    tensor_free(h);
    tensor_free(out);
}

#[test]
fn test_reduce_var_axis0() {
    // var of [1,4] along axis 0 = var([1,4]) = ((1-2.5)^2 + (4-2.5)^2)/2 = 2.25
    let h = alloc_nd(&[1.0, 2.0, 4.0, 8.0], &[2, 2]);
    let out = tensor_reduce_var_axis(h, 0, 0);
    let result = read_f32(out);
    let shape = read_shape(out);
    assert_eq!(shape, vec![2]);
    // col0: mean=2.5, var=((1-2.5)^2+(4-2.5)^2)/2=2.25
    // col1: mean=5.0, var=((2-5)^2+(8-5)^2)/2=9.0
    assert!((result[0] - 2.25).abs() < 1e-4, "col0 var expected 2.25 got {}", result[0]);
    assert!((result[1] - 9.0).abs() < 1e-4, "col1 var expected 9.0 got {}", result[1]);
    tensor_free(h);
    tensor_free(out);
}

#[test]
fn test_reduce_negative_axis() {
    // axis=-1 on (3,4) should equal axis=1.
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let h = alloc_nd(&data, &[3, 4]);
    let out_pos = tensor_reduce_sum_axis(h, 1, 0);
    let out_neg = tensor_reduce_sum_axis(h, -1, 0);
    assert_eq!(read_f32(out_pos), read_f32(out_neg));
    assert_eq!(read_shape(out_pos), read_shape(out_neg));
    tensor_free(h);
    tensor_free(out_pos);
    tensor_free(out_neg);
}

#[test]
fn test_tape_record_reduce_backward_sum() {
    // sum(x, axis=0) backward: dx = broadcast_to(dout, x.shape).
    tape_reset();
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = alloc_nd(&x_data, &[2, 3]);
    tensor_retain(x);
    tape_register_leaf(x);

    let out = tensor_reduce_sum_axis(x, 0, 0); // shape [3]
    tensor_retain(out);
    tape_record_reduce(OpTag::ReduceSumAxis as i32, x, out, 0, 0);

    backward(out);

    let dx = tape_get_grad(x);
    let dx_data = read_f32(dx);
    let dx_shape = read_shape(dx);
    // dx should be ones of shape [2,3] (dout=[1,1,1] broadcast to [2,3]).
    assert_eq!(dx_shape, vec![2, 3]);
    for v in &dx_data { assert!((v - 1.0).abs() < 1e-5, "expected 1.0 got {v}"); }
    tensor_release(dx);
    tensor_release(x);
    tensor_free(out);
    tape_reset();
}

#[test]
fn test_broadcast_add_backward() {
    // (1,3) + (2,3) — sum VJP reduces dout to each operand's shape.
    tape_reset();
    let a_data = vec![1.0f32, 1.0, 1.0];
    let b_data = vec![1.0f32; 6];
    let a = alloc_nd(&a_data, &[1, 3]);
    let b = alloc_nd(&b_data, &[2, 3]);
    tensor_retain(a);
    tensor_retain(b);
    tape_register_leaf(a);
    tape_register_leaf(b);

    let out = tensor_broadcast_add(0, a, b); // shape [2,3]
    tensor_retain(out);
    tape_record_binop(OpTag::Add as i32, a, b, out);

    // Use sum to get a scalar loss.
    let loss_h = tensor_sum(out);
    tensor_retain(loss_h);
    tape_record_unary(OpTag::Sum as i32, out, loss_h);

    backward(loss_h);

    let da = tape_get_grad(a);
    let db = tape_get_grad(b);

    let da_data = read_f32(da);
    let db_data = read_f32(db);

    // dout is all ones shape [2,3]; da = sum over axis 0 → shape [1,3], each = 2.
    assert_eq!(read_shape(da), vec![1, 3]);
    for v in &da_data { assert!((v - 2.0).abs() < 1e-4, "da expected 2.0 got {v}"); }
    // db = dout [2,3] — no reduction needed, each = 1.
    assert_eq!(read_shape(db), vec![2, 3]);
    for v in &db_data { assert!((v - 1.0).abs() < 1e-4, "db expected 1.0 got {v}"); }

    tensor_release(da);
    tensor_release(db);
    tensor_release(a);
    tensor_release(b);
    tensor_free(out);
    tensor_free(loss_h);
    tape_reset();
}
