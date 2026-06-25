use crate::{Dtype, TensorBuffer, tensor_alloc_gpu, tensor_free, tensor_print, kernel_dispatch, gpu_barrier};

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
#[should_panic(expected = "unknown dtype")]
fn test_dtype_unknown_tag_panics() {
    Dtype::from_tag(99);
}

fn alloc_f32(data: &[f32]) -> i64 {
    tensor_alloc_gpu(0, data.len() as i64, data.as_ptr())
}

#[test]
fn test_tensor_alloc_roundtrip() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let handle = alloc_f32(&data);
    assert!(handle != 0, "handle should be non-null");

    let tb = unsafe { &*(handle as *const TensorBuffer) };
    assert_eq!(tb.len, 4);
    assert_eq!(tb.dtype, Dtype::F32);

    let ptr = tb.buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, tb.len) };
    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);

    tensor_free(handle);
}

#[test]
fn test_tensor_alloc_null_data() {
    let handle = tensor_alloc_gpu(0, 4, std::ptr::null());
    assert!(handle != 0);

    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let ptr = tb.buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, tb.len) };
    assert!(slice.iter().all(|v| *v == 0.0), "buffer should be zeroed");

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
fn test_gpu_barrier_returns() {
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

#[test]
fn test_kernel_dispatch_stub() {
    let input = alloc_f32(&[1.0f32, 2.0, 3.0, 4.0]);
    let handles = [input];
    let output = kernel_dispatch(std::ptr::null(), handles.as_ptr(), 1);
    assert!(output != 0);

    let tb = unsafe { &*(output as *const TensorBuffer) };
    assert_eq!(tb.len, 4, "output len should match first input");

    let ptr = tb.buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, tb.len) };
    assert!(slice.iter().all(|v| *v == 0.0), "output should be zeroed");

    tensor_free(input);
    tensor_free(output);
}
