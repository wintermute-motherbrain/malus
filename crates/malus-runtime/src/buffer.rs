// M22 — Buffer<i32>: mutable CPU-side staging buffer, freeze → Tensor<i32>.

use crate::metal::tensor_alloc_gpu;

struct BufferData {
    len: usize,
    data: Vec<i32>,
}

#[no_mangle]
pub extern "C" fn malus_buffer_i32(len: i64) -> i64 {
    let n = len as usize;
    let data = vec![0i32; n];
    let b = Box::new(BufferData { len: n, data });
    Box::into_raw(b) as i64
}

#[no_mangle]
pub extern "C" fn malus_buffer_get_i32(handle: i64, idx: i64) -> i64 {
    let bd = unsafe { &*(handle as *const BufferData) };
    let i = idx as usize;
    bd.data[i] as i64
}

#[no_mangle]
pub extern "C" fn malus_buffer_set_i32(handle: i64, idx: i64, val: i64) {
    let bd = unsafe { &mut *(handle as *mut BufferData) };
    let i = idx as usize;
    bd.data[i] = val as i32;
}

#[no_mangle]
pub extern "C" fn malus_buffer_free(handle: i64) {
    if handle == 0 { return; }
    unsafe { drop(Box::from_raw(handle as *mut BufferData)); }
}

#[no_mangle]
pub extern "C" fn malus_buffer_freeze_i32(handle: i64) -> i64 {
    let bd = unsafe { &*(handle as *const BufferData) };
    let shape = [bd.len];
    // dtype_tag 5 = I32 (matches malus-runtime's Dtype::I32.to_tag()).
    // Cast i32 slice to *const f32 — raw byte copy; tensor_alloc_gpu copies bytes.
    tensor_alloc_gpu(
        5,
        shape.as_ptr(),
        1,
        bd.data.as_ptr() as *const f32,
    )
}
