#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

#[cfg(target_os = "macos")]
mod metal;

#[cfg(target_os = "macos")]
pub use metal::{
    runtime_init, tensor_alloc_gpu, tensor_alloc_zeros_gpu, tensor_alloc_ones_gpu,
    tensor_retain, tensor_release, tensor_free, tensor_print, tensor_len,
    tensor_matmul, tensor_transpose, tensor_sum,
    kernel_dispatch, gpu_barrier, Dtype, TensorBuffer,
};

#[cfg(target_os = "macos")]
mod tape;

#[cfg(target_os = "macos")]
pub use tape::{
    tape_record_binop, tape_record_unary, tape_register_leaf,
    tape_pause, tape_resume, tape_get_grad, tape_clear,
    backward, tape_zero_grad, OpTag, tape_reset,
};

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests;
