#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

#[cfg(target_os = "macos")]
mod metal;

#[cfg(target_os = "macos")]
pub use metal::{
    runtime_init, tensor_alloc_gpu, tensor_free, tensor_print,
    kernel_dispatch, gpu_barrier, Dtype, TensorBuffer,
};

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests;
