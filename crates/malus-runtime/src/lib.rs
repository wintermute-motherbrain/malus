// M23 — CPU-compute counter. Platform-independent; always compiled.
use std::sync::atomic::{AtomicI64, Ordering};

static CPU_COMPUTE_CALLS: AtomicI64 = AtomicI64::new(0);

/// Increment the CPU-compute counter.  Called at the entry of every Rust
/// function that loops over tensor element values.  The V4 CI gate asserts
/// this count is 0 over any hot-path dispatch that should run on the GPU.
pub fn cpu_compute_inc() {
    CPU_COMPUTE_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn malus_cpu_compute_count() -> i64 {
    CPU_COMPUTE_CALLS.load(Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn malus_cpu_compute_reset() {
    CPU_COMPUTE_CALLS.store(0, Ordering::SeqCst);
}

// M22 string I/O — platform-independent.
mod strio;
pub use strio::{
    malus_str_box, malus_read_file, malus_str_len, malus_str_char_at, malus_str_from_char,
    StrBox,
};

#[cfg(target_os = "macos")]
mod metal;

// M22 Buffer<i32> — macOS-only because freeze calls tensor_alloc_gpu.
#[cfg(target_os = "macos")]
mod buffer;

#[cfg(target_os = "macos")]
pub use buffer::{
    malus_buffer_i32, malus_buffer_get_i32, malus_buffer_set_i32,
    malus_buffer_free, malus_buffer_freeze_i32,
};

#[cfg(target_os = "macos")]
pub use metal::{
    runtime_init, tensor_alloc_gpu, tensor_alloc_zeros_gpu, tensor_alloc_ones_gpu,
    tensor_retain, tensor_release, tensor_free, tensor_print, tensor_len,
    // M25 — metadata accessors (no cpu_compute_inc)
    tensor_ndim, tensor_dim,
    tensor_matmul, tensor_matmul_cpu, tensor_transpose, tensor_sum,
    tensor_broadcast_add, tensor_broadcast_sub, tensor_broadcast_mul, tensor_broadcast_div,
    tensor_reduce_sum_axis, tensor_reduce_mean_axis, tensor_reduce_max_axis, tensor_reduce_var_axis,
    tensor_reshape, tensor_permute,
    // M18 transformer stdlib
    tensor_softmax_axis, tensor_layernorm_axis, tensor_gelu,
    tensor_cross_entropy, tensor_causal_mask,
    // M19 embeddings + randn
    tensor_embedding, tensor_randn,
    // M22 rand_uniform + rand_int + tensor_get_f32
    malus_rand_uniform, malus_rand_int, malus_tensor_get_f32,
    kernel_dispatch, gpu_barrier, Dtype, TensorBuffer,
    // M23 — extended dispatch ABI (de-risk spike retired in M24)
    kernel_dispatch_v2,
};

#[cfg(target_os = "macos")]
mod tape;

#[cfg(target_os = "macos")]
pub use tape::{
    tape_record_binop, tape_record_unary, tape_record_reduce, tape_record_perm,
    // M18 transformer stdlib recorders
    tape_record_layernorm, tape_record_cross_entropy,
    // M19 embedding recorder
    tape_record_embedding,
    tape_register_leaf,
    tape_pause, tape_resume, tape_get_grad, tape_clear,
    backward, tape_zero_grad, OpTag, tape_reset,
};

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests;
