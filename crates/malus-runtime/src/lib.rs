// M22 string I/O — platform-independent.
mod strio;
pub use strio::{
    malus_str_box, malus_read_file, malus_str_len, malus_str_char_at, malus_str_from_char,
    StrBox,
};

#[cfg(target_os = "macos")]
mod metal;

#[cfg(target_os = "macos")]
pub use metal::{
    runtime_init, tensor_alloc_gpu, tensor_alloc_zeros_gpu, tensor_alloc_ones_gpu,
    tensor_retain, tensor_release, tensor_free, tensor_print, tensor_len,
    tensor_matmul, tensor_matmul_cpu, tensor_transpose, tensor_sum,
    tensor_broadcast_add, tensor_broadcast_sub, tensor_broadcast_mul, tensor_broadcast_div,
    tensor_reduce_sum_axis, tensor_reduce_mean_axis, tensor_reduce_max_axis, tensor_reduce_var_axis,
    tensor_reshape, tensor_permute,
    // M18 transformer stdlib
    tensor_softmax_axis, tensor_layernorm_axis, tensor_gelu,
    tensor_cross_entropy, tensor_causal_mask,
    // M19 embeddings + randn
    tensor_embedding, tensor_randn,
    kernel_dispatch, gpu_barrier, Dtype, TensorBuffer,
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
