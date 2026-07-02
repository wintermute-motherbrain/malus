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

// M26 — gradient-check test infra. Tracks the max |analytic - numeric| diff
// seen across a malus run via record_diff(value), so Rust test code can
// assert a tolerance after the run without needing to capture stdout or
// reach into the JIT for an arbitrary return value. Platform-independent,
// same pattern as the CPU-compute counter above.
use std::sync::atomic::AtomicU32;

static GRADCHECK_MAX_DIFF_BITS: AtomicU32 = AtomicU32::new(0);

#[no_mangle]
pub extern "C" fn malus_record_diff(v: f32) {
    let bits = v.abs().to_bits();
    let mut cur = GRADCHECK_MAX_DIFF_BITS.load(Ordering::Relaxed);
    loop {
        if f32::from_bits(bits) <= f32::from_bits(cur) {
            return;
        }
        match GRADCHECK_MAX_DIFF_BITS.compare_exchange_weak(cur, bits, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

#[no_mangle]
pub extern "C" fn malus_gradcheck_max_diff() -> f32 {
    f32::from_bits(GRADCHECK_MAX_DIFF_BITS.load(Ordering::SeqCst))
}

#[no_mangle]
pub extern "C" fn malus_gradcheck_reset() {
    GRADCHECK_MAX_DIFF_BITS.store(0, Ordering::SeqCst);
}

// M29 — RC-op counters. Platform-independent; always compiled. Not the RC-ratio
// CI gate (that's a compile-time count of CTMM-emitted Retain/Release nodes,
// measured in malus-sema — see ADR-0026 "why this supersedes ADR-0002/0016").
// These runtime counters back a non-gating net-zero leak assertion
// (retain_count == release_count after a step) and are called from every
// tensor_retain/tensor_release/tensor_alloc_gpu entry point in metal.rs.
static RETAIN_COUNT: AtomicI64 = AtomicI64::new(0);
static RELEASE_COUNT: AtomicI64 = AtomicI64::new(0);
static ALLOC_COUNT: AtomicI64 = AtomicI64::new(0);

pub fn retain_inc() {
    RETAIN_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub fn release_inc() {
    RELEASE_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub fn alloc_inc() {
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
}

// Not `extern "C"`: a Rust tuple has no stable FFI ABI. This is a Rust-to-Rust
// test helper (never a JIT symbol), same pattern as `tape::registry_lens`.
pub fn malus_rc_counts() -> (i64, i64, i64) {
    (
        RETAIN_COUNT.load(Ordering::SeqCst),
        RELEASE_COUNT.load(Ordering::SeqCst),
        ALLOC_COUNT.load(Ordering::SeqCst),
    )
}

#[no_mangle]
pub extern "C" fn malus_rc_reset() {
    RETAIN_COUNT.store(0, Ordering::SeqCst);
    RELEASE_COUNT.store(0, Ordering::SeqCst);
    ALLOC_COUNT.store(0, Ordering::SeqCst);
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
    tensor_matmul, tensor_matmul_cpu,
    tensor_reshape, tensor_permute,
    tensor_causal_mask,
    // M19 randn
    tensor_randn,
    // M22 rand_uniform + rand_int + tensor_get_f32
    malus_rand_uniform, malus_rand_int, malus_tensor_get_f32,
    kernel_dispatch, gpu_barrier, flush_if_pending, tensor_is_pending, Dtype, TensorBuffer,
    // M23 — extended dispatch ABI (de-risk spike retired in M24)
    kernel_dispatch_v2,
};

// M26 / ADR-0031 / ADR-0032: retired CPU forward fallback (replaced by malus
// .ml kernels) — not in RuntimeSymbols, already unreachable from the JIT.
// Re-exported only behind cpu_fallback for malus-runtime's own direct-call
// unit tests (tests.rs).
#[cfg(target_os = "macos")]
#[cfg(feature = "cpu_fallback")]
pub use metal::{
    tensor_transpose, tensor_sum,
    tensor_broadcast_add, tensor_broadcast_sub, tensor_broadcast_mul, tensor_broadcast_div,
    tensor_reduce_sum_axis, tensor_reduce_mean_axis, tensor_reduce_max_axis, tensor_reduce_var_axis,
    tensor_softmax_axis, tensor_layernorm_axis, tensor_gelu,
    tensor_cross_entropy, tensor_embedding,
};

// M30 — warm per-step median timer (ADR-0038); macOS-only because
// bench_step_end flushes via gpu_barrier.
#[cfg(target_os = "macos")]
mod bench;

#[cfg(target_os = "macos")]
pub use bench::{
    bench_enable, bench_report, bench_reset, bench_step_begin, bench_step_end, BenchReport,
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
    // M26 — BwdSlot fn-ptr table (ADR-0032)
    tape_register_backward_fn, BwdSlot, N_BWD_SLOTS,
};

// M26: malus-runtime's own unit tests exercise backward() in isolation
// (without compile_and_run/malus-stdlib in the loop) and several retired
// CPU forward fns directly — both depend on cpu_fallback.
#[cfg(test)]
#[cfg(target_os = "macos")]
#[cfg(feature = "cpu_fallback")]
mod tests;
