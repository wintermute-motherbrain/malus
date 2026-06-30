#![cfg(target_os = "macos")]

use malus_codegen_cpu::{compile_and_run, RuntimeSymbols};
use malus_sema::check;
use malus_syntax::parse;
use std::collections::HashMap;
use std::sync::Mutex;

static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn real_symbols() -> RuntimeSymbols {
    RuntimeSymbols {
        tensor_alloc_gpu:       malus_runtime::tensor_alloc_gpu,
        tensor_free:            malus_runtime::tensor_free,
        tensor_print:           malus_runtime::tensor_print,
        kernel_dispatch:        malus_runtime::kernel_dispatch,
        gpu_barrier:            malus_runtime::gpu_barrier,
        tensor_alloc_zeros_gpu: malus_runtime::tensor_alloc_zeros_gpu,
        tensor_alloc_ones_gpu:  malus_runtime::tensor_alloc_ones_gpu,
        tensor_matmul:          malus_runtime::tensor_matmul,
        tensor_len:             malus_runtime::tensor_len,
        tensor_retain:          malus_runtime::tensor_retain,
        tensor_release:         malus_runtime::tensor_release,
        tape_record_binop:      malus_runtime::tape_record_binop,
        tape_record_unary:      malus_runtime::tape_record_unary,
        tape_register_leaf:     malus_runtime::tape_register_leaf,
        tape_pause:             malus_runtime::tape_pause,
        tape_resume:            malus_runtime::tape_resume,
        tape_clear:             malus_runtime::tape_clear,
        tape_get_grad:          malus_runtime::tape_get_grad,
        backward:               malus_runtime::backward,
        tape_zero_grad:         malus_runtime::tape_zero_grad,
        tape_record_reduce:     malus_runtime::tape_record_reduce,
        tensor_reshape:         malus_runtime::tensor_reshape,
        tensor_permute:         malus_runtime::tensor_permute,
        tape_record_perm:       malus_runtime::tape_record_perm,
        // M18 transformer stdlib.
        tensor_causal_mask:        malus_runtime::tensor_causal_mask,
        tape_record_layernorm:     malus_runtime::tape_record_layernorm,
        tape_record_cross_entropy: malus_runtime::tape_record_cross_entropy,
        // M19 randn.
        tensor_randn:              malus_runtime::tensor_randn,
        tape_record_embedding:     malus_runtime::tape_record_embedding,
        // M22 string I/O.
        malus_str_box:             malus_runtime::malus_str_box,
        malus_read_file:           malus_runtime::malus_read_file,
        malus_str_len:             malus_runtime::malus_str_len,
        malus_str_char_at:         malus_runtime::malus_str_char_at,
        malus_str_from_char:       malus_runtime::malus_str_from_char,
        // M22 rand_uniform.
        malus_rand_uniform:        malus_runtime::malus_rand_uniform,
        // M22 Buffer<i32>.
        malus_buffer_i32:          malus_runtime::malus_buffer_i32,
        malus_buffer_get_i32:      malus_runtime::malus_buffer_get_i32,
        malus_buffer_set_i32:      malus_runtime::malus_buffer_set_i32,
        malus_buffer_free:         malus_runtime::malus_buffer_free,
        malus_buffer_freeze_i32:   malus_runtime::malus_buffer_freeze_i32,
        // M22 rand_int + tensor_get_f32.
        malus_rand_int:            malus_runtime::malus_rand_int,
        malus_tensor_get_f32:      malus_runtime::malus_tensor_get_f32,
        // M25 metadata accessors + kernel_dispatch_v2.
        tensor_ndim:               malus_runtime::tensor_ndim,
        tensor_dim:                malus_runtime::tensor_dim,
        kernel_dispatch_v2:        malus_runtime::kernel_dispatch_v2,
        tape_register_backward_fn: malus_runtime::tape_register_backward_fn,
        malus_record_diff:         malus_runtime::malus_record_diff,
    }
}

fn run_metal_src(src: &str) {
    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut user_program = parse(malus_syntax::FileId(0), src).expect("parse failed");
    let mut stdlib = malus_stdlib::stdlib_items();
    stdlib.extend(user_program.items.drain(..));
    let program = malus_syntax::ast::Program { items: stdlib };
    let aliases = HashMap::new();
    let typed = check(&program, &aliases).expect("type check failed");
    let (registry, kernel_ids) =
        malus_codegen_gpu::compile_kernels(&typed).expect("kernel compilation failed");
    malus_runtime::runtime_init(&registry.into_hashmap());
    let symbols = real_symbols();
    compile_and_run(&typed, &symbols, &kernel_ids).expect("JIT compile and run failed");
}

#[test]
fn test_fn_body_tensor_add_correct_on_metal() {
    let src = r#"
fn add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    println(c)
"#;
    run_metal_src(src);
}

#[test]
fn test_fn_body_binop_direct_correct_on_metal() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = a + b
    println(c)
"#;
    run_metal_src(src);
}

#[test]
fn test_chained_fn_body_binops_correct_on_metal() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = Tensor.gpu<f32>([5.0, 6.0])
    let r = a + b * c
    println(r)
"#;
    run_metal_src(src);
}

#[test]
fn test_golden_example_still_runs_on_metal() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    println(c)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
"#;
    run_metal_src(src);
}

// V4 M24 CI gate: softmax/layernorm/gelu authored in the malus kernel language,
// dispatched via kernel_dispatch_v2, produce numerically correct output with
// cpu_compute_count()==0.  The M23 spike (register_m23_softmax_row_kernel /
// M23_SOFTMAX_ROW_KERNEL_ID) is retired; these tests replace it.

fn setup_kernel(src: &str) -> (malus_codegen_gpu::KernelRegistry, std::collections::HashMap<String, u64>) {
    let program = malus_syntax::parse(malus_syntax::FileId(0), src).expect("parse failed");
    let aliases = std::collections::HashMap::new();
    let typed = malus_sema::check(&program, &aliases).expect("type check failed");
    let (registry, kernel_ids) =
        malus_codegen_gpu::compile_kernels(&typed).expect("kernel compilation failed");
    malus_runtime::runtime_init(&registry.clone().into_hashmap());
    (registry, kernel_ids)
}

#[test]
fn test_v4_m24_softmax_gpu_counter_zero() {
    fn softmax_ref(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row = &input[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for c in 0..cols { out[r * cols + c] = exps[c] / sum; }
        }
        out
    }

    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Inline the malus kernel source with a dummy main() so sema is satisfied.
    let src = r#"
kernel softmax(input: Tensor<f32>, cols: i32) -> Tensor<f32>:
    let row = threadgroup_id()
    let col = thread_in_threadgroup()
    let start = row * cols
    let shared scratch: Array<f32, 1024>
    scratch[col] = input[start + col]
    barrier()
    let mut m = scratch[0]
    for i in range(1, cols):
        m = fmax(m, scratch[i])
    barrier()
    scratch[col] = exp(scratch[col] - m)
    barrier()
    let mut s = 0.0
    for j in range(0, cols):
        s = s + scratch[j]
    barrier()
    out[start + col] = scratch[col] / s

fn main():
    let _a = 0
"#;
    let (_registry, kernel_ids) = setup_kernel(src);
    let kernel_id = *kernel_ids.get("softmax").expect("softmax kernel not found");

    let rows: usize = 4;
    let cols: usize = 8;
    let input_data: [f32; 32] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6,
    ];
    let reference = softmax_ref(&input_data, rows, cols);

    malus_runtime::malus_cpu_compute_reset();

    let in_shape = [rows, cols];
    let in_handle = unsafe {
        malus_runtime::tensor_alloc_gpu(0, in_shape.as_ptr(), 2, input_data.as_ptr())
    };

    let out_shape = [rows, cols];
    let grid: [usize; 3] = [rows, 1, 1];
    let tg:   [usize; 3] = [cols, 1, 1];
    let uniforms: i32 = cols as i32;

    let out_handle = unsafe {
        malus_runtime::kernel_dispatch_v2(
            kernel_id,
            &in_handle as *const i64,
            1,
            grid.as_ptr(),
            tg.as_ptr(),
            out_shape.as_ptr(),
            2,
            0,
            &uniforms as *const i32 as *const std::ffi::c_void,
            std::mem::size_of::<i32>(),
        )
    };
    malus_runtime::gpu_barrier();

    let cpu_count = malus_runtime::malus_cpu_compute_count();

    for i in 0..(rows * cols) {
        let got = unsafe { malus_runtime::malus_tensor_get_f32(out_handle, i as i64) };
        assert!(
            (got - reference[i]).abs() < 1e-5,
            "softmax output[{i}]: got {got:.6}, expected {:.6}", reference[i]
        );
    }
    assert_eq!(cpu_count, 0, "CPU compute was invoked during GPU softmax dispatch");

    malus_runtime::tensor_free(in_handle);
    malus_runtime::tensor_free(out_handle);
}

#[test]
fn test_v4_m24_layernorm_gpu_counter_zero() {
    fn layernorm_ref(input: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row = &input[r * cols..(r + 1) * cols];
            let mean = row.iter().sum::<f32>() / cols as f32;
            let var = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
            let inv_std = (var + eps).sqrt().recip();
            for c in 0..cols { out[r * cols + c] = (row[c] - mean) * inv_std; }
        }
        out
    }

    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
kernel layernorm(input: Tensor<f32>, cols: i32, inv_cols: f32, eps: f32) -> Tensor<f32>:
    let row = threadgroup_id()
    let col = thread_in_threadgroup()
    let start = row * cols
    let shared scratch: Array<f32, 1024>
    scratch[col] = input[start + col]
    barrier()
    let mut s = 0.0
    for i in range(0, cols):
        s = s + scratch[i]
    let mean = s * inv_cols
    barrier()
    let mut v = 0.0
    for j in range(0, cols):
        let d = scratch[j] - mean
        v = v + d * d
    let inv_std = rsqrt(v * inv_cols + eps)
    barrier()
    out[start + col] = (scratch[col] - mean) * inv_std

fn main():
    let _a = 0
"#;
    let (_registry, kernel_ids) = setup_kernel(src);
    let kernel_id = *kernel_ids.get("layernorm").expect("layernorm kernel not found");

    let rows: usize = 4;
    let cols: usize = 8;
    let eps: f32 = 1e-5;
    let input_data: [f32; 32] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
        10.0, 10.1, 9.9, 10.2, 9.8, 10.3, 9.7, 10.4,
    ];
    let reference = layernorm_ref(&input_data, rows, cols, eps);

    malus_runtime::malus_cpu_compute_reset();

    let in_shape = [rows, cols];
    let in_handle = unsafe {
        malus_runtime::tensor_alloc_gpu(0, in_shape.as_ptr(), 2, input_data.as_ptr())
    };

    let out_shape = [rows, cols];
    let grid: [usize; 3] = [rows, 1, 1];
    let tg:   [usize; 3] = [cols, 1, 1];

    #[repr(C)]
    struct LnUniforms { cols: i32, inv_cols: f32, eps: f32 }
    let uniforms = LnUniforms { cols: cols as i32, inv_cols: 1.0 / cols as f32, eps };

    let out_handle = unsafe {
        malus_runtime::kernel_dispatch_v2(
            kernel_id,
            &in_handle as *const i64,
            1,
            grid.as_ptr(),
            tg.as_ptr(),
            out_shape.as_ptr(),
            2,
            0,
            &uniforms as *const LnUniforms as *const std::ffi::c_void,
            std::mem::size_of::<LnUniforms>(),
        )
    };
    malus_runtime::gpu_barrier();

    let cpu_count = malus_runtime::malus_cpu_compute_count();

    for i in 0..(rows * cols) {
        let got = unsafe { malus_runtime::malus_tensor_get_f32(out_handle, i as i64) };
        assert!(
            (got - reference[i]).abs() < 1e-5,
            "layernorm output[{i}]: got {got:.6}, expected {:.6}", reference[i]
        );
    }
    assert_eq!(cpu_count, 0, "CPU compute was invoked during GPU layernorm dispatch");

    malus_runtime::tensor_free(in_handle);
    malus_runtime::tensor_free(out_handle);
}

#[test]
fn test_v4_m24_gelu_gpu_counter_zero() {
    fn gelu_ref(input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| {
            let inner = 0.7978845608028654_f32 * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }).collect()
    }

    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let src = r#"
kernel gelu(input: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    let x = input[i]
    let x3 = x * x * x
    let inner = 0.7978845608028654 * (x + 0.044715 * x3)
    out[i] = 0.5 * x * (1.0 + tanh(inner))

fn main():
    let _a = 0
"#;
    let (_registry, kernel_ids) = setup_kernel(src);
    let kernel_id = *kernel_ids.get("gelu").expect("gelu kernel not found");

    let n: usize = 8;
    let input_data: [f32; 8] = [1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5, 3.0];
    let reference = gelu_ref(&input_data);

    malus_runtime::malus_cpu_compute_reset();

    let in_shape = [n];
    let in_handle = unsafe {
        malus_runtime::tensor_alloc_gpu(0, in_shape.as_ptr(), 1, input_data.as_ptr())
    };

    let out_shape = [n];
    let grid: [usize; 3] = [n, 1, 1];
    let tg:   [usize; 3] = [1, 1, 1];

    let out_handle = unsafe {
        malus_runtime::kernel_dispatch_v2(
            kernel_id,
            &in_handle as *const i64,
            1,
            grid.as_ptr(),
            tg.as_ptr(),
            out_shape.as_ptr(),
            1,
            0,
            std::ptr::null(),
            0,
        )
    };
    malus_runtime::gpu_barrier();

    let cpu_count = malus_runtime::malus_cpu_compute_count();

    for i in 0..n {
        let got = unsafe { malus_runtime::malus_tensor_get_f32(out_handle, i as i64) };
        assert!(
            (got - reference[i]).abs() < 1e-5,
            "gelu output[{i}]: got {got:.6}, expected {:.6}", reference[i]
        );
    }
    assert_eq!(cpu_count, 0, "CPU compute was invoked during GPU gelu dispatch");

    malus_runtime::tensor_free(in_handle);
    malus_runtime::tensor_free(out_handle);
}

// Mini-GPT training loop: n_embd=8, n_head=1, block_size=4, batch=2, vocab=8, 20 steps.
// Fixed token sequences so no read_file/rand_int needed. Asserts no panic (shape errors,
// NaN propagation, and RC bugs all manifest as panics/segfaults in this configuration).
#[test]
fn test_nanogpt_mini_trains_without_panic() {
    let src = r#"
struct AdamW:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32
    wd: f32

struct Block:
    ln1_w: Variable<f32>
    wq: Variable<f32>
    wk: Variable<f32>
    wv: Variable<f32>
    wo: Variable<f32>
    ln2_w: Variable<f32>
    w1: Variable<f32>
    w2: Variable<f32>

struct GPT:
    wte: Variable<f32>
    wpe: Variable<f32>
    ln_f: Variable<f32>
    lm_head: Variable<f32>

struct BlockState:
    ms_ln1: Tensor<f32>
    vs_ln1: Tensor<f32>
    ms_wq: Tensor<f32>
    vs_wq: Tensor<f32>
    ms_wk: Tensor<f32>
    vs_wk: Tensor<f32>
    ms_wv: Tensor<f32>
    vs_wv: Tensor<f32>
    ms_wo: Tensor<f32>
    vs_wo: Tensor<f32>
    ms_ln2: Tensor<f32>
    vs_ln2: Tensor<f32>
    ms_w1: Tensor<f32>
    vs_w1: Tensor<f32>
    ms_w2: Tensor<f32>
    vs_w2: Tensor<f32>

struct GPTState:
    ms_wte: Tensor<f32>
    vs_wte: Tensor<f32>
    ms_wpe: Tensor<f32>
    vs_wpe: Tensor<f32>
    ms_ln_f: Tensor<f32>
    vs_ln_f: Tensor<f32>
    ms_lm_head: Tensor<f32>
    vs_lm_head: Tensor<f32>

fn forward(gpt: GPT, blk: Block, toks: Tensor<i32>, B: i64, T: i64, C: i64) -> Variable<f32>:
    let mut pos_buf = buffer_i32(B * T)
    for b in range(B):
        for t in range(T):
            pos_buf[b * T + t] = t
    let pos_toks = freeze(pos_buf)
    let te = embedding(gpt.wte, toks)
    let pe = embedding(gpt.wpe, pos_toks)
    let x = te + pe
    let xn1 = layernorm(x, axis=1) * blk.ln1_w
    let Q = reshape(xn1 @ blk.wq, B, T, C)
    let K = reshape(xn1 @ blk.wk, B, T, C)
    let V = reshape(xn1 @ blk.wv, B, T, C)
    let Kt = permute(K, 0, 2, 1)
    let scale = variable(ones(1, 1) * 0.35355)
    let scores = (Q @ Kt) * scale
    let mask = causal_mask(T)
    let vmask = variable(mask)
    let masked = scores + vmask
    let attn = softmax(masked, axis=2)
    let att_out = reshape(attn @ V, B * T, C)
    let proj = att_out @ blk.wo
    let x2 = x + proj
    let xn2 = layernorm(x2, axis=1) * blk.ln2_w
    let hidden = gelu(xn2 @ blk.w1)
    let mlp_out = hidden @ blk.w2
    let x3 = x2 + mlp_out
    let xf = layernorm(x3, axis=1) * gpt.ln_f
    return xf @ gpt.lm_head

fn adamw_block(opt: AdamW, mut blk: Block, mut st: BlockState, t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    let g_ln1 = blk.ln1_w.grad + opt.wd * blk.ln1_w.data
    st.ms_ln1 = opt.beta1 * st.ms_ln1 + (1.0 - opt.beta1) * g_ln1
    st.vs_ln1 = opt.beta2 * st.vs_ln1 + (1.0 - opt.beta2) * g_ln1 * g_ln1
    let m_hat_ln1 = st.ms_ln1 / bc1
    let v_hat_ln1 = st.vs_ln1 / bc2
    blk.ln1_w = variable(blk.ln1_w.data - opt.lr * m_hat_ln1 / (sqrt(v_hat_ln1) + opt.eps))
    let g_wq = blk.wq.grad + opt.wd * blk.wq.data
    st.ms_wq = opt.beta1 * st.ms_wq + (1.0 - opt.beta1) * g_wq
    st.vs_wq = opt.beta2 * st.vs_wq + (1.0 - opt.beta2) * g_wq * g_wq
    let m_hat_wq = st.ms_wq / bc1
    let v_hat_wq = st.vs_wq / bc2
    blk.wq = variable(blk.wq.data - opt.lr * m_hat_wq / (sqrt(v_hat_wq) + opt.eps))
    let g_wk = blk.wk.grad + opt.wd * blk.wk.data
    st.ms_wk = opt.beta1 * st.ms_wk + (1.0 - opt.beta1) * g_wk
    st.vs_wk = opt.beta2 * st.vs_wk + (1.0 - opt.beta2) * g_wk * g_wk
    let m_hat_wk = st.ms_wk / bc1
    let v_hat_wk = st.vs_wk / bc2
    blk.wk = variable(blk.wk.data - opt.lr * m_hat_wk / (sqrt(v_hat_wk) + opt.eps))
    let g_wv = blk.wv.grad + opt.wd * blk.wv.data
    st.ms_wv = opt.beta1 * st.ms_wv + (1.0 - opt.beta1) * g_wv
    st.vs_wv = opt.beta2 * st.vs_wv + (1.0 - opt.beta2) * g_wv * g_wv
    let m_hat_wv = st.ms_wv / bc1
    let v_hat_wv = st.vs_wv / bc2
    blk.wv = variable(blk.wv.data - opt.lr * m_hat_wv / (sqrt(v_hat_wv) + opt.eps))
    let g_wo = blk.wo.grad + opt.wd * blk.wo.data
    st.ms_wo = opt.beta1 * st.ms_wo + (1.0 - opt.beta1) * g_wo
    st.vs_wo = opt.beta2 * st.vs_wo + (1.0 - opt.beta2) * g_wo * g_wo
    let m_hat_wo = st.ms_wo / bc1
    let v_hat_wo = st.vs_wo / bc2
    blk.wo = variable(blk.wo.data - opt.lr * m_hat_wo / (sqrt(v_hat_wo) + opt.eps))
    let g_ln2 = blk.ln2_w.grad + opt.wd * blk.ln2_w.data
    st.ms_ln2 = opt.beta1 * st.ms_ln2 + (1.0 - opt.beta1) * g_ln2
    st.vs_ln2 = opt.beta2 * st.vs_ln2 + (1.0 - opt.beta2) * g_ln2 * g_ln2
    let m_hat_ln2 = st.ms_ln2 / bc1
    let v_hat_ln2 = st.vs_ln2 / bc2
    blk.ln2_w = variable(blk.ln2_w.data - opt.lr * m_hat_ln2 / (sqrt(v_hat_ln2) + opt.eps))
    let g_w1 = blk.w1.grad + opt.wd * blk.w1.data
    st.ms_w1 = opt.beta1 * st.ms_w1 + (1.0 - opt.beta1) * g_w1
    st.vs_w1 = opt.beta2 * st.vs_w1 + (1.0 - opt.beta2) * g_w1 * g_w1
    let m_hat_w1 = st.ms_w1 / bc1
    let v_hat_w1 = st.vs_w1 / bc2
    blk.w1 = variable(blk.w1.data - opt.lr * m_hat_w1 / (sqrt(v_hat_w1) + opt.eps))
    let g_w2 = blk.w2.grad + opt.wd * blk.w2.data
    st.ms_w2 = opt.beta1 * st.ms_w2 + (1.0 - opt.beta1) * g_w2
    st.vs_w2 = opt.beta2 * st.vs_w2 + (1.0 - opt.beta2) * g_w2 * g_w2
    let m_hat_w2 = st.ms_w2 / bc1
    let v_hat_w2 = st.vs_w2 / bc2
    blk.w2 = variable(blk.w2.data - opt.lr * m_hat_w2 / (sqrt(v_hat_w2) + opt.eps))

fn adamw_gpt_params(opt: AdamW, mut gpt: GPT, mut st: GPTState, t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    let g_wte = gpt.wte.grad + opt.wd * gpt.wte.data
    st.ms_wte = opt.beta1 * st.ms_wte + (1.0 - opt.beta1) * g_wte
    st.vs_wte = opt.beta2 * st.vs_wte + (1.0 - opt.beta2) * g_wte * g_wte
    let m_hat_wte = st.ms_wte / bc1
    let v_hat_wte = st.vs_wte / bc2
    gpt.wte = variable(gpt.wte.data - opt.lr * m_hat_wte / (sqrt(v_hat_wte) + opt.eps))
    let g_wpe = gpt.wpe.grad + opt.wd * gpt.wpe.data
    st.ms_wpe = opt.beta1 * st.ms_wpe + (1.0 - opt.beta1) * g_wpe
    st.vs_wpe = opt.beta2 * st.vs_wpe + (1.0 - opt.beta2) * g_wpe * g_wpe
    let m_hat_wpe = st.ms_wpe / bc1
    let v_hat_wpe = st.vs_wpe / bc2
    gpt.wpe = variable(gpt.wpe.data - opt.lr * m_hat_wpe / (sqrt(v_hat_wpe) + opt.eps))
    let g_ln_f = gpt.ln_f.grad + opt.wd * gpt.ln_f.data
    st.ms_ln_f = opt.beta1 * st.ms_ln_f + (1.0 - opt.beta1) * g_ln_f
    st.vs_ln_f = opt.beta2 * st.vs_ln_f + (1.0 - opt.beta2) * g_ln_f * g_ln_f
    let m_hat_ln_f = st.ms_ln_f / bc1
    let v_hat_ln_f = st.vs_ln_f / bc2
    gpt.ln_f = variable(gpt.ln_f.data - opt.lr * m_hat_ln_f / (sqrt(v_hat_ln_f) + opt.eps))
    let g_lm = gpt.lm_head.grad + opt.wd * gpt.lm_head.data
    st.ms_lm_head = opt.beta1 * st.ms_lm_head + (1.0 - opt.beta1) * g_lm
    st.vs_lm_head = opt.beta2 * st.vs_lm_head + (1.0 - opt.beta2) * g_lm * g_lm
    let m_hat_lm = st.ms_lm_head / bc1
    let v_hat_lm = st.vs_lm_head / bc2
    gpt.lm_head = variable(gpt.lm_head.data - opt.lr * m_hat_lm / (sqrt(v_hat_lm) + opt.eps))

fn main():
    let C = 8
    let T = 4
    let B = 2
    let V = 8
    let C4 = 32
    let max_steps = 20
    let init_scale = ones(1, 1) * 0.02
    let ln1_w = variable(ones(C))
    let wq = variable(randn(C, C) * init_scale)
    let wk = variable(randn(C, C) * init_scale)
    let wv = variable(randn(C, C) * init_scale)
    let wo = variable(randn(C, C) * init_scale)
    let ln2_w = variable(ones(C))
    let w1 = variable(randn(C, C4) * init_scale)
    let w2 = variable(randn(C4, C) * init_scale)
    let blk = Block(ln1_w=ln1_w, wq=wq, wk=wk, wv=wv, wo=wo, ln2_w=ln2_w, w1=w1, w2=w2)
    let wte = variable(randn(V, C) * init_scale)
    let wpe = variable(randn(T, C) * init_scale)
    let ln_f = variable(ones(C))
    let lm_head = variable(randn(C, V) * init_scale)
    let gpt = GPT(wte=wte, wpe=wpe, ln_f=ln_f, lm_head=lm_head)
    let blk_st = BlockState(
        ms_ln1=zeros(C), vs_ln1=zeros(C),
        ms_wq=zeros(C, C), vs_wq=zeros(C, C),
        ms_wk=zeros(C, C), vs_wk=zeros(C, C),
        ms_wv=zeros(C, C), vs_wv=zeros(C, C),
        ms_wo=zeros(C, C), vs_wo=zeros(C, C),
        ms_ln2=zeros(C), vs_ln2=zeros(C),
        ms_w1=zeros(C, C4), vs_w1=zeros(C, C4),
        ms_w2=zeros(C4, C), vs_w2=zeros(C4, C),
    )
    let gpt_st = GPTState(
        ms_wte=zeros(V, C), vs_wte=zeros(V, C),
        ms_wpe=zeros(T, C), vs_wpe=zeros(T, C),
        ms_ln_f=zeros(C), vs_ln_f=zeros(C),
        ms_lm_head=zeros(C, V), vs_lm_head=zeros(C, V),
    )
    let opt = AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)
    for step in range(1, max_steps + 1):
        let mut x_buf = buffer_i32(B * T)
        let mut y_buf = buffer_i32(B * T)
        x_buf[0] = 0
        x_buf[1] = 1
        x_buf[2] = 2
        x_buf[3] = 3
        y_buf[0] = 1
        y_buf[1] = 2
        y_buf[2] = 3
        y_buf[3] = 4
        x_buf[4] = 4
        x_buf[5] = 5
        x_buf[6] = 6
        x_buf[7] = 7
        y_buf[4] = 5
        y_buf[5] = 6
        y_buf[6] = 7
        y_buf[7] = 0
        let x_toks = freeze(x_buf)
        let y_toks = freeze(y_buf)
        zero_grad(gpt.wte, gpt.wpe, gpt.ln_f, gpt.lm_head,
                  blk.ln1_w, blk.wq, blk.wk, blk.wv,
                  blk.wo, blk.ln2_w, blk.w1, blk.w2)
        let logits = forward(gpt, blk, x_toks, B, T, C)
        let loss = cross_entropy(logits, y_toks)
        backward(loss)
        adamw_block(opt, blk, blk_st, step)
        adamw_gpt_params(opt, gpt, gpt_st, step)
"#;
    run_metal_src(src);
}

/// M25 Done-When gate: nanoGPT forward pass must not increment the CPU compute counter.
/// Covers: embedding, layernorm, softmax, gelu, cross_entropy, permute, broadcast +/*, matmul.
#[test]
fn test_nanogpt_forward_zero_cpu_compute() {
    let src = r#"
struct Block:
    ln1_w: Variable<f32>
    wq: Variable<f32>
    wk: Variable<f32>
    wv: Variable<f32>
    wo: Variable<f32>
    ln2_w: Variable<f32>
    w1: Variable<f32>
    w2: Variable<f32>

struct GPT:
    wte: Variable<f32>
    wpe: Variable<f32>
    ln_f: Variable<f32>
    lm_head: Variable<f32>

fn forward(gpt: GPT, blk: Block, toks: Tensor<i32>, B: i64, T: i64, C: i64) -> Variable<f32>:
    let mut pos_buf = buffer_i32(B * T)
    for b in range(B):
        for t in range(T):
            pos_buf[b * T + t] = t
    let pos_toks = freeze(pos_buf)
    let te = embedding(gpt.wte, toks)
    let pe = embedding(gpt.wpe, pos_toks)
    let x = te + pe
    let xn1 = layernorm(x, axis=1) * blk.ln1_w
    let Q = reshape(xn1 @ blk.wq, B, T, C)
    let K = reshape(xn1 @ blk.wk, B, T, C)
    let V = reshape(xn1 @ blk.wv, B, T, C)
    let Kt = permute(K, 0, 2, 1)
    let scale = variable(ones(1, 1) * 0.35355)
    let scores = (Q @ Kt) * scale
    let mask = causal_mask(T)
    let vmask = variable(mask)
    let masked = scores + vmask
    let attn = softmax(masked, axis=2)
    let att_out = reshape(attn @ V, B * T, C)
    let proj = att_out @ blk.wo
    let x2 = x + proj
    let xn2 = layernorm(x2, axis=1) * blk.ln2_w
    let hidden = gelu(xn2 @ blk.w1)
    let mlp_out = hidden @ blk.w2
    let x3 = x2 + mlp_out
    let xf = layernorm(x3, axis=1) * gpt.ln_f
    return xf @ gpt.lm_head

fn main():
    let B = 2
    let T = 4
    let C = 8
    let V = 10
    let init_scale = 0.02
    let ln1_w = variable(ones(C))
    let wq = variable(randn(C, C) * init_scale)
    let wk = variable(randn(C, C) * init_scale)
    let wv = variable(randn(C, C) * init_scale)
    let wo = variable(randn(C, C) * init_scale)
    let ln2_w = variable(ones(C))
    let w1 = variable(randn(C, C) * init_scale)
    let w2 = variable(randn(C, C) * init_scale)
    let blk = Block(ln1_w=ln1_w, wq=wq, wk=wk, wv=wv, wo=wo, ln2_w=ln2_w, w1=w1, w2=w2)
    let wte = variable(randn(V, C) * init_scale)
    let wpe = variable(randn(T, C) * init_scale)
    let ln_f = variable(ones(C))
    let lm_head = variable(randn(C, V) * init_scale)
    let gpt = GPT(wte=wte, wpe=wpe, ln_f=ln_f, lm_head=lm_head)
    let mut x_buf = buffer_i32(B * T)
    x_buf[0] = 0
    x_buf[1] = 1
    x_buf[2] = 2
    x_buf[3] = 3
    x_buf[4] = 4
    x_buf[5] = 5
    x_buf[6] = 6
    x_buf[7] = 7
    let mut y_buf = buffer_i32(B * T)
    y_buf[0] = 1
    y_buf[1] = 2
    y_buf[2] = 3
    y_buf[3] = 4
    y_buf[4] = 5
    y_buf[5] = 6
    y_buf[6] = 7
    y_buf[7] = 0
    let x_toks = freeze(x_buf)
    let y_toks = freeze(y_buf)
    let logits = forward(gpt, blk, x_toks, B, T, C)
    let _loss = cross_entropy(logits, y_toks)
"#;
    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut user_program = parse(malus_syntax::FileId(0), src).expect("parse failed");
    let mut stdlib = malus_stdlib::stdlib_items();
    stdlib.extend(user_program.items.drain(..));
    let program = malus_syntax::ast::Program { items: stdlib };
    let aliases = std::collections::HashMap::new();
    let typed = check(&program, &aliases).expect("type check failed");
    let (registry, kernel_ids) =
        malus_codegen_gpu::compile_kernels(&typed).expect("kernel compilation failed");
    malus_runtime::runtime_init(&registry.into_hashmap());
    let symbols = real_symbols();
    malus_runtime::malus_cpu_compute_reset();
    compile_and_run(&typed, &symbols, &kernel_ids).expect("JIT compile and run failed");
    let cpu_count = malus_runtime::malus_cpu_compute_count();
    assert_eq!(cpu_count, 0, "M25: nanoGPT forward pass must use 0 CPU compute ops, got {}", cpu_count);
}

// ── M26 canonical gate tests (ADR-0031/0032) ─────────────────────────────────

const NANOGPT_BOILERPLATE: &str = r#"
struct AdamW:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32
    wd: f32

struct Block:
    ln1_w: Variable<f32>
    wq: Variable<f32>
    wk: Variable<f32>
    wv: Variable<f32>
    wo: Variable<f32>
    ln2_w: Variable<f32>
    w1: Variable<f32>
    w2: Variable<f32>

struct GPT:
    wte: Variable<f32>
    wpe: Variable<f32>
    ln_f: Variable<f32>
    lm_head: Variable<f32>

struct BlockState:
    ms_ln1: Tensor<f32>
    vs_ln1: Tensor<f32>
    ms_wq: Tensor<f32>
    vs_wq: Tensor<f32>
    ms_wk: Tensor<f32>
    vs_wk: Tensor<f32>
    ms_wv: Tensor<f32>
    vs_wv: Tensor<f32>
    ms_wo: Tensor<f32>
    vs_wo: Tensor<f32>
    ms_ln2: Tensor<f32>
    vs_ln2: Tensor<f32>
    ms_w1: Tensor<f32>
    vs_w1: Tensor<f32>
    ms_w2: Tensor<f32>
    vs_w2: Tensor<f32>

struct GPTState:
    ms_wte: Tensor<f32>
    vs_wte: Tensor<f32>
    ms_wpe: Tensor<f32>
    vs_wpe: Tensor<f32>
    ms_ln_f: Tensor<f32>
    vs_ln_f: Tensor<f32>
    ms_lm_head: Tensor<f32>
    vs_lm_head: Tensor<f32>

fn forward(gpt: GPT, blk: Block, toks: Tensor<i32>, B: i64, T: i64, C: i64) -> Variable<f32>:
    let mut pos_buf = buffer_i32(B * T)
    for b in range(B):
        for t in range(T):
            pos_buf[b * T + t] = t
    let pos_toks = freeze(pos_buf)
    let te = embedding(gpt.wte, toks)
    let pe = embedding(gpt.wpe, pos_toks)
    let x = te + pe
    let xn1 = layernorm(x, axis=1) * blk.ln1_w
    let Q = reshape(xn1 @ blk.wq, B, T, C)
    let K = reshape(xn1 @ blk.wk, B, T, C)
    let V = reshape(xn1 @ blk.wv, B, T, C)
    let Kt = permute(K, 0, 2, 1)
    let scale = variable(ones(1, 1) * 0.35355)
    let scores = (Q @ Kt) * scale
    let mask = causal_mask(T)
    let vmask = variable(mask)
    let masked = scores + vmask
    let attn = softmax(masked, axis=2)
    let att_out = reshape(attn @ V, B * T, C)
    let proj = att_out @ blk.wo
    let x2 = x + proj
    let xn2 = layernorm(x2, axis=1) * blk.ln2_w
    let hidden = gelu(xn2 @ blk.w1)
    let mlp_out = hidden @ blk.w2
    let x3 = x2 + mlp_out
    let xf = layernorm(x3, axis=1) * gpt.ln_f
    return xf @ gpt.lm_head

fn adamw_block(opt: AdamW, mut blk: Block, mut st: BlockState, t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    let g_ln1 = blk.ln1_w.grad + opt.wd * blk.ln1_w.data
    st.ms_ln1 = opt.beta1 * st.ms_ln1 + (1.0 - opt.beta1) * g_ln1
    st.vs_ln1 = opt.beta2 * st.vs_ln1 + (1.0 - opt.beta2) * g_ln1 * g_ln1
    let m_hat_ln1 = st.ms_ln1 / bc1
    let v_hat_ln1 = st.vs_ln1 / bc2
    blk.ln1_w = variable(blk.ln1_w.data - opt.lr * m_hat_ln1 / (sqrt(v_hat_ln1) + opt.eps))
    let g_wq = blk.wq.grad + opt.wd * blk.wq.data
    st.ms_wq = opt.beta1 * st.ms_wq + (1.0 - opt.beta1) * g_wq
    st.vs_wq = opt.beta2 * st.vs_wq + (1.0 - opt.beta2) * g_wq * g_wq
    let m_hat_wq = st.ms_wq / bc1
    let v_hat_wq = st.vs_wq / bc2
    blk.wq = variable(blk.wq.data - opt.lr * m_hat_wq / (sqrt(v_hat_wq) + opt.eps))
    let g_wk = blk.wk.grad + opt.wd * blk.wk.data
    st.ms_wk = opt.beta1 * st.ms_wk + (1.0 - opt.beta1) * g_wk
    st.vs_wk = opt.beta2 * st.vs_wk + (1.0 - opt.beta2) * g_wk * g_wk
    let m_hat_wk = st.ms_wk / bc1
    let v_hat_wk = st.vs_wk / bc2
    blk.wk = variable(blk.wk.data - opt.lr * m_hat_wk / (sqrt(v_hat_wk) + opt.eps))
    let g_wv = blk.wv.grad + opt.wd * blk.wv.data
    st.ms_wv = opt.beta1 * st.ms_wv + (1.0 - opt.beta1) * g_wv
    st.vs_wv = opt.beta2 * st.vs_wv + (1.0 - opt.beta2) * g_wv * g_wv
    let m_hat_wv = st.ms_wv / bc1
    let v_hat_wv = st.vs_wv / bc2
    blk.wv = variable(blk.wv.data - opt.lr * m_hat_wv / (sqrt(v_hat_wv) + opt.eps))
    let g_wo = blk.wo.grad + opt.wd * blk.wo.data
    st.ms_wo = opt.beta1 * st.ms_wo + (1.0 - opt.beta1) * g_wo
    st.vs_wo = opt.beta2 * st.vs_wo + (1.0 - opt.beta2) * g_wo * g_wo
    let m_hat_wo = st.ms_wo / bc1
    let v_hat_wo = st.vs_wo / bc2
    blk.wo = variable(blk.wo.data - opt.lr * m_hat_wo / (sqrt(v_hat_wo) + opt.eps))
    let g_ln2 = blk.ln2_w.grad + opt.wd * blk.ln2_w.data
    st.ms_ln2 = opt.beta1 * st.ms_ln2 + (1.0 - opt.beta1) * g_ln2
    st.vs_ln2 = opt.beta2 * st.vs_ln2 + (1.0 - opt.beta2) * g_ln2 * g_ln2
    let m_hat_ln2 = st.ms_ln2 / bc1
    let v_hat_ln2 = st.vs_ln2 / bc2
    blk.ln2_w = variable(blk.ln2_w.data - opt.lr * m_hat_ln2 / (sqrt(v_hat_ln2) + opt.eps))
    let g_w1 = blk.w1.grad + opt.wd * blk.w1.data
    st.ms_w1 = opt.beta1 * st.ms_w1 + (1.0 - opt.beta1) * g_w1
    st.vs_w1 = opt.beta2 * st.vs_w1 + (1.0 - opt.beta2) * g_w1 * g_w1
    let m_hat_w1 = st.ms_w1 / bc1
    let v_hat_w1 = st.vs_w1 / bc2
    blk.w1 = variable(blk.w1.data - opt.lr * m_hat_w1 / (sqrt(v_hat_w1) + opt.eps))
    let g_w2 = blk.w2.grad + opt.wd * blk.w2.data
    st.ms_w2 = opt.beta1 * st.ms_w2 + (1.0 - opt.beta1) * g_w2
    st.vs_w2 = opt.beta2 * st.vs_w2 + (1.0 - opt.beta2) * g_w2 * g_w2
    let m_hat_w2 = st.ms_w2 / bc1
    let v_hat_w2 = st.vs_w2 / bc2
    blk.w2 = variable(blk.w2.data - opt.lr * m_hat_w2 / (sqrt(v_hat_w2) + opt.eps))

fn adamw_gpt_params(opt: AdamW, mut gpt: GPT, mut st: GPTState, t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    let g_wte = gpt.wte.grad + opt.wd * gpt.wte.data
    st.ms_wte = opt.beta1 * st.ms_wte + (1.0 - opt.beta1) * g_wte
    st.vs_wte = opt.beta2 * st.vs_wte + (1.0 - opt.beta2) * g_wte * g_wte
    let m_hat_wte = st.ms_wte / bc1
    let v_hat_wte = st.vs_wte / bc2
    gpt.wte = variable(gpt.wte.data - opt.lr * m_hat_wte / (sqrt(v_hat_wte) + opt.eps))
    let g_wpe = gpt.wpe.grad + opt.wd * gpt.wpe.data
    st.ms_wpe = opt.beta1 * st.ms_wpe + (1.0 - opt.beta1) * g_wpe
    st.vs_wpe = opt.beta2 * st.vs_wpe + (1.0 - opt.beta2) * g_wpe * g_wpe
    let m_hat_wpe = st.ms_wpe / bc1
    let v_hat_wpe = st.vs_wpe / bc2
    gpt.wpe = variable(gpt.wpe.data - opt.lr * m_hat_wpe / (sqrt(v_hat_wpe) + opt.eps))
    let g_ln_f = gpt.ln_f.grad + opt.wd * gpt.ln_f.data
    st.ms_ln_f = opt.beta1 * st.ms_ln_f + (1.0 - opt.beta1) * g_ln_f
    st.vs_ln_f = opt.beta2 * st.vs_ln_f + (1.0 - opt.beta2) * g_ln_f * g_ln_f
    let m_hat_ln_f = st.ms_ln_f / bc1
    let v_hat_ln_f = st.vs_ln_f / bc2
    gpt.ln_f = variable(gpt.ln_f.data - opt.lr * m_hat_ln_f / (sqrt(v_hat_ln_f) + opt.eps))
    let g_lm = gpt.lm_head.grad + opt.wd * gpt.lm_head.data
    st.ms_lm_head = opt.beta1 * st.ms_lm_head + (1.0 - opt.beta1) * g_lm
    st.vs_lm_head = opt.beta2 * st.vs_lm_head + (1.0 - opt.beta2) * g_lm * g_lm
    let m_hat_lm = st.ms_lm_head / bc1
    let v_hat_lm = st.vs_lm_head / bc2
    gpt.lm_head = variable(gpt.lm_head.data - opt.lr * m_hat_lm / (sqrt(v_hat_lm) + opt.eps))
"#;

fn nanogpt_train_src(max_steps: i64, record_loss_at_last_step: bool) -> String {
    // gpu_barrier() is only inserted by CTMM before a *drop* of a pending
    // tensor, never before a plain read — loss (a Variable, RC-managed, not
    // CTMM-static-dropped) can be read before its GPU computation is
    // visible. A throwaway static-drop Tensor read reliably forces the
    // flush (see examples/gradient_check.ml's __flush()).
    let record_call = if record_loss_at_last_step {
        "        if step == max_steps:\n            let flush_t = ones(1)\n            let flush_v = flush_t[0]\n            record_diff(loss.data[0])\n"
    } else {
        ""
    };
    format!(
        r#"{boilerplate}
fn main():
    let C = 8
    let T = 4
    let B = 2
    let V = 8
    let C4 = 32
    let max_steps = {max_steps}
    let init_scale = ones(1, 1) * 0.02
    let ln1_w = variable(ones(C))
    let wq = variable(randn(C, C) * init_scale)
    let wk = variable(randn(C, C) * init_scale)
    let wv = variable(randn(C, C) * init_scale)
    let wo = variable(randn(C, C) * init_scale)
    let ln2_w = variable(ones(C))
    let w1 = variable(randn(C, C4) * init_scale)
    let w2 = variable(randn(C4, C) * init_scale)
    let blk = Block(ln1_w=ln1_w, wq=wq, wk=wk, wv=wv, wo=wo, ln2_w=ln2_w, w1=w1, w2=w2)
    let wte = variable(randn(V, C) * init_scale)
    let wpe = variable(randn(T, C) * init_scale)
    let ln_f = variable(ones(C))
    let lm_head = variable(randn(C, V) * init_scale)
    let gpt = GPT(wte=wte, wpe=wpe, ln_f=ln_f, lm_head=lm_head)
    let blk_st = BlockState(
        ms_ln1=zeros(C), vs_ln1=zeros(C),
        ms_wq=zeros(C, C), vs_wq=zeros(C, C),
        ms_wk=zeros(C, C), vs_wk=zeros(C, C),
        ms_wv=zeros(C, C), vs_wv=zeros(C, C),
        ms_wo=zeros(C, C), vs_wo=zeros(C, C),
        ms_ln2=zeros(C), vs_ln2=zeros(C),
        ms_w1=zeros(C, C4), vs_w1=zeros(C, C4),
        ms_w2=zeros(C4, C), vs_w2=zeros(C4, C),
    )
    let gpt_st = GPTState(
        ms_wte=zeros(V, C), vs_wte=zeros(V, C),
        ms_wpe=zeros(T, C), vs_wpe=zeros(T, C),
        ms_ln_f=zeros(C), vs_ln_f=zeros(C),
        ms_lm_head=zeros(C, V), vs_lm_head=zeros(C, V),
    )
    let opt = AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)
    for step in range(1, max_steps + 1):
        let mut x_buf = buffer_i32(B * T)
        let mut y_buf = buffer_i32(B * T)
        x_buf[0] = 0
        x_buf[1] = 1
        x_buf[2] = 2
        x_buf[3] = 3
        y_buf[0] = 1
        y_buf[1] = 2
        y_buf[2] = 3
        y_buf[3] = 4
        x_buf[4] = 4
        x_buf[5] = 5
        x_buf[6] = 6
        x_buf[7] = 7
        y_buf[4] = 5
        y_buf[5] = 6
        y_buf[6] = 7
        y_buf[7] = 0
        let x_toks = freeze(x_buf)
        let y_toks = freeze(y_buf)
        zero_grad(gpt.wte, gpt.wpe, gpt.ln_f, gpt.lm_head,
                  blk.ln1_w, blk.wq, blk.wk, blk.wv,
                  blk.wo, blk.ln2_w, blk.w1, blk.w2)
        let logits = forward(gpt, blk, x_toks, B, T, C)
        let loss = cross_entropy(logits, y_toks)
{record_call}        backward(loss)
        adamw_block(opt, blk, blk_st, step)
        adamw_gpt_params(opt, gpt, gpt_st, step)
"#,
        boilerplate = NANOGPT_BOILERPLATE,
        max_steps = max_steps,
        record_call = record_call,
    )
}

/// M26 canonical gate: a full nanoGPT train step (forward + backward(loss) +
/// AdamW optimizer update) must dispatch zero CPU compute. This is the V4
/// north-star property — every backward kernel (ADR-0032) plus the existing
/// M25 forward kernels combine to make the canonical gate build's hot path
/// structurally GPU-only (cfg(cpu_fallback) makes the alternative not even
/// compile, see ADR-0031).
#[test]
fn test_v4_m3_full_step_zero_cpu_compute() {
    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let src = nanogpt_train_src(2, false);
    let user_program = parse(malus_syntax::FileId(0), &src).expect("parse failed");
    let mut stdlib = malus_stdlib::stdlib_items();
    stdlib.extend(user_program.items.into_iter());
    let program = malus_syntax::ast::Program { items: stdlib };
    let aliases = HashMap::new();
    let typed = check(&program, &aliases).expect("type check failed");
    let (registry, kernel_ids) =
        malus_codegen_gpu::compile_kernels(&typed).expect("kernel compilation failed");
    malus_runtime::runtime_init(&registry.into_hashmap());
    let symbols = real_symbols();
    malus_runtime::malus_cpu_compute_reset();
    compile_and_run(&typed, &symbols, &kernel_ids).expect("JIT compile and run failed");
    let cpu_count = malus_runtime::malus_cpu_compute_count();
    assert_eq!(
        cpu_count, 0,
        "M26 canonical gate: full nanoGPT train step (forward+backward+AdamW) \
         must use 0 CPU compute ops, got {cpu_count}"
    );
}

/// M26 done-when: every op's analytic backward() gradient matches a
/// finite-difference numeric gradient within 1e-3, across every op the
/// nanoGPT forward pass exercises (matmul, softmax, layernorm, gelu,
/// embedding, cross-entropy, elementwise add/mul) — see
/// examples/gradient_check.ml.
#[test]
fn test_gradient_check_all_ops() {
    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    malus_runtime::malus_gradcheck_reset();
    let src = include_str!("../../../examples/gradient_check.ml");
    run_metal_src(src);
    let max_diff = malus_runtime::malus_gradcheck_max_diff();
    assert!(
        max_diff < 1e-3,
        "gradient check: max |analytic - numeric| = {max_diff}, expected < 1e-3"
    );
}

/// M26 done-when: training loss decreases (deterministic Philox seed, no
/// user-supplied RNG seed — see ADR-0024) over a 10-step regression run.
/// Two independent compiles (truncated at step 1 and step 10) are
/// deterministically identical up to the shared prefix, so loss[0] and
/// loss[9] are directly comparable without needing a mid-run readback hook.
#[test]
fn test_nanogpt_loss_decreases() {
    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Two separate type-checked programs (1-step vs 10-step main()) but a
    // single runtime_init: the kernel set is identical between them (same
    // stdlib + same boilerplate; kernel IDs are determined solely by the
    // combined item list, which doesn't depend on max_steps), so a second
    // full pipeline rebuild is both redundant and, empirically, unreliable
    // when it immediately follows another compile_and_run in this process.
    fn typecheck(max_steps: i64) -> malus_sema::TypedProgram {
        let src = nanogpt_train_src(max_steps, true);
        let mut user_program = parse(malus_syntax::FileId(0), &src).expect("parse failed");
        let mut stdlib = malus_stdlib::stdlib_items();
        stdlib.extend(user_program.items.drain(..));
        let program = malus_syntax::ast::Program { items: stdlib };
        let aliases = HashMap::new();
        check(&program, &aliases).expect("type check failed")
    }

    let typed_1 = typecheck(1);
    let typed_10 = typecheck(10);
    let (registry, kernel_ids) =
        malus_codegen_gpu::compile_kernels(&typed_1).expect("kernel compilation failed");
    malus_runtime::runtime_init(&registry.into_hashmap());
    let symbols = real_symbols();

    malus_runtime::malus_gradcheck_reset();
    compile_and_run(&typed_1, &symbols, &kernel_ids).expect("JIT compile and run failed");
    let loss_first = malus_runtime::malus_gradcheck_max_diff();

    malus_runtime::malus_gradcheck_reset();
    compile_and_run(&typed_10, &symbols, &kernel_ids).expect("JIT compile and run failed");
    let loss_last = malus_runtime::malus_gradcheck_max_diff();

    assert!(
        loss_last < loss_first * 0.9,
        "loss did not decrease by the expected margin: loss[0]={loss_first}, loss[9]={loss_last}"
    );
}
