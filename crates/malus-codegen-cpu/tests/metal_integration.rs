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
        tensor_transpose:       malus_runtime::tensor_transpose,
        tensor_sum:             malus_runtime::tensor_sum,
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
        tensor_broadcast_add:   malus_runtime::tensor_broadcast_add,
        tensor_broadcast_sub:   malus_runtime::tensor_broadcast_sub,
        tensor_broadcast_mul:   malus_runtime::tensor_broadcast_mul,
        tensor_broadcast_div:   malus_runtime::tensor_broadcast_div,
        tensor_reduce_sum_axis:  malus_runtime::tensor_reduce_sum_axis,
        tensor_reduce_mean_axis: malus_runtime::tensor_reduce_mean_axis,
        tensor_reduce_max_axis:  malus_runtime::tensor_reduce_max_axis,
        tensor_reduce_var_axis:  malus_runtime::tensor_reduce_var_axis,
        tape_record_reduce:     malus_runtime::tape_record_reduce,
        tensor_reshape:         malus_runtime::tensor_reshape,
        tensor_permute:         malus_runtime::tensor_permute,
        tape_record_perm:       malus_runtime::tape_record_perm,
        // M18 transformer stdlib.
        tensor_softmax_axis:       malus_runtime::tensor_softmax_axis,
        tensor_layernorm_axis:     malus_runtime::tensor_layernorm_axis,
        tensor_gelu:               malus_runtime::tensor_gelu,
        tensor_cross_entropy:      malus_runtime::tensor_cross_entropy,
        tensor_causal_mask:        malus_runtime::tensor_causal_mask,
        tape_record_layernorm:     malus_runtime::tape_record_layernorm,
        tape_record_cross_entropy: malus_runtime::tape_record_cross_entropy,
        // M19 embeddings + randn.
        tensor_embedding:          malus_runtime::tensor_embedding,
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
    }
}

fn run_metal_src(src: &str) {
    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let program = parse(malus_syntax::FileId(0), src).expect("parse failed");
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

// V4 M23 CI gate: softmax_row dispatched via kernel_dispatch_v2 produces
// numerically correct output and triggers zero CPU-compute counter increments.
#[test]
fn test_v4_m23_softmax_row_gpu_counter_zero() {
    // Pure-Rust reference — no malus_runtime calls, so no counter pollution.
    fn softmax_ref(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row = &input[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for c in 0..cols {
                out[r * cols + c] = exps[c] / sum;
            }
        }
        out
    }

    let _guard = METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let rows: usize = 4;
    let cols: usize = 8;
    let input_data: [f32; 32] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6,
    ];
    let reference = softmax_ref(&input_data, rows, cols);

    malus_runtime::register_m23_softmax_row_kernel();
    malus_runtime::malus_cpu_compute_reset();

    let in_shape = [rows, cols];
    let in_handle = unsafe {
        malus_runtime::tensor_alloc_gpu(0, in_shape.as_ptr(), 2, input_data.as_ptr())
    };

    let out_shape = [rows, cols];
    let grid: [usize; 3] = [rows, 1, 1];
    let tg: [usize; 3]   = [cols, 1, 1];
    let uniforms: u32 = cols as u32;

    let out_handle = unsafe {
        malus_runtime::kernel_dispatch_v2(
            malus_runtime::M23_SOFTMAX_ROW_KERNEL_ID,
            &in_handle as *const i64,
            1,
            grid.as_ptr(),
            tg.as_ptr(),
            out_shape.as_ptr(),
            2,
            0, // f32 dtype_tag
            &uniforms as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
        )
    };
    malus_runtime::gpu_barrier();

    // Snapshot the count before element reads so the assertion is logically tight.
    let cpu_count = malus_runtime::malus_cpu_compute_count();

    for i in 0..(rows * cols) {
        let got = unsafe { malus_runtime::malus_tensor_get_f32(out_handle, i as i64) };
        assert!(
            (got - reference[i]).abs() < 1e-5,
            "output[{i}]: got {got:.6}, expected {:.6}",
            reference[i]
        );
    }

    assert_eq!(cpu_count, 0, "CPU compute was invoked during GPU softmax dispatch");

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
