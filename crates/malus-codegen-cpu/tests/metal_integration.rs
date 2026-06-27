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
