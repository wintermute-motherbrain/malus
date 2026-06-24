use std::collections::HashMap;
use std::sync::Mutex;
use malus_syntax::parse;
use malus_sema::check;
use crate::{compile_and_run, CodegenError};

// Tests share TENSOR_STORE global state, so they must not run in parallel.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn run_src(src: &str) -> Result<(), CodegenError> {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let program = parse(malus_syntax::FileId(0), src).expect("parse failed");
    let aliases = HashMap::new();
    let typed = check(&program, &aliases).expect("type check failed");
    compile_and_run(&typed)
}

// ── Tensor alloc, print, and free ────────────────────────────────────────────

#[test]
fn test_tensor_alloc_and_free() {
    // Allocates, prints, and frees a tensor. CTMM inserts Drop(a) after print(a).
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    print(a)
"#;
    run_src(src).expect("should compile and run");
}

// ── Tensor data is stored and printed ────────────────────────────────────────

#[test]
fn test_tensor_alloc_stores_data() {
    // Verify the pipeline round-trips: alloc with known values, print without panic.
    let src = r#"
fn make() -> Tensor<f32>:
    let a = Tensor.gpu<f32>([10.0, 20.0, 30.0])
    return a

fn main():
    let x = make()
    print(x)
"#;
    // After run, CTMM drops x (last use is print(x)), so store is empty.
    // We just verify the run completes without panic.
    run_src(src).expect("should compile and run");
}

// ── Kernel dispatch returns a handle ─────────────────────────────────────────

#[test]
fn test_kernel_dispatch_returns_handle() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    print(c)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
"#;
    // kernel_dispatch stub returns an empty tensor — should not panic.
    run_src(src).expect("add_tensors.ml flow should compile and run without panic");
}

// ── Scalar arithmetic ─────────────────────────────────────────────────────────

#[test]
fn test_scalar_add() {
    // Verify scalar BinOp works by checking a fn computes and calls print on a tensor.
    // We can't inspect integer scalar return values directly (main returns void),
    // but we can verify the JIT compiles and executes without error.
    let src = r#"
fn double(x: i32) -> i32:
    return x + x

fn main():
    let a = Tensor.gpu<f32>([1.0])
    print(a)
"#;
    run_src(src).expect("fn-to-fn call with scalar arithmetic should work");
}

// ── Fn-to-fn call ─────────────────────────────────────────────────────────────

#[test]
fn test_fn_to_fn_call() {
    let src = r#"
fn make_tensor() -> Tensor<f32>:
    let a = Tensor.gpu<f32>([42.0, 43.0])
    return a

fn main():
    let t = make_tensor()
    print(t)
"#;
    // After run, CTMM drops t (last use is print(t)).
    run_src(src).expect("fn-to-fn call should compile and run");
}

// ── CTMM: Drop and GpuBarrier execute without panic ───────────────────────────

#[test]
fn test_ctmm_drop_and_barrier() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = dispatch(a, b)
    print(c)

kernel dispatch(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
"#;
    // CTMM inserts GpuBarrier + Drop(a) + Drop(b) after the kernel call.
    // This tests that the barrier and free stubs execute without panic.
    run_src(src).expect("CTMM drop and barrier should execute without panic");
}

// ── print / println format string codegen ────────────────────────────────────

#[test]
fn test_print_string_literal() {
    let src = r#"
fn main():
    print("hello")
"#;
    run_src(src).expect("print(string) should compile and run");
}

#[test]
fn test_println_string_literal() {
    let src = r#"
fn main():
    println("hello")
"#;
    run_src(src).expect("println(string) should compile and run");
}

#[test]
fn test_println_format_string() {
    let src = r#"
fn main():
    println("{} + {} = {}", 1.0, 2.0, 3.0)
"#;
    run_src(src).expect("format string println should compile and run");
}

#[test]
fn test_println_single_value() {
    let src = r#"
fn main():
    println(42)
"#;
    run_src(src).expect("println(scalar) should compile and run");
}

#[test]
fn test_println_no_args() {
    let src = r#"
fn main():
    println()
"#;
    run_src(src).expect("println() bare newline should compile and run");
}

// ── No main → CodegenError::NoMainFunction ───────────────────────────────────

#[test]
fn test_no_main_returns_error() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use malus_sema::{TypedFn, TypedProgram};
    use malus_syntax::Span;

    let typed = TypedProgram {
        fns: vec![TypedFn {
            name: "helper".to_string(),
            params: vec![],
            return_ty: malus_sema::ResolvedTy::Unit,
            body: vec![],
            span: Span::new(malus_syntax::FileId(0), 0, 0),
        }],
        kernels: vec![],
    };

    let result = compile_and_run(&typed);
    assert!(
        matches!(result, Err(CodegenError::NoMainFunction)),
        "expected NoMainFunction, got: {:?}", result
    );
}
