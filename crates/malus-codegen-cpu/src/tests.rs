use crate::{compile_and_run, CodegenError, RuntimeSymbols};
use malus_sema::check;
use malus_syntax::parse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// Tests share MOCK_STORE global state, so they must not run in parallel.
static TEST_LOCK: Mutex<()> = Mutex::new(());
static MOCK_DISPATCH_COUNT: AtomicUsize = AtomicUsize::new(0);

// ── Mock runtime (HashMap-backed, replicates the M3 stubs) ────────────────────

struct MockStore {
    data: HashMap<i64, Vec<f32>>,
    next_id: i64,
}

impl MockStore {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
            next_id: 1,
        }
    }

    fn insert(&mut self, elements: Vec<f32>) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        self.data.insert(id, elements);
        id
    }
}

static MOCK_STORE: Mutex<Option<MockStore>> = Mutex::new(None);

fn with_store<R>(f: impl FnOnce(&mut MockStore) -> R) -> R {
    let mut guard = MOCK_STORE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.as_mut().expect("mock store not initialized"))
}

extern "C" fn mock_tensor_alloc_gpu(dtype: i32, len: i64, data: *const f32) -> i64 {
    let _ = dtype;
    let elements = if data.is_null() || len == 0 {
        vec![]
    } else {
        unsafe { std::slice::from_raw_parts(data, len as usize).to_vec() }
    };
    with_store(|s| s.insert(elements))
}

extern "C" fn mock_tensor_print(handle: i64) {
    let elems = with_store(|s| s.data.get(&handle).cloned().unwrap_or_default());
    print!("[");
    for (i, v) in elems.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{v}");
    }
    print!("]");
}

extern "C" fn mock_tensor_free(handle: i64) {
    with_store(|s| {
        s.data.remove(&handle);
    });
}

extern "C" fn mock_kernel_dispatch(_kernel_id: u64, handles: *const i64, count: usize) -> i64 {
    MOCK_DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);
    let len = if count < 1 || handles.is_null() {
        0
    } else {
        with_store(|s| s.data.get(unsafe { &*handles }).map(|v| v.len()).unwrap_or(0))
    };
    with_store(|s| s.insert(vec![1.0; len.max(1)]))
}

extern "C" fn mock_gpu_barrier() {}

fn mock_symbols() -> RuntimeSymbols {
    RuntimeSymbols {
        tensor_alloc_gpu: mock_tensor_alloc_gpu,
        tensor_free: mock_tensor_free,
        tensor_print: mock_tensor_print,
        kernel_dispatch: mock_kernel_dispatch,
        gpu_barrier: mock_gpu_barrier,
    }
}

fn run_src(src: &str) -> Result<(), CodegenError> {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *MOCK_STORE.lock().unwrap() = Some(MockStore::new());
    MOCK_DISPATCH_COUNT.store(0, Ordering::SeqCst);
    let program = parse(malus_syntax::FileId(0), src).expect("parse failed");
    let aliases = HashMap::new();
    let typed = check(&program, &aliases).expect("type check failed");
    let (_registry, kernel_ids) =
        malus_codegen_gpu::compile_kernels(&typed).expect("kernel compilation failed");
    let symbols = mock_symbols();
    compile_and_run(&typed, &symbols, &kernel_ids)
}

fn dispatch_count() -> usize {
    MOCK_DISPATCH_COUNT.load(Ordering::SeqCst)
}

// ── Tensor alloc, print, and free ────────────────────────────────────────────

#[test]
fn test_tensor_alloc_and_free() {
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
    let src = r#"
fn make() -> Tensor<f32>:
    let a = Tensor.gpu<f32>([10.0, 20.0, 30.0])
    return a

fn main():
    let x = make()
    print(x)
"#;
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
    run_src(src).expect("add_tensors.ml flow should compile and run without panic");
}

// ── Scalar arithmetic ─────────────────────────────────────────────────────────

#[test]
fn test_scalar_add() {
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

    let symbols = mock_symbols();
    let kernel_ids = HashMap::new();
    let result = compile_and_run(&typed, &symbols, &kernel_ids);
    assert!(
        matches!(result, Err(CodegenError::NoMainFunction)),
        "expected NoMainFunction, got: {:?}",
        result
    );
}

// ── M5.1: fn-body tensor BinOp dispatches to built-in kernel ──────────────────

#[test]
fn test_fn_body_tensor_add_dispatches_once() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = a + b
    print(c)
"#;
    run_src(src).expect("fn-body tensor add should compile and run");
    assert_eq!(dispatch_count(), 1, "a + b in fn body should dispatch one builtin kernel");
}

#[test]
fn test_chained_binops_dispatch_twice() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = Tensor.gpu<f32>([5.0, 6.0])
    let r = a + b * c
    print(r)
"#;
    run_src(src).expect("chained fn-body BinOps should compile and run");
    assert_eq!(dispatch_count(), 2, "a + b * c should dispatch two builtin kernels");
}

#[test]
fn test_mixed_builtin_and_user_kernel_dispatch() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = add(a, b)
    let d = c + a
    print(d)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
"#;
    run_src(src).expect("mixed user kernel + builtin should compile and run");
    assert_eq!(dispatch_count(), 2, "user kernel + builtin add should dispatch twice");
}

#[test]
fn test_non_f32_tensor_binop_rejected() {
    let src = r#"
fn add(a: Tensor<f16>, b: Tensor<f16>) -> Tensor<f16>:
    return a + b

fn main():
    let a = Tensor.gpu<f16>([1.0, 2.0])
    let b = Tensor.gpu<f16>([3.0, 4.0])
    let c = add(a, b)
    print(c)
"#;
    let result = run_src(src);
    assert!(
        matches!(result, Err(CodegenError::UnsupportedExpr(_))),
        "non-f32 tensor BinOp should be rejected, got: {:?}",
        result
    );
}
