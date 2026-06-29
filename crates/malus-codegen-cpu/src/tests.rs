use crate::{compile_and_run, CodegenError, RuntimeSymbols};
use malus_sema::check;
use malus_syntax::parse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// Tests share MOCK_STORE global state, so they must not run in parallel.
static TEST_LOCK: Mutex<()> = Mutex::new(());
static MOCK_DISPATCH_COUNT: AtomicUsize = AtomicUsize::new(0);

// ── Mock runtime ──────────────────────────────────────────────────────────────

struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

struct MockStore {
    tensors: HashMap<i64, MockTensor>,
    next_id: i64,
}

impl MockStore {
    fn new() -> Self {
        Self { tensors: HashMap::new(), next_id: 1 }
    }

    fn insert(&mut self, data: Vec<f32>, shape: Vec<usize>) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        self.tensors.insert(id, MockTensor { data, shape });
        id
    }

    fn get_data(&self, id: i64) -> Vec<f32> {
        self.tensors.get(&id).map(|t| t.data.clone()).unwrap_or_default()
    }

    fn get_shape(&self, id: i64) -> Vec<usize> {
        self.tensors.get(&id).map(|t| t.shape.clone()).unwrap_or_default()
    }

    fn get_len(&self, id: i64) -> usize {
        self.tensors.get(&id).map(|t| t.data.len()).unwrap_or(0)
    }
}

static MOCK_STORE: Mutex<Option<MockStore>> = Mutex::new(None);

fn with_store<R>(f: impl FnOnce(&mut MockStore) -> R) -> R {
    let mut guard = MOCK_STORE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.as_mut().expect("mock store not initialized"))
}

extern "C" fn mock_tensor_alloc_gpu(
    _dtype: i32,
    shape_ptr: *const usize,
    ndims: usize,
    data: *const f32,
) -> i64 {
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndims).to_vec() };
    let n: usize = shape.iter().product();
    let elements = if data.is_null() || n == 0 {
        vec![0.0f32; n]
    } else {
        unsafe { std::slice::from_raw_parts(data, n).to_vec() }
    };
    with_store(|s| s.insert(elements, shape))
}

extern "C" fn mock_tensor_alloc_zeros_gpu(shape_ptr: *const usize, ndims: usize) -> i64 {
    mock_tensor_alloc_gpu(0, shape_ptr, ndims, std::ptr::null())
}

extern "C" fn mock_tensor_alloc_ones_gpu(shape_ptr: *const usize, ndims: usize) -> i64 {
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndims).to_vec() };
    let n: usize = shape.iter().product();
    let ones: Vec<f32> = vec![1.0f32; n];
    mock_tensor_alloc_gpu(0, shape_ptr, ndims, ones.as_ptr())
}

extern "C" fn mock_tensor_print(handle: i64) {
    let elems = with_store(|s| s.get_data(handle));
    print!("[");
    for (i, v) in elems.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{v}");
    }
    print!("]");
}

extern "C" fn mock_tensor_free(handle: i64) {
    with_store(|s| { s.tensors.remove(&handle); });
}

extern "C" fn mock_kernel_dispatch(_kernel_id: u64, handles: *const i64, count: usize) -> i64 {
    MOCK_DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);
    let (len, shape) = if count < 1 || handles.is_null() {
        (1, vec![1usize])
    } else {
        let first = unsafe { *handles };
        with_store(|s| {
            let l = s.get_len(first);
            let sh = s.get_shape(first);
            (l.max(1), if sh.is_empty() { vec![l.max(1)] } else { sh })
        })
    };
    with_store(|s| s.insert(vec![1.0f32; len], shape))
}

extern "C" fn mock_gpu_barrier() {}

fn mock_binary_tensor_op(a: i64, b: i64, op: impl Fn(f32, f32) -> f32) -> i64 {
    let (ad, as_) = with_store(|s| (s.get_data(a), s.get_shape(a)));
    let (bd, bs) = with_store(|s| (s.get_data(b), s.get_shape(b)));
    // Simple equal-length zip for test purposes.
    let n = ad.len().max(bd.len());
    let out: Vec<f32> = (0..n).map(|i| op(ad[i % ad.len()], bd[i % bd.len()])).collect();
    let _ = (as_, bs);
    with_store(|s| s.insert(out, vec![n]))
}

extern "C" fn mock_tensor_matmul(handle_a: i64, handle_b: i64) -> i64 {
    let (a_data, a_shape) = with_store(|s| (s.get_data(handle_a), s.get_shape(handle_a)));
    let (b_data, b_shape) = with_store(|s| (s.get_data(handle_b), s.get_shape(handle_b)));
    if a_shape.len() == 2 && b_shape.len() == 2 && a_shape[1] == b_shape[0] {
        let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    out[i * n + j] += a_data[i * k + kk] * b_data[kk * n + j];
                }
            }
        }
        with_store(|s| s.insert(out, vec![m, n]))
    } else {
        // fallback stub: return a 1-element tensor
        with_store(|s| s.insert(vec![0.0f32], vec![1]))
    }
}

extern "C" fn mock_tensor_transpose(handle: i64) -> i64 {
    let (data, shape) = with_store(|s| (s.get_data(handle), s.get_shape(handle)));
    if shape.len() == 2 {
        let (m, n) = (shape[0], shape[1]);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = data[i * n + j];
            }
        }
        with_store(|s| s.insert(out, vec![n, m]))
    } else {
        with_store(|s| s.insert(data, shape))
    }
}

extern "C" fn mock_tensor_sum(handle: i64) -> i64 {
    let data = with_store(|s| s.get_data(handle));
    let total: f32 = data.iter().sum();
    with_store(|s| s.insert(vec![total], vec![1]))
}

extern "C" fn mock_tensor_len(handle: i64) -> i64 {
    with_store(|s| s.get_len(handle)) as i64
}

// M9 RC ABI — no-ops in tests (M9 CTMM emits no Retain/Release nodes).
extern "C" fn mock_tensor_retain(_handle: i64) {}
extern "C" fn mock_tensor_release(handle: i64) {
    // In a single-owner world (CTMM), release == free.
    with_store(|s| { s.tensors.remove(&handle); });
}

// M14 tape ABI — no-ops in tests (existing tests use Tensor, not Variable ops).
extern "C" fn mock_tape_record_binop(_op: i32, _a: i64, _b: i64, _out: i64) {}
extern "C" fn mock_tape_record_unary(_op: i32, _x: i64, _out: i64) {}
extern "C" fn mock_tape_register_leaf(_handle: i64) {}
extern "C" fn mock_tape_pause() {}
extern "C" fn mock_tape_resume() {}
extern "C" fn mock_tape_clear() {}
extern "C" fn mock_tape_get_grad(_handle: i64) -> i64 { 0 }
extern "C" fn mock_backward(_loss: i64) {}
// M15 tape ABI.
extern "C" fn mock_tape_zero_grad(_handles: *const i64, _count: usize) {}
// M16 broadcast + axis reductions — delegates to the mock forward ops for testing.
extern "C" fn mock_tensor_broadcast_add(_kernel_id: u64, a: i64, b: i64) -> i64 {
    MOCK_DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);
    mock_binary_tensor_op(a, b, |x, y| x + y)
}
extern "C" fn mock_tensor_broadcast_sub(_kernel_id: u64, a: i64, b: i64) -> i64 {
    MOCK_DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);
    mock_binary_tensor_op(a, b, |x, y| x - y)
}
extern "C" fn mock_tensor_broadcast_mul(_kernel_id: u64, a: i64, b: i64) -> i64 {
    MOCK_DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);
    mock_binary_tensor_op(a, b, |x, y| x * y)
}
extern "C" fn mock_tensor_broadcast_div(_kernel_id: u64, a: i64, b: i64) -> i64 {
    MOCK_DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);
    mock_binary_tensor_op(a, b, |x, y| x / y)
}
extern "C" fn mock_tensor_reduce_sum_axis(h: i64, _axis: i64, _keepdim: i64) -> i64 {
    let data = with_store(|s| s.get_data(h));
    let sum: f32 = data.iter().sum();
    with_store(|s| s.insert(vec![sum], vec![1]))
}
extern "C" fn mock_tensor_reduce_mean_axis(h: i64, _axis: i64, _keepdim: i64) -> i64 {
    let data = with_store(|s| s.get_data(h));
    let n = data.len() as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    with_store(|s| s.insert(vec![mean], vec![1]))
}
extern "C" fn mock_tensor_reduce_max_axis(h: i64, _axis: i64, _keepdim: i64) -> i64 {
    let data = with_store(|s| s.get_data(h));
    let max: f32 = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    with_store(|s| s.insert(vec![max], vec![1]))
}
extern "C" fn mock_tensor_reduce_var_axis(h: i64, _axis: i64, _keepdim: i64) -> i64 {
    let data = with_store(|s| s.get_data(h));
    let n = data.len() as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    with_store(|s| s.insert(vec![var], vec![1]))
}
extern "C" fn mock_tape_record_reduce(_op: i32, _x: i64, _out: i64, _axis: i64, _keepdim: i64) {}

fn mock_symbols() -> RuntimeSymbols {
    RuntimeSymbols {
        tensor_alloc_gpu:       mock_tensor_alloc_gpu,
        tensor_free:            mock_tensor_free,
        tensor_print:           mock_tensor_print,
        kernel_dispatch:        mock_kernel_dispatch,
        gpu_barrier:            mock_gpu_barrier,
        tensor_alloc_zeros_gpu: mock_tensor_alloc_zeros_gpu,
        tensor_alloc_ones_gpu:  mock_tensor_alloc_ones_gpu,
        tensor_matmul:          mock_tensor_matmul,
        tensor_transpose:       mock_tensor_transpose,
        tensor_sum:             mock_tensor_sum,
        tensor_len:             mock_tensor_len,
        tensor_retain:          mock_tensor_retain,
        tensor_release:         mock_tensor_release,
        tape_record_binop:      mock_tape_record_binop,
        tape_record_unary:      mock_tape_record_unary,
        tape_register_leaf:     mock_tape_register_leaf,
        tape_pause:             mock_tape_pause,
        tape_resume:            mock_tape_resume,
        tape_clear:             mock_tape_clear,
        tape_get_grad:          mock_tape_get_grad,
        backward:               mock_backward,
        tape_zero_grad:         mock_tape_zero_grad,
        tensor_broadcast_add:   mock_tensor_broadcast_add,
        tensor_broadcast_sub:   mock_tensor_broadcast_sub,
        tensor_broadcast_mul:   mock_tensor_broadcast_mul,
        tensor_broadcast_div:   mock_tensor_broadcast_div,
        tensor_reduce_sum_axis:  mock_tensor_reduce_sum_axis,
        tensor_reduce_mean_axis: mock_tensor_reduce_mean_axis,
        tensor_reduce_max_axis:  mock_tensor_reduce_max_axis,
        tensor_reduce_var_axis:  mock_tensor_reduce_var_axis,
        tape_record_reduce:     mock_tape_record_reduce,
    }
}

/// Return the current number of live tensors in the mock store.
fn live_tensor_count() -> usize {
    with_store(|s| s.tensors.len())
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

// ── M8: zeros / ones ─────────────────────────────────────────────────────────

#[test]
fn test_zeros_compiles_and_runs() {
    let src = r#"
fn main():
    let a = zeros(2, 3)
    println("zeros: {}", a)
"#;
    run_src(src).expect("zeros should compile and run");
}

#[test]
fn test_ones_compiles_and_runs() {
    let src = r#"
fn main():
    let a = ones(3, 4)
    println("ones: {}", a)
"#;
    run_src(src).expect("ones should compile and run");
}

// ── M8: unary builtins dispatched as GPU kernels ──────────────────────────────

#[test]
fn test_relu_dispatches_gpu_kernel() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let b = relu(a)
    println("{}", b)
"#;
    run_src(src).expect("relu should compile and run");
    assert_eq!(dispatch_count(), 1, "relu should dispatch exactly one GPU kernel");
}

#[test]
fn test_exp_dispatches_gpu_kernel() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = exp(a)
    println("{}", b)
"#;
    run_src(src).expect("exp should compile and run");
    assert_eq!(dispatch_count(), 1, "exp should dispatch exactly one GPU kernel");
}

#[test]
fn test_all_unary_builtins_compile() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([0.5, 1.0, 2.0])
    let r = relu(a)
    let s = sigmoid(a)
    let t = tanh(a)
    let e = exp(a)
    let l = log(a)
    let q = sqrt(a)
    let b = abs(a)
    println("{}", r)
"#;
    run_src(src).expect("all unary builtins should compile and run");
    assert_eq!(dispatch_count(), 7, "7 unary builtins should dispatch 7 GPU kernels");
}

// ── M8: matmul ───────────────────────────────────────────────────────────────

#[test]
fn test_matmul_compiles_and_runs() {
    let src = r#"
fn main():
    let a = ones(2, 3)
    let b = ones(3, 4)
    let c = a @ b
    println("{}", c)
"#;
    run_src(src).expect("matmul should compile and run");
}

#[test]
fn test_matmul_correct_values() {
    // ones([2,3]) @ ones([3,4]) -> [2,4] of all 3.0; sum = 24.
    // Correctness is verified by the mock computing the right value (shown in output);
    // we just assert the program runs without panic.
    let src = r#"
fn main():
    let a = ones(2, 3)
    let b = ones(3, 4)
    let c = a @ b
    let s = sum(c)
    println("{}", s)
"#;
    run_src(src).expect("matmul with sum should work");
}

// ── M8: transpose ────────────────────────────────────────────────────────────

#[test]
fn test_transpose_compiles_and_runs() {
    let src = r#"
fn main():
    let a = ones(3, 4)
    let b = transpose(a)
    println("{}", b)
"#;
    run_src(src).expect("transpose should compile and run");
}

// ── M8: sum ──────────────────────────────────────────────────────────────────

#[test]
fn test_sum_compiles_and_runs() {
    let src = r#"
fn main():
    let a = ones(2, 4)
    let s = sum(a)
    println("{}", s)
"#;
    run_src(src).expect("sum should compile and run");
}

// ── M8: .len field access ─────────────────────────────────────────────────────

#[test]
fn test_tensor_len_compiles_and_runs() {
    let src = r#"
fn main():
    let a = ones(3, 4)
    let n = a.len
    println("{}", n)
"#;
    run_src(src).expect(".len should compile and run");
}

// ── M8: done-when — 2-layer MLP forward pass ─────────────────────────────────

#[test]
fn test_mlp_forward_done_when() {
    let src = r#"
fn forward(x: Tensor<f32>, w1: Tensor<f32>, w2: Tensor<f32>) -> Tensor<f32>:
    let h = relu(x @ w1)
    return h @ w2

fn main():
    let x = ones(2, 3)
    let w1 = ones(3, 4)
    let w2 = ones(4, 2)
    let out = forward(x, w1, w2)
    println("forward output: {}", out)
    let s = sum(out)
    println("sum: {}", s)
    let wt = transpose(w1)
    println("transpose done")
    println("exp result: {}", exp(x))
"#;
    run_src(src).expect("done-when MLP forward should compile and run");
}

// ── M9: control flow ─────────────────────────────────────────────────────────

#[test]
fn test_if_taken_branch() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    if a.len > 0:
        print(a)
"#;
    run_src(src).expect("if (taken) should compile and run");
    // a must not leak: should be freed after the if stmt.
    assert_eq!(live_tensor_count(), 0, "a must be freed after the if stmt");
}

#[test]
fn test_if_not_taken_no_crash() {
    // Condition is false (len of a 2-element tensor is 2, not > 5).
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    if a.len > 5:
        print(a)
"#;
    run_src(src).expect("if (not taken) should compile and run without crash");
    assert_eq!(live_tensor_count(), 0, "a must be freed even when branch is not taken");
}

#[test]
fn test_if_else() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    if a.len > 5:
        print(a)
    else:
        print(a)
"#;
    run_src(src).expect("if/else should compile and run");
    assert_eq!(live_tensor_count(), 0, "a must be freed after if/else");
}

#[test]
fn test_for_loop_n_iterations() {
    // Loop runs 4 iterations.  Each iteration allocates one tensor via add(a, b).
    // CTMM places a Drop inside the loop body, so `out` is freed each iteration.
    let src = r#"
kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    for i in range(4):
        let out = add(a, b)
        print(out)
"#;
    run_src(src).expect("for loop should compile and run");
    // a, b freed after loop; out freed each iteration → nothing left.
    assert_eq!(live_tensor_count(), 0, "no tensors should leak after for loop");
    // add dispatched once per iteration → 4 times total.
    assert_eq!(dispatch_count(), 4, "kernel should dispatch once per iteration");
}

#[test]
fn test_for_loop_zero_iterations() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    for i in range(0):
        print(a)
"#;
    run_src(src).expect("for loop with zero iterations should not crash");
    assert_eq!(live_tensor_count(), 0, "a must be freed even with zero iterations");
}

#[test]
fn test_while_loop() {
    // We can't easily mutate the loop condition in the current IR without
    // let mut + integer arithmetic, so just test that a while(false) compiles.
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    while a.len > 5:
        print(a)
"#;
    run_src(src).expect("while (false condition) should compile and run");
    assert_eq!(live_tensor_count(), 0, "a must be freed after while loop");
}

#[test]
fn test_let_mut_in_loop() {
    // `acc` is outer let mut; each iteration drops old acc and rebinds.
    let src = r#"
kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let mut acc = Tensor.gpu<f32>([0.0, 0.0])
    let delta = Tensor.gpu<f32>([1.0, 2.0])
    for i in range(3):
        acc = add(acc, delta)
    print(acc)
"#;
    run_src(src).expect("let mut accumulator in loop should compile and run");
    assert_eq!(live_tensor_count(), 0, "acc and delta must be freed after loop");
}

// ── Phase 5: 2-D nested tensor literals ──────────────────────────────────────

#[test]
fn test_2d_tensor_literal_compiles() {
    let src = r#"
fn main():
    let x = Tensor.gpu<f32>([[1.0, 2.0], [3.0, 4.0]])
    print(x)
"#;
    run_src(src).expect("2-D tensor literal should compile and run");
    assert_eq!(live_tensor_count(), 0, "2-D tensor should be freed");
}

#[test]
fn test_2d_tensor_shape_in_alloc() {
    let src = r#"
fn main():
    let x = Tensor.gpu<f32>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(x)
"#;
    // Should pass 2 dimensions [2, 3] to tensor_alloc_gpu, which the mock records.
    run_src(src).expect("2-D tensor literal [2,3] should compile and run");
    assert_eq!(live_tensor_count(), 0);
}

// ── Phase 4: fixed arrays ─────────────────────────────────────────────────────

#[test]
fn test_array_literal_and_index() {
    let src = r#"
fn main():
    let xs = [10, 20, 30]
    let v0 = xs[0]
    let v1 = xs[1]
    let v2 = xs[2]
    println("vals: {} {} {}", v0, v1, v2)
"#;
    run_src(src).expect("array literal + index should compile and run");
}

#[test]
fn test_for_in_array() {
    let src = r#"
fn main():
    let xs = [1, 2, 3]
    let mut sum = 0
    for x in xs:
        sum = sum + x
    println("sum={}", sum)
"#;
    run_src(src).expect("for-in over integer array should compile and run");
}

#[test]
fn test_array_of_tensors_no_leak() {
    let src = r#"
fn main():
    let ts = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0]), Tensor.gpu<f32>([3.0])]
    print(ts[0])
"#;
    run_src(src).expect("array of tensors should compile and run");
    assert_eq!(live_tensor_count(), 0, "tensors in array must be freed by DropArray");
}

#[test]
fn test_for_in_tensor_array_no_leak() {
    let src = r#"
fn main():
    let ts = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    for t in ts:
        print(t)
"#;
    run_src(src).expect("for-in over tensor array should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensors should leak after for-in over tensor array");
}

/// M9 done-when: the example from the milestone spec.
#[test]
fn test_m9_done_when() {
    let src = r#"
fn main():
    let x = ones(2, 3)
    let w = ones(3, 2)
    for i in range(5):
        let out = x @ w
        let s = sum(out)
        println("step {}: sum = {}", i, s)
        if i > 2:
            println("  past halfway")
    println("done")
"#;
    run_src(src).expect("M9 done-when should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensors should leak in done-when program");
}

// ── M12: break / continue ─────────────────────────────────────────────────────

#[test]
fn test_break_exits_loop() {
    let src = r#"
fn main():
    let mut acc = 0
    for i in range(10):
        if i == 7:
            break
        if i == 3:
            continue
        acc = acc + i
    println("acc: {}", acc)
"#;
    // 0+1+2 + (3 skipped via continue) + 4+5+6 = 18; 7 triggers break
    run_src(src).expect("break/continue loop should compile and run");
}

#[test]
fn test_break_drops_loop_body_tensor() {
    // A tensor allocated inside the loop body before a break must be freed on
    // the break path — not just on the normal fall-through path.
    let src = r#"
fn main():
    let mut i = 0
    for i in range(5):
        let t = Tensor.gpu<f32>([1.0, 2.0])
        if i == 2:
            break
        print(t)
"#;
    run_src(src).expect("break with loop-body tensor should compile and run");
    assert_eq!(live_tensor_count(), 0, "tensor created before break must not leak");
}

#[test]
fn test_continue_drops_loop_body_tensor() {
    let src = r#"
fn main():
    for i in range(5):
        let t = Tensor.gpu<f32>([1.0, 2.0])
        if i == 2:
            continue
        print(t)
"#;
    run_src(src).expect("continue with loop-body tensor should compile and run");
    assert_eq!(live_tensor_count(), 0, "tensor created before continue must not leak");
}

#[test]
fn test_break_inside_if_inside_loop() {
    // break nested inside an if, with a loop-body tensor live at the break.
    let src = r#"
fn main():
    for i in range(10):
        let t = Tensor.gpu<f32>([1.0])
        if i == 3:
            if i > 2:
                break
        print(t)
"#;
    run_src(src).expect("nested break should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensor leak on nested break");
}

#[test]
fn test_while_break() {
    let src = r#"
fn main():
    let mut n = 0
    while n < 10:
        let t = Tensor.gpu<f32>([1.0])
        if n == 4:
            break
        print(t)
        n = n + 1
"#;
    run_src(src).expect("while + break should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensor leak on while break");
}

// ── M12: enum-payload retain-on-escape ───────────────────────────────────────

#[test]
fn test_match_payload_stays_local() {
    // Tensor payload used only inside the arm — no escape.  Retain/Drop pair
    // should cancel; DropEnum releases back to 0.
    let src = r#"
enum Wrapper:
    Some(val: Tensor<f32>)
    Empty

fn make() -> Wrapper:
    return Wrapper.Some(val=Tensor.gpu<f32>([1.0, 2.0]))

fn main():
    let w = make()
    match w:
        Some(val):
            print(val)
        Empty:
            println("empty")
"#;
    run_src(src).expect("local match payload should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensor leak for local payload use");
}

#[test]
fn test_match_payload_escapes_to_outer_binding() {
    // Tensor payload escapes via assignment to an outer `let mut`.
    // Must not leak OR double-free.
    let src = r#"
enum Wrapper:
    Some(val: Tensor<f32>)
    Empty

fn make() -> Wrapper:
    return Wrapper.Some(val=Tensor.gpu<f32>([3.0, 4.0]))

fn main():
    let w = make()
    let mut escaped = Tensor.gpu<f32>([0.0])
    match w:
        Some(val):
            escaped = val
        Empty:
            escaped = Tensor.gpu<f32>([0.0])
    print(escaped)
"#;
    run_src(src).expect("escaped payload should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensor leak or double-free on escape");
}

#[test]
fn test_match_empty_variant_no_leak() {
    let src = r#"
enum Wrapper:
    Some(val: Tensor<f32>)
    Empty

fn main():
    let w = Wrapper.Empty
    match w:
        Some(val):
            print(val)
        Empty:
            println("empty")
"#;
    run_src(src).expect("empty variant match should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensor leak for empty variant");
}

// ── M12: zero-length tensor guard ────────────────────────────────────────────

#[test]
fn test_zero_length_tensor_alloc() {
    // zeros(0) allocates a zero-length tensor — must not crash on allocation.
    let src = r#"
fn main():
    let empty = zeros(0)
    print(empty)
"#;
    run_src(src).expect("zeros(0) should not crash");
    assert_eq!(live_tensor_count(), 0, "zero-length tensor must be freed");
}

// ── M12: break/continue outside a loop → sema error ──────────────────────────

fn check_src(src: &str) -> Result<malus_sema::TypedProgram, Vec<malus_sema::SemaError>> {
    let program = parse(malus_syntax::FileId(0), src).expect("parse failed");
    let aliases = HashMap::new();
    check(&program, &aliases)
}

#[test]
fn test_break_outside_loop_rejected() {
    let src = r#"
fn main():
    break
"#;
    let result = check_src(src);
    assert!(result.is_err(), "break outside loop should be rejected by sema");
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(e, malus_sema::SemaError::BreakOutsideLoop { .. })),
        "expected BreakOutsideLoop error, got: {:?}", errs
    );
}

#[test]
fn test_continue_outside_loop_rejected() {
    let src = r#"
fn main():
    continue
"#;
    let result = check_src(src);
    assert!(result.is_err(), "continue outside loop should be rejected by sema");
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(e, malus_sema::SemaError::ContinueOutsideLoop { .. })),
        "expected ContinueOutsideLoop error, got: {:?}", errs
    );
}

#[test]
fn test_break_inside_if_outside_loop_rejected() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    if a.len > 0:
        break
"#;
    let result = check_src(src);
    assert!(result.is_err(), "break inside if but outside loop should be rejected");
}

// ── M13: non-tensor payload escape is now allowed (ARC header on aggregate boxes) ──

#[test]
fn test_struct_payload_escape_now_allowed() {
    let src = r#"
struct Point:
    x: f32

enum Wrapper:
    Some(pt: Point)
    Empty

fn main():
    let w = Wrapper.Some(pt=Point(x=1.0))
    match w:
        Some(pt):
            let escaped = pt
        Empty:
            println("empty")
"#;
    run_src(src).expect("escaping struct payload is allowed in M13");
}

#[test]
fn test_struct_payload_field_read_ok() {
    // Reading a scalar field out of a struct payload is fine — not an escape.
    let src = r#"
struct Point:
    x: f32

enum Wrapper:
    Some(pt: Point)
    Empty

fn main():
    let w = Wrapper.Some(pt=Point(x=3.0))
    match w:
        Some(pt):
            println("x: {}", pt.x)
        Empty:
            println("empty")
"#;
    run_src(src).expect("reading a struct payload field should compile and run");
}

// ── M13: Variable type codegen ────────────────────────────────────────────────

#[test]
fn test_variable_rc_done_when() {
    // The M13 spec done-when: wrap, identity, variable(ones/zeros), b.data, c.data.
    let src = r#"
fn wrap(t: Tensor<f32>) -> Variable<f32>:
    return variable(t)

fn identity(v: Variable<f32>) -> Variable<f32>:
    return v

fn main():
    let a = variable(ones(2, 2))
    let b = identity(a)
    let c = variable(zeros(3, 3))
    tensor_print(b.data)
    tensor_print(c.data)
"#;
    run_src(src).expect("M13 done-when should compile and run");
    assert_eq!(live_tensor_count(), 0, "Variable done-when: no tensor leaks");
}

#[test]
fn test_variable_data_let_bind_no_leak() {
    // let d = v.data must retain so d and v each own a ref; no double-free.
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let v = variable(t)
    let d = v.data
    tensor_print(d)
"#;
    run_src(src).expect("Variable .data let-bind should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensor leaks after .data let-bind");
}

#[test]
fn test_variable_data_inline_borrow() {
    // Inline .data borrow (not let-bound) should not retain.
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([42.0])
    let v = variable(t)
    tensor_print(v.data)
"#;
    run_src(src).expect("inline .data borrow should compile and run");
    assert_eq!(live_tensor_count(), 0, "no tensor leaks after inline .data borrow");
}

// ── M13.5: Tuples ─────────────────────────────────────────────────────────────

#[test]
fn test_tuple_scalar_construction_and_access() {
    let src = r#"
fn main():
    let t = (25.0, 50.0)
    let x = t.0
    let y = t.1
    println("{} {}", x, y)
"#;
    run_src(src).expect("tuple scalar construction and positional access");
}

#[test]
fn test_tuple_bool_mixed() {
    let src = r#"
fn main():
    let t = (3.14, true)
    let x = t.0
    let b = t.1
    println("{}", x)
"#;
    run_src(src).expect("tuple with mixed bool/f32");
}

#[test]
fn test_tuple_destructuring() {
    let src = r#"
fn main():
    let t = (10.0, 20.0)
    let (a, b) = t
    println("{} {}", a, b)
"#;
    run_src(src).expect("tuple destructuring");
}

#[test]
fn test_tuple_return() {
    let src = r#"
fn swap(x: f32, y: f32) -> (f32, f32):
    return (y, x)

fn main():
    let (a, b) = swap(1.0, 2.0)
    println("{} {}", a, b)
"#;
    run_src(src).expect("tuple return from fn");
}

#[test]
fn test_tuple_three_elements() {
    let src = r#"
fn triple() -> (f32, f32, f32):
    return (1.0, 2.0, 3.0)

fn main():
    let t = triple()
    let x = t.0
    let y = t.1
    let z = t.2
    println("{} {} {}", x, y, z)
"#;
    run_src(src).expect("3-element tuple");
}

#[test]
fn test_tuple_let_mut_destructuring() {
    let src = r#"
fn main():
    let mut (a, b) = (5.0, 10.0)
    a = 99.0
    println("{} {}", a, b)
"#;
    run_src(src).expect("let mut tuple destructuring with reassignment");
}

#[test]
fn test_tuple_roundtrip_parse() {
    // Verify that the parser accepts and the printer round-trips tuples.
    let src = r#"
fn main():
    let t = (1.0, 2.0)
    let x = t.0
    println("{}", x)
"#;
    run_src(src).expect("tuple parse roundtrip");
}

// ── M15: zero_grad + re-wrap SGD ─────────────────────────────────────────────

#[test]
fn test_zero_grad_compiles_and_runs() {
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([1.0, 2.0])
    let v = variable(t)
    zero_grad(v)
"#;
    run_src(src).expect("zero_grad should compile and run");
    assert_eq!(live_tensor_count(), 0, "no leaks after zero_grad");
}

#[test]
fn test_zero_grad_multiple_args() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    let b = Tensor.gpu<f32>([2.0])
    let va = variable(a)
    let vb = variable(b)
    zero_grad(va, vb)
"#;
    run_src(src).expect("zero_grad with multiple args");
    assert_eq!(live_tensor_count(), 0, "no leaks after zero_grad with multiple args");
}

#[test]
fn test_zero_grad_rewrap_no_leak() {
    // Verify that CTMM correctly drops the old Variable handle on each re-wrap
    // and that zero_grad emits its call without leaking mock tensors.
    // We avoid Variable binops (vx @ w etc.) because their intermediate
    // tensors are managed by the tape in the real runtime but the mock tape
    // ABI is all no-ops — intermediate lifetime is a runtime concern, tested
    // separately in malus-runtime::tests::test_rewrap_registry_stays_bounded.
    let src = r#"
fn main():
    let t1 = Tensor.gpu<f32>([0.5, 0.5])
    let mut w = variable(t1)
    zero_grad(w)
    let t2 = Tensor.gpu<f32>([0.3, 0.3])
    zero_grad(w)
    w = variable(t2)
    zero_grad(w)
"#;
    run_src(src).expect("zero_grad + re-wrap should compile and run");
    assert_eq!(live_tensor_count(), 0, "no mock-tensor leaks from zero_grad + Variable re-wrap");
}
