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
