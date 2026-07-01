use std::collections::HashMap;

use malus_syntax::parse;
use malus_sema::check;

use crate::{compile_kernels, CodegenError, KernelRegistry};

fn compile_src(src: &str) -> Result<(KernelRegistry, HashMap<String, u64>), CodegenError> {
    let program = parse(malus_syntax::FileId(0), src).expect("parse failed");
    let aliases = HashMap::new();
    let typed = check(&program, &aliases).expect("type check failed");
    compile_kernels(&typed)
}

#[test]
fn test_empty_program() {
    let src = "fn main():\n    print(\"hello\")\n";
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    assert!(registry.is_empty());
    assert!(name_to_id.is_empty());
}

#[test]
fn test_single_add_kernel_msl() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    println(c)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");

    assert_eq!(name_to_id.len(), 1);
    let add_id = name_to_id["add"];

    let msl = registry.msl_for(add_id).expect("add kernel should exist");
    assert!(msl.contains(&format!("kernel void malus_kernel_{add_id}")));
    assert!(msl.contains("device float* a [[buffer(0)]]"));
    assert!(msl.contains("device float* b [[buffer(1)]]"));
    assert!(msl.contains("device float* out [[buffer(2)]]"));
    assert!(msl.contains("uint tid [[thread_position_in_grid]]"));
    assert!(msl.contains("out[tid] = (a[tid] + b[tid]);"));
}

#[test]
fn test_multiple_kernels_get_sequential_ids() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = add(a, b)
    let d = sub(a, b)
    println(c)
    println(d)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

kernel sub(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a - b
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");

    let add_id = name_to_id["add"];
    let sub_id = name_to_id["sub"];
    // Ids come from a process-global counter (ADR-0033), so concurrently-running
    // tests' compilations may interleave and steal values in between — only the
    // relative order within this one compilation is guaranteed, not adjacency.
    assert!(sub_id > add_id, "kernel ids within one compilation must be assigned in order");
    assert!(registry.msl_for(add_id).unwrap().contains(&format!("malus_kernel_{add_id}")));
    assert!(registry.msl_for(sub_id).unwrap().contains(&format!("malus_kernel_{sub_id}")));
}

#[test]
fn test_sub_mul_div_ops() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = ksub(a, b)
    let d = kmul(a, b)
    let e = kdiv(a, b)
    println(c)
    println(d)
    println(e)

kernel ksub(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a - b

kernel kmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a * b

kernel kdiv(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a / b
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");

    assert!(registry.msl_for(name_to_id["ksub"]).unwrap().contains("(a[tid] - b[tid])"));
    assert!(registry.msl_for(name_to_id["kmul"]).unwrap().contains("(a[tid] * b[tid])"));
    assert!(registry.msl_for(name_to_id["kdiv"]).unwrap().contains("(a[tid] / b[tid])"));
}

#[test]
fn test_unary_neg() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = neg(a)
    println(b)

kernel neg(a: Tensor<f32>) -> Tensor<f32>:
    return -a
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    assert!(registry.msl_for(name_to_id["neg"]).unwrap().contains("(-a[tid])"));
}

#[test]
fn test_nested_binop_precedence() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = fma(a, b)
    println(c)

kernel fma(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return (a + b) * a
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(name_to_id["fma"]).unwrap();
    assert!(msl.contains("((a[tid] + b[tid]) * a[tid])"));
}

#[test]
fn test_single_param_copy_kernel() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = copy(a)
    println(b)

kernel copy(a: Tensor<f32>) -> Tensor<f32>:
    return a
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(name_to_id["copy"]).unwrap();
    assert!(msl.contains("device float* a [[buffer(0)]]"));
    assert!(msl.contains("device float* out [[buffer(1)]]"));
    assert!(msl.contains("out[tid] = a[tid];"));
}

#[test]
fn test_matmul_rejected() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = mm(a, b)
    println(c)

kernel mm(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a @ b
"#;
    let result = compile_src(src);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, CodegenError::UnsupportedKernelBody(_)));
    assert!(err.to_string().contains("matmul"));
}

#[test]
fn test_f16_dtype() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f16>([1.0, 2.0])
    let b = add(a, a)
    println(b)

kernel add(a: Tensor<f16>, b: Tensor<f16>) -> Tensor<f16>:
    return a + b
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(name_to_id["add"]).unwrap();
    assert!(msl.contains("device half* a"));
    assert!(msl.contains("device half* out"));
}

// ── M5.1: built-in element-wise kernel synthesis ──────────────────────────────

#[test]
fn test_builtin_add_synthesized_from_fn_body_binop() {
    let src = r#"
fn add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = add(a, b)
    println(c)
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    assert_eq!(name_to_id.len(), 1);
    let id = name_to_id["malus_add"];
    let msl = registry.msl_for(id).expect("malus_add should exist");
    assert!(msl.contains("kernel void malus_kernel_"));
    assert!(msl.contains("device float* a [[buffer(0)]]"));
    assert!(msl.contains("device float* b [[buffer(1)]]"));
    assert!(msl.contains("device float* out [[buffer(2)]]"));
    assert!(msl.contains("out[tid] = (a[tid] + b[tid]);"));
}

#[test]
fn test_builtin_ops_synthesized_for_sub_mul_div() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let s = a - b
    let m = a * b
    let d = a / b
    println(s)
    println(m)
    println(d)
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    assert_eq!(name_to_id.len(), 3);

    let sub_msl = registry.msl_for(name_to_id["malus_sub"]).unwrap();
    assert!(sub_msl.contains("out[tid] = (a[tid] - b[tid]);"));
    let mul_msl = registry.msl_for(name_to_id["malus_mul"]).unwrap();
    assert!(mul_msl.contains("out[tid] = (a[tid] * b[tid]);"));
    let div_msl = registry.msl_for(name_to_id["malus_div"]).unwrap();
    assert!(div_msl.contains("out[tid] = (a[tid] / b[tid]);"));
}

#[test]
fn test_builtin_ids_append_after_user_kernels() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = ksub(a, b)
    let d = a + b
    println(c)
    println(d)

kernel ksub(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a - b
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let ksub_id = name_to_id["ksub"];
    let add_id = name_to_id["malus_add"];
    // Relative order only — see note in test_multiple_kernels_get_sequential_ids.
    assert!(add_id > ksub_id, "builtin ids must append after user kernel ids (ADR-0010)");
    assert!(registry.msl_for(ksub_id).unwrap().contains(&format!("malus_kernel_{ksub_id}")));
    assert!(registry.msl_for(add_id).unwrap().contains(&format!("malus_kernel_{add_id}")));
}

#[test]
fn test_chained_binops_synthesize_two_builtins() {
    let src = r#"
fn fma(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) -> Tensor<f32>:
    return a + b * c

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = Tensor.gpu<f32>([5.0, 6.0])
    let r = fma(a, b, c)
    println(r)
"#;
    let (_registry, name_to_id) = compile_src(src).expect("should compile");
    assert!(name_to_id.contains_key("malus_add"));
    assert!(name_to_id.contains_key("malus_mul"));
}

#[test]
fn test_tensor_matmul_in_fn_body_not_synthesized() {
    let src = r#"
fn mm(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a @ b

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = mm(a, b)
    println(c)
"#;
    let (_registry, name_to_id) = compile_src(src).expect("should compile");
    assert!(!name_to_id.contains_key("malus_matmul"));
    assert!(name_to_id.is_empty());
}

// ── M7: multi-statement kernel bodies ────────────────────────────────────────

#[test]
fn test_multistmt_kernel_let_then_return() {
    let src = r#"
kernel relu_backward(grad_out: Tensor<f32>, x: Tensor<f32>) -> Tensor<f32>:
    let mask = x > 0.0
    return grad_out * mask

fn main():
    let g = Tensor.gpu<f32>([1.0, 2.0])
    let x = Tensor.gpu<f32>([1.0, -1.0])
    let d = relu_backward(g, x)
    println(d)
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let id = name_to_id["relu_backward"];
    let msl = registry.msl_for(id).unwrap();
    // Let binding: compare produces a float mask in element-space
    assert!(msl.contains("float mask = (x[tid] > 0.0f);"), "expected mask = (x[tid] > 0.0f) in MSL, got:\n{msl}");
    // Return: multiply grad_out by mask (local, no indexing)
    assert!(msl.contains("out[tid] = (grad_out[tid] * mask);"), "expected out[tid] = (grad_out[tid] * mask) in MSL, got:\n{msl}");
}

#[test]
fn test_kernel_comparison_ops_in_msl() {
    let src = r#"
kernel cmp_kernel(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    let lt_mask = a < b
    return lt_mask

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([2.0, 1.0])
    let c = cmp_kernel(a, b)
    println(c)
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(name_to_id["cmp_kernel"]).unwrap();
    assert!(msl.contains("(a[tid] < b[tid])"), "< comparison should appear in MSL, got:\n{msl}");
}

#[test]
fn test_kernel_float_literal_in_msl() {
    let src = r#"
kernel scale(a: Tensor<f32>) -> Tensor<f32>:
    let half = 0.5
    return a * half

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = scale(a)
    println(b)
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(name_to_id["scale"]).unwrap();
    assert!(msl.contains("0.5f"), "float literal should emit as 0.5f in MSL, got:\n{msl}");
    assert!(msl.contains("float half = 0.5f;"), "let binding should emit correctly, got:\n{msl}");
}

// ── M7: scalar-broadcast builtin synthesis ───────────────────────────────────

#[test]
fn test_scalar_broadcast_mul_synthesized() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let scaled = a * 0.5
    println(scaled)
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    assert!(name_to_id.contains_key("malus_mul_scalar"), "malus_mul_scalar should be synthesized");
    let id = name_to_id["malus_mul_scalar"];
    let msl = registry.msl_for(id).unwrap();
    assert!(msl.contains("device float* a [[buffer(0)]]"), "tensor param at buffer(0)");
    assert!(msl.contains("device float* scalar_val [[buffer(1)]]"), "scalar param at buffer(1)");
    assert!(msl.contains("device float* out [[buffer(2)]]"), "output at buffer(2)");
    assert!(msl.contains("(a[tid] * scalar_val[0])"), "MSL body should multiply a[tid] by scalar_val[0]");
}

#[test]
fn test_scalar_broadcast_add_synthesized() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = a + 1.0
    println(b)
"#;
    let (_registry, name_to_id) = compile_src(src).expect("should compile");
    assert!(name_to_id.contains_key("malus_add_scalar"));
}

#[test]
fn test_scalar_broadcast_commutative_dedup() {
    // Both `a + 0.5` and `0.5 + a` use the same kernel malus_add_scalar.
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = a + 0.5
    let c = 0.5 + a
    println(b)
    println(c)
"#;
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    // Only one malus_add_scalar kernel (commutative dedup).
    let add_scalar_count = name_to_id.keys().filter(|k| k.as_str() == "malus_add_scalar").count();
    assert_eq!(add_scalar_count, 1, "malus_add_scalar should appear exactly once");
    let id = name_to_id["malus_add_scalar"];
    assert!(registry.msl_for(id).is_some());
}

#[test]
fn test_scalar_broadcast_sub_div_both_orders() {
    // sub and div are non-commutative: both orders produce distinct kernels.
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = a - 1.0
    let c = 1.0 - a
    let d = a / 2.0
    println(b)
    println(c)
    println(d)
"#;
    let (_registry, name_to_id) = compile_src(src).expect("should compile");
    assert!(name_to_id.contains_key("malus_sub_scalar"),  "tensor - scalar should produce malus_sub_scalar");
    assert!(name_to_id.contains_key("malus_rsub_scalar"), "scalar - tensor should produce malus_rsub_scalar");
    assert!(name_to_id.contains_key("malus_div_scalar"),  "tensor / scalar should produce malus_div_scalar");
}

#[test]
fn test_scalar_builtin_ids_after_user_and_tensor_tensor_kernels() {
    // User kernel first, then tensor-tensor builtin, then scalar builtins (ADR-0010 ordering).
    let src = r#"
kernel copy(a: Tensor<f32>) -> Tensor<f32>:
    return a

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = copy(a)
    let d = a + b
    let e = a * 0.5
    println(c)
    println(d)
    println(e)
"#;
    let (_registry, name_to_id) = compile_src(src).expect("should compile");
    let copy_id  = name_to_id["copy"];
    let add_id   = name_to_id["malus_add"];
    let mul_s_id = name_to_id["malus_mul_scalar"];
    // User kernel < tensor-tensor builtin < scalar builtin
    assert!(copy_id  < add_id,   "user kernel id ({copy_id}) < tensor-tensor builtin id ({add_id})");
    assert!(add_id   < mul_s_id, "tensor-tensor builtin id ({add_id}) < scalar builtin id ({mul_s_id})");
}

// ── M24: explicit kernel codegen tests ───────────────────────────────────────

#[test]
fn test_m24_explicit_kernel_emits_msl() {
    // Smoke-test: an explicit kernel that exercises let shared, barrier(), for loop,
    // if, threadgroup_id(), thread_in_threadgroup(), flat indexing, and a scalar
    // uniform — all required M24 features.  Only validates the MSL is emitted
    // (no xcrun available in CI); structure assertions cover correctness.
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
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let kernel_id = name_to_id["softmax"];
    let msl = registry.msl_for(kernel_id).expect("MSL for softmax not found");

    assert!(msl.contains("threadgroup float scratch[1024]"), "shared memory declaration missing");
    assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"), "barrier missing");
    assert!(msl.contains("for(int i = 1; i < u.cols; i++)"), "for loop (range 1..cols) missing");
    assert!(msl.contains("for(int j = 0; j < u.cols; j++)"), "for loop (range 0..cols) missing");
    assert!(msl.contains("uint _tgid [[threadgroup_position_in_grid]]"), "threadgroup_id attr missing");
    assert!(msl.contains("uint _lid [[thread_position_in_threadgroup]]"), "thread_in_threadgroup attr missing");
    assert!(msl.contains(&format!("struct Uniforms_{kernel_id}")), "uniforms struct missing");
    assert!(msl.contains("int cols"), "cols field in uniforms missing");
    assert!(msl.contains("fmax("), "fmax call missing");
    assert!(msl.contains("exp("), "exp call missing");
}

#[test]
fn test_m24_gelu_explicit_kernel_no_uniforms() {
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
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let kernel_id = name_to_id["gelu"];
    let msl = registry.msl_for(kernel_id).expect("MSL for gelu not found");

    assert!(msl.contains("uint _tid [[thread_position_in_grid]]"), "thread_id attr missing");
    assert!(!msl.contains("Uniforms_"), "no uniforms struct expected for gelu");
    assert!(msl.contains("tanh("), "tanh call missing");
    // No threadgroup shared memory in gelu.
    assert!(!msl.contains("threadgroup"), "no threadgroup memory expected in gelu");
}

#[test]
fn test_m24_layernorm_uniforms_struct() {
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
    let (registry, name_to_id) = compile_src(src).expect("should compile");
    let kernel_id = name_to_id["layernorm"];
    let msl = registry.msl_for(kernel_id).expect("MSL for layernorm not found");

    assert!(msl.contains(&format!("struct Uniforms_{kernel_id}")), "uniforms struct missing");
    assert!(msl.contains("int cols"), "cols field missing");
    assert!(msl.contains("float inv_cols"), "inv_cols field missing");
    assert!(msl.contains("float eps"), "eps field missing");
    assert!(msl.contains("rsqrt("), "rsqrt call missing");
    assert!(msl.contains("threadgroup float scratch[1024]"), "shared memory missing");
}
