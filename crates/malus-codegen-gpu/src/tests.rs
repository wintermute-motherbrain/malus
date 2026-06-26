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
    assert_eq!(name_to_id["add"], 0);

    let msl = registry.msl_for(0).expect("kernel 0 should exist");
    assert!(msl.contains("kernel void malus_kernel_0"));
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

    assert_eq!(name_to_id["add"], 0);
    assert_eq!(name_to_id["sub"], 1);
    assert!(registry.msl_for(0).unwrap().contains("malus_kernel_0"));
    assert!(registry.msl_for(1).unwrap().contains("malus_kernel_1"));
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
    let (registry, _) = compile_src(src).expect("should compile");

    assert!(registry.msl_for(0).unwrap().contains("(a[tid] - b[tid])"));
    assert!(registry.msl_for(1).unwrap().contains("(a[tid] * b[tid])"));
    assert!(registry.msl_for(2).unwrap().contains("(a[tid] / b[tid])"));
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
    let (registry, _) = compile_src(src).expect("should compile");
    assert!(registry.msl_for(0).unwrap().contains("(-a[tid])"));
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
    let (registry, _) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(0).unwrap();
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
    let (registry, _) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(0).unwrap();
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
    let (registry, _) = compile_src(src).expect("should compile");
    let msl = registry.msl_for(0).unwrap();
    assert!(msl.contains("device half* a"));
    assert!(msl.contains("device half* out"));
}
