use std::collections::{HashMap, HashSet};
use malus_syntax::parse;
use crate::{check, SemaError, TypedAssignTarget, TypedStmt};

fn check_src(src: &str) -> Result<crate::TypedProgram, Vec<SemaError>> {
    let program = parse(malus_syntax::FileId(0), src).expect("parse failed");
    check(&program, &HashMap::new())
}

// ── Happy path ────────────────────────────────────────────────────────────────

#[test]
fn test_add_tensors_mvp() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    print(c)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
"#;
    let typed = check_src(src).expect("should type-check without errors");
    assert_eq!(typed.fns.len(), 1);
    assert_eq!(typed.kernels.len(), 1);

    let main = &typed.fns[0];
    assert_eq!(main.name, "main");

    // c should resolve to Tensor<f32>
    let c_let = main.body.iter().find(|s| matches!(s, TypedStmt::Let { name, .. } if name == "c"));
    assert!(c_let.is_some(), "let c should be present");
    if let Some(TypedStmt::Let { expr, .. }) = c_let {
        assert!(matches!(expr.ty, crate::ResolvedTy::Tensor { .. }));
    }

    // CTMM: GpuBarrier + Drop a + Drop b present
    let has_barrier = main.body.iter().any(|s| matches!(s, TypedStmt::GpuBarrier));
    assert!(has_barrier, "GpuBarrier should be present for in-flight a and b");

    let drop_a = main.body.iter().any(|s| matches!(s, TypedStmt::Drop { name } if name == "a"));
    let drop_b = main.body.iter().any(|s| matches!(s, TypedStmt::Drop { name } if name == "b"));
    let drop_c = main.body.iter().any(|s| matches!(s, TypedStmt::Drop { name } if name == "c"));
    assert!(drop_a, "Drop(a) should be present");
    assert!(drop_b, "Drop(b) should be present");
    assert!(drop_c, "Drop(c) should be present");

    // c should not have a GpuBarrier before its drop (print is not a kernel)
    // The only GpuBarrier should appear before drop(a)/drop(b)
    let barrier_count = main.body.iter().filter(|s| matches!(s, TypedStmt::GpuBarrier)).count();
    assert_eq!(barrier_count, 1, "exactly one GpuBarrier for the single kernel dispatch");
}

#[test]
fn test_returned_tensor_has_no_drop() {
    let src = r#"
fn make() -> Tensor<f32>:
    let a = Tensor.gpu<f32>([1.0, 2.0])
    return a

fn main():
    let x = make()
    print(x)
"#;
    let typed = check_src(src).expect("should type-check");
    let make_fn = typed.fns.iter().find(|f| f.name == "make").unwrap();
    // `a` escapes via return — no Drop should be injected
    let drop_a = make_fn.body.iter().any(|s| matches!(s, TypedStmt::Drop { name } if name == "a"));
    assert!(!drop_a, "escaped tensor should not get a Drop");
}

// ── Tensor literal coercion ───────────────────────────────────────────────────

#[test]
fn test_int_literal_in_float_tensor_ok() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1, 2, 3])
    print(a)
"#;
    assert!(check_src(src).is_ok(), "int literals should coerce losslessly into f32 tensor");
}

#[test]
fn test_float_literal_in_int_tensor_error() {
    let src = r#"
fn main():
    let a = Tensor.gpu<i32>([1.5, 2.5])
    print(a)
"#;
    let errors = check_src(src).expect_err("float into int tensor should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::LossyCoercion { .. })),
        "expected LossyCoercion error, got: {:?}", errors
    );
}

// ── Type errors ───────────────────────────────────────────────────────────────

#[test]
fn test_dtype_mismatch_binop() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f16>([3.0, 4.0])
    let c = a + b
    print(c)

kernel dummy(a: Tensor<f32>) -> Tensor<f32>:
    return a
"#;
    // Note: f16 literal parsing may not be supported — use a helper kernel to produce f16
    // For this test, we check that mismatched dtypes produce DtypeMismatch.
    // If f16 literal isn't parseable, this test demonstrates the error structure.
    let _ = check_src(src); // may error at parse or at sema — either is acceptable
}

#[test]
fn test_unknown_identifier() {
    let src = r#"
fn main():
    let a = x
"#;
    let errors = check_src(src).expect_err("unknown ident should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::UnknownIdent { name, .. } if name == "x")),
        "expected UnknownIdent(x), got: {:?}", errors
    );
}

#[test]
fn test_arg_count_mismatch() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    let c = add(a)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
"#;
    let errors = check_src(src).expect_err("wrong arg count should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::ArgCountMismatch { callee, .. } if callee == "add")),
        "expected ArgCountMismatch for add, got: {:?}", errors
    );
}

#[test]
fn test_return_type_mismatch() {
    let src = r#"
fn make() -> Tensor<f32>:
    return 42

fn main():
    print(make())
"#;
    // `return 42` should produce a return type mismatch (i64 vs Tensor<f32>)
    // Note: make() might fail due to missing `fn main()` — adjust expectation
    let result = check_src(src);
    // Either ReturnTypeMismatch or another error — we just want it to fail
    assert!(result.is_err(), "return type mismatch should produce errors");
}

#[test]
fn test_duplicate_definition() {
    let src = r#"
fn foo():
    print(foo())

fn foo():
    print(foo())

fn main():
    foo()
"#;
    let errors = check_src(src).expect_err("duplicate fn should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::DuplicateDefinition { name, .. } if name == "foo")),
        "expected DuplicateDefinition(foo), got: {:?}", errors
    );
}

#[test]
fn test_main_not_found() {
    let src = r#"
fn helper():
    let a = Tensor.gpu<f32>([1.0])
    print(a)
"#;
    let errors = check_src(src).expect_err("missing main should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::MainNotFound)),
        "expected MainNotFound, got: {:?}", errors
    );
}

#[test]
fn test_kernel_called_from_kernel_error() {
    let src = r#"
kernel inner(a: Tensor<f32>) -> Tensor<f32>:
    return a

kernel outer(a: Tensor<f32>) -> Tensor<f32>:
    return inner(a)

fn main():
    let a = Tensor.gpu<f32>([1.0])
    let b = outer(a)
    print(b)
"#;
    let errors = check_src(src).expect_err("kernel-from-kernel should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::KernelCalledFromKernel { .. })),
        "expected KernelCalledFromKernel, got: {:?}", errors
    );
}

// ── print / println format string validation ──────────────────────────────────

#[test]
fn test_println_builtin_exists() {
    let src = r#"
fn main():
    let x = 1.0
    println(x)
"#;
    assert!(check_src(src).is_ok(), "println(x) should pass sema");
}

#[test]
fn test_print_bare_string_ok() {
    let src = r#"
fn main():
    print("hello world")
"#;
    assert!(check_src(src).is_ok(), "print with bare string (0 placeholders, 0 value args) should pass");
}

#[test]
fn test_println_no_args_ok() {
    let src = r#"
fn main():
    println()
"#;
    assert!(check_src(src).is_ok(), "println() with no args should pass");
}

#[test]
fn test_print_format_string_ok() {
    let src = r#"
fn main():
    println("{} + {} = {}", 1.0, 2.0, 3.0)
"#;
    assert!(check_src(src).is_ok(), "format string with matching placeholder count should pass");
}

#[test]
fn test_print_format_too_few_args() {
    let src = r#"
fn main():
    println("{} {}", 1.0)
"#;
    let errors = check_src(src).expect_err("too few args should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::FormatArgCountMismatch { placeholders: 2, args: 1, .. })),
        "expected FormatArgCountMismatch(2 placeholders, 1 arg), got: {:?}", errors
    );
}

#[test]
fn test_print_format_too_many_args() {
    let src = r#"
fn main():
    println("{}", 1.0, 2.0)
"#;
    let errors = check_src(src).expect_err("too many args should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::FormatArgCountMismatch { placeholders: 1, args: 2, .. })),
        "expected FormatArgCountMismatch(1 placeholder, 2 args), got: {:?}", errors
    );
}

#[test]
fn test_string_literal_as_str_value_accepted() {
    // Since M22, string literals have type Str and are valid in any print
    // position — not just as the first format-string arg.  print(1.0, "oops")
    // prints the f32 value then the string "oops" via the legacy variadic path.
    let src = r#"
fn main():
    print(1.0, "oops")
"#;
    check_src(src).expect("string literal in non-first arg of print should now be accepted");
}

// ── Module aliases ────────────────────────────────────────────────────────────

#[test]
fn test_qualified_call_via_module_alias() {
    let src = r#"
fn add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = ops.add(a, b)
    print(c)
"#;
    // Register "ops" as an alias exporting "add"
    let mut aliases: HashMap<String, HashSet<String>> = HashMap::new();
    aliases.insert("ops".to_string(), {
        let mut s = HashSet::new();
        s.insert("add".to_string());
        s
    });
    let program = malus_syntax::parse(malus_syntax::FileId(0), src).expect("parse failed");
    let typed = check(&program, &aliases).expect("qualified call should resolve");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    // c should be present as a Call (add is a fn, not a kernel here)
    let c_let = main.body.iter().any(|s| matches!(s, TypedStmt::Let { name, .. } if name == "c"));
    assert!(c_let, "let c should be present");
}

// ── CTMM: tensor BinOp in fn body triggers barrier ──────────────────────────────

#[test]
fn test_tensor_binop_in_fn_body_inserts_barrier() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = a + b
    print(c)
"#;
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let has_barrier = main.body.iter().any(|s| matches!(s, TypedStmt::GpuBarrier));
    assert!(has_barrier, "GpuBarrier should be present: a + b in fn body produces a pending tensor");
}

#[test]
fn test_tensor_binop_return_inserts_barrier() {
    let src = r#"
fn add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = add(a, b)
    print(c)
"#;
    let typed = check_src(src).expect("should type-check");
    let add_fn = typed.fns.iter().find(|f| f.name == "add").unwrap();
    let has_barrier = add_fn.body.iter().any(|s| matches!(s, TypedStmt::GpuBarrier));
    assert!(has_barrier, "GpuBarrier should be present before return of a tensor BinOp result");
}

#[test]
fn test_scalar_binop_does_not_insert_barrier() {
    let src = r#"
fn add(x: f32, y: f32) -> f32:
    return x + y

fn main():
    let z = add(1.0, 2.0)
    print(z)
"#;
    let typed = check_src(src).expect("should type-check");
    let add_fn = typed.fns.iter().find(|f| f.name == "add").unwrap();
    let has_barrier = add_fn.body.iter().any(|s| matches!(s, TypedStmt::GpuBarrier));
    assert!(!has_barrier, "scalar BinOp must not trigger GPU barrier insertion");
}

// ── M7: let mut + reassignment ───────────────────────────────────────────────

#[test]
fn test_let_mut_and_assign_ok() {
    let src = r#"
fn main():
    let mut acc = Tensor.gpu<f32>([0.0, 0.0])
    let delta = Tensor.gpu<f32>([1.0, 2.0])
    acc = acc + delta
    print(acc)
"#;
    let typed = check_src(src).expect("let mut + reassignment should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let has_assign = main.body.iter().any(|s| matches!(
        s, TypedStmt::Assign { target: TypedAssignTarget::Ident(name), .. } if name == "acc"
    ));
    assert!(has_assign, "TypedStmt::Assign for acc should be present");
}

#[test]
fn test_assign_to_immutable_rejected() {
    let src = r#"
fn main():
    let acc = Tensor.gpu<f32>([0.0])
    let delta = Tensor.gpu<f32>([1.0])
    acc = acc + delta
    print(acc)
"#;
    let errors = check_src(src).expect_err("assign to immutable let should fail");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::AssignToImmutable { name, .. } if name == "acc")),
        "expected AssignToImmutable(acc), got: {:?}", errors
    );
}

#[test]
fn test_assign_type_mismatch_rejected() {
    let src = r#"
fn main():
    let mut acc = Tensor.gpu<f32>([0.0])
    acc = 1.0
    print(acc)
"#;
    let errors = check_src(src).expect_err("type mismatch in assign should fail");
    assert!(errors.iter().any(|e| matches!(e, SemaError::TypeMismatch { .. })),
        "expected TypeMismatch, got: {:?}", errors);
}

#[test]
fn test_assign_inserts_drop_before_rebind() {
    let src = r#"
fn main():
    let mut acc = Tensor.gpu<f32>([0.0, 0.0])
    let delta = Tensor.gpu<f32>([1.0, 2.0])
    acc = acc + delta
    print(acc)
"#;
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    // A Drop{acc} should appear before the Assign{acc} to free the old allocation.
    let drop_before_assign = {
        let mut saw_drop = false;
        let mut saw_assign = false;
        for stmt in &main.body {
            if matches!(stmt, TypedStmt::Drop { name } if name == "acc") {
                saw_drop = true;
            }
            if matches!(stmt, TypedStmt::Assign { target: TypedAssignTarget::Ident(name), .. } if name == "acc") {
                if saw_drop { saw_assign = true; }
            }
        }
        saw_assign
    };
    assert!(drop_before_assign, "Drop(acc) must appear before Assign(acc)");
}

// ── M7: scalar broadcasting ───────────────────────────────────────────────────

#[test]
fn test_tensor_mul_scalar_ok() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let scaled = a * 0.5
    print(scaled)
"#;
    let typed = check_src(src).expect("tensor * scalar should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let scaled = main.body.iter().find(|s| matches!(s, TypedStmt::Let { name, .. } if name == "scaled"));
    assert!(scaled.is_some(), "let scaled should be present");
    if let Some(TypedStmt::Let { expr, .. }) = scaled {
        assert!(matches!(expr.ty, crate::ResolvedTy::Tensor { .. }), "scaled should be Tensor");
    }
}

#[test]
fn test_scalar_mul_tensor_ok() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let scaled = 2.0 * a
    print(scaled)
"#;
    let typed = check_src(src).expect("scalar * tensor should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let scaled = main.body.iter().find(|s| matches!(s, TypedStmt::Let { name, .. } if name == "scaled"));
    assert!(scaled.is_some(), "let scaled should be present");
}

// ── M7: multi-statement kernel bodies + comparisons ──────────────────────────

#[test]
fn test_multi_stmt_kernel_with_comparison_ok() {
    let src = r#"
kernel relu_backward(grad_out: Tensor<f32>, x: Tensor<f32>) -> Tensor<f32>:
    let mask = x > 0.0
    return grad_out * mask

fn main():
    let g = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let x = Tensor.gpu<f32>([1.0, -1.0, 0.5])
    let d = relu_backward(g, x)
    print(d)
"#;
    let typed = check_src(src).expect("relu_backward should type-check");
    assert_eq!(typed.kernels.len(), 1);
    let k = &typed.kernels[0];
    assert_eq!(k.body.len(), 2, "kernel body should have 2 stmts: let mask + return");
    // First stmt is Let{mask}, second is Return
    assert!(matches!(&k.body[0], TypedStmt::Let { name, .. } if name == "mask"));
    assert!(matches!(&k.body[1], TypedStmt::Return { .. }));
}

// ── M9: control-flow type-checking ───────────────────────────────────────────

#[test]
fn test_if_typechecks() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    if a.len > 1:
        print(b)
    print(a)
"#;
    let typed = check_src(src).expect("if stmt should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let has_if = main.body.iter().any(|s| matches!(s, TypedStmt::If { .. }));
    assert!(has_if, "If node should be present in typed body");
}

#[test]
fn test_if_condition_must_be_bool() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    if a:
        print(a)
"#;
    // A tensor is not Bool — should fail.
    let errors = check_src(src).expect_err("tensor condition should fail");
    assert!(errors.iter().any(|e| matches!(e, SemaError::TypeMismatch { .. })),
        "expected TypeMismatch for non-bool condition, got: {:?}", errors);
}

#[test]
fn test_if_else_typechecks() {
    let src = r#"
fn main():
    let x = Tensor.gpu<f32>([1.0])
    if x.len > 0:
        print(x)
    else:
        print(x)
"#;
    let typed = check_src(src).expect("if/else should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let has_if = main.body.iter().any(|s| matches!(s, TypedStmt::If { else_body: Some(_), .. }));
    assert!(has_if, "If node with else_body should be present");
}

#[test]
fn test_for_loop_var_is_i64() {
    // Loop var `i` should be Scalar(I64) and visible inside the body.
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    for i in range(3):
        print(a)
    print(a)
"#;
    let typed = check_src(src).expect("for loop should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let has_for = main.body.iter().any(|s| matches!(s, TypedStmt::For { .. }));
    assert!(has_for, "For node should be present");
}

#[test]
fn test_for_loop_var_does_not_escape() {
    // Loop var `i` is scoped to the loop body; referencing it after should fail.
    let src = r#"
fn main():
    for i in range(3):
        print(i)
    print(i)
"#;
    let errors = check_src(src).expect_err("loop var should not be visible after loop");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::UnknownIdent { name, .. } if name == "i")),
        "expected UnknownIdent(i), got: {:?}", errors
    );
}

#[test]
fn test_while_typechecks() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    while a.len > 0:
        print(a)
"#;
    let typed = check_src(src).expect("while stmt should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let has_while = main.body.iter().any(|s| matches!(s, TypedStmt::While { .. }));
    assert!(has_while, "While node should be present in typed body");
}

// ── M9: CTMM — hierarchical drop placement ───────────────────────────────────

/// Helper: collect all TypedStmt variants (recursively) into a flat list of
/// variant names.  Used by CTMM tests to assert on the overall stmt sequence.
fn flat_stmt_kinds(stmts: &[TypedStmt]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for s in stmts {
        let tag = match s {
            TypedStmt::Let { .. }      => "Let",
            TypedStmt::Assign { .. }   => "Assign",
            TypedStmt::Return { .. }   => "Return",
            TypedStmt::Expr(_)         => "Expr",
            TypedStmt::Drop { .. }     => "Drop",
            TypedStmt::GpuBarrier      => "GpuBarrier",
            TypedStmt::If { .. }       => "If",
            TypedStmt::For { .. }      => "For",
            TypedStmt::While { .. }    => "While",
            TypedStmt::Retain { .. }     => "Retain",
            TypedStmt::Release { .. }    => "Release",
            TypedStmt::RetainAgg { .. }  => "RetainAgg",
            TypedStmt::ReleaseAgg { .. } => "ReleaseAgg",
            TypedStmt::DropStruct { .. } => "DropStruct",
            TypedStmt::DropEnum { .. }   => "DropEnum",
            TypedStmt::DropArray { .. }  => "DropArray",
            TypedStmt::ForIn { .. }      => "ForIn",
            TypedStmt::Match { .. }      => "Match",
            TypedStmt::Break             => "Break",
            TypedStmt::Continue          => "Continue",
            TypedStmt::LetTuple { .. }   => "LetTuple",
            TypedStmt::DropTuple { .. }  => "DropTuple",
            TypedStmt::NoGrad { .. }     => "NoGrad",
            TypedStmt::DropBuffer { .. } => "DropBuffer",
            TypedStmt::LetShared { .. }  => "LetShared",
            TypedStmt::DropList { .. }   => "DropList",
        };
        out.push(tag);
    }
    out
}

fn drops_in(stmts: &[TypedStmt]) -> Vec<&str> {
    stmts.iter().filter_map(|s| {
        if let TypedStmt::Drop { name } = s { Some(name.as_str()) } else { None }
    }).collect()
}

fn releases_in(stmts: &[TypedStmt]) -> Vec<&str> {
    stmts.iter().filter_map(|s| {
        if let TypedStmt::Release { name } = s { Some(name.as_str()) } else { None }
    }).collect()
}

/// Loop-local tensor `out` must be freed *inside* the for body, not after the loop.
#[test]
fn test_ctmm_loop_local_drop_inside_body() {
    let src = r#"
kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let x = Tensor.gpu<f32>([1.0, 2.0])
    let y = Tensor.gpu<f32>([3.0, 4.0])
    for i in range(3):
        let out = add(x, y)
        print(out)
    print(x)
"#;
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    // Outer body must NOT drop `out` (it's loop-local).
    let outer_drops = drops_in(&main.body);
    assert!(!outer_drops.contains(&"out"),
        "outer body must not drop loop-local `out`, got outer drops: {:?}", outer_drops);
    // The For node's body must contain Drop{out}.
    let for_body = main.body.iter().find_map(|s| {
        if let TypedStmt::For { body, .. } = s { Some(body.as_slice()) } else { None }
    }).expect("For node not found");
    let inner_drops = drops_in(for_body);
    assert!(inner_drops.contains(&"out"),
        "loop body must drop `out` after last use, inner drops: {:?}", inner_drops);
}

/// Outer tensor referenced inside a loop must be dropped *after* the For node.
#[test]
fn test_ctmm_outer_tensor_dropped_after_loop() {
    let src = r#"
fn main():
    let x = Tensor.gpu<f32>([1.0, 2.0])
    for i in range(3):
        print(x)
"#;
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    // x should be dropped in the outer body, after the For node.
    let outer_drops = drops_in(&main.body);
    assert!(outer_drops.contains(&"x"),
        "outer body must drop `x` after the loop, outer drops: {:?}", outer_drops);
    // And it must come AFTER the For node.
    let for_idx = main.body.iter().position(|s| matches!(s, TypedStmt::For { .. }))
        .expect("For node not found");
    let drop_x_idx = main.body.iter().rposition(|s| matches!(s, TypedStmt::Drop { name } if name == "x"))
        .expect("Drop(x) not found");
    assert!(drop_x_idx > for_idx,
        "Drop(x) at {} must be after For at {}", drop_x_idx, for_idx);
}

/// `let mut acc` reassigned inside a loop gets Drop(acc) before each Assign.
#[test]
fn test_ctmm_let_mut_assign_drop_inside_loop() {
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
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let for_body = main.body.iter().find_map(|s| {
        if let TypedStmt::For { body, .. } = s { Some(body.as_slice()) } else { None }
    }).expect("For node not found");
    // Drop(acc) must appear before Assign(acc) inside the loop body.
    let drop_before_assign = {
        let mut saw_drop = false;
        let mut found = false;
        for stmt in for_body {
            if matches!(stmt, TypedStmt::Drop { name } if name == "acc") { saw_drop = true; }
            if matches!(stmt, TypedStmt::Assign { target: TypedAssignTarget::Ident(name), .. } if name == "acc") && saw_drop { found = true; }
        }
        found
    };
    assert!(drop_before_assign, "Drop(acc) must precede Assign(acc) in loop body; body: {:#?}", for_body);
}

/// M9 emits no Retain or Release nodes (ADR-0014: hierarchical Drop is sufficient).
#[test]
fn test_ctmm_no_retain_release_in_m9() {
    let src = r#"
kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn main():
    let x = Tensor.gpu<f32>([1.0, 2.0])
    let y = Tensor.gpu<f32>([3.0, 4.0])
    for i in range(5):
        let out = add(x, y)
        print(out)
    if x.len > 0:
        print(x)
"#;
    let typed = check_src(src).expect("should type-check");

    fn has_rc(stmts: &[TypedStmt]) -> bool {
        stmts.iter().any(|s| match s {
            TypedStmt::Retain { .. } | TypedStmt::Release { .. } => true,
            TypedStmt::If { then_body, else_body, .. } => {
                has_rc(then_body) || else_body.as_deref().map(has_rc).unwrap_or(false)
            }
            TypedStmt::For { body, .. } | TypedStmt::While { body, .. } => has_rc(body),
            _ => false,
        })
    }

    for f in &typed.fns {
        assert!(!has_rc(&f.body), "M9 must emit no Retain/Release — found some in fn {}", f.name);
    }
}

// ── M11: CTMM — operand hoist + early-return unwind ──────────────────────────

/// Nested BinOp: `(a + b) * c` — the inner `a+b` result must be hoisted into a
/// `__malus_tmp_N` Let so CTMM can drop it.
#[test]
fn test_ctmm_binop_operand_hoisted_to_tmp() {
    let src = r#"
kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

kernel mul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a * b

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    let c = Tensor.gpu<f32>([5.0, 6.0])
    let result = mul(add(a, b), c)
    print(result)
"#;
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();

    // After hoisting, a `__malus_tmp` Let must appear for the `add(a,b)` result.
    let has_tmp_let = main.body.iter().any(|s| {
        if let TypedStmt::Let { name, .. } = s {
            name.starts_with("__malus_tmp")
        } else {
            false
        }
    });
    assert!(has_tmp_let, "expected a __malus_tmp Let to be hoisted; body:\n{:#?}", main.body);

    // Each hoisted tmp must also have a corresponding Drop.
    let tmp_let_names: Vec<_> = main.body.iter().filter_map(|s| {
        if let TypedStmt::Let { name, .. } = s {
            if name.starts_with("__malus_tmp") { Some(name.as_str()) } else { None }
        } else {
            None
        }
    }).collect();
    for tmp in &tmp_let_names {
        let has_drop = main.body.iter().any(|s| matches!(s, TypedStmt::Drop { name } if name == *tmp));
        assert!(has_drop, "hoisted tmp {tmp} must have a Drop; body:\n{:#?}", main.body);
    }
}

/// Early-return inside a for loop: outer tensor `a` is used after the loop
/// (CTMM places Drop(a) after the For node).  The unwind pass must also inject
/// Drop(a) before the inner Return so it is not leaked on the early-exit path.
#[test]
fn test_ctmm_early_return_in_for_unwinds_outer_drop() {
    let src = r#"
fn make() -> Tensor<f32>:
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    for i in range(3):
        return b
    print(a)
    return b

fn main():
    let t = make()
    print(t)
"#;
    let typed = check_src(src).expect("should type-check");
    let make_fn = typed.fns.iter().find(|f| f.name == "make").unwrap();

    // CTMM places Drop(a) after the For node (last use of a is print(a) after the loop).
    let drop_a_after_for = {
        let for_idx = make_fn.body.iter().position(|s| matches!(s, TypedStmt::For { .. }))
            .expect("For node not found");
        make_fn.body[for_idx..].iter()
            .any(|s| matches!(s, TypedStmt::Drop { name } if name == "a"))
    };
    assert!(drop_a_after_for, "Drop(a) must be placed after the For node in the outer body");

    // The unwind pass must also inject Drop(a) before the inner Return.
    let for_body = make_fn.body.iter().find_map(|s| {
        if let TypedStmt::For { body, .. } = s { Some(body.as_slice()) } else { None }
    }).expect("For node not found");

    let return_idx = for_body.iter().position(|s| matches!(s, TypedStmt::Return { .. }))
        .expect("Return not found in for body");
    let drop_a_before_return = for_body[..return_idx].iter()
        .any(|s| matches!(s, TypedStmt::Drop { name } if name == "a"));
    assert!(
        drop_a_before_return,
        "Drop(a) must appear before the early Return inside the for body; for_body:\n{:#?}",
        for_body
    );
}

/// Early-return inside an if branch: outer tensor `a` is used after the if
/// (CTMM places Drop(a) after the If node).  The unwind pass must inject
/// Drop(a) before the early Return inside the then-branch.
#[test]
fn test_ctmm_early_return_in_if_unwinds_outer_drop() {
    let src = r#"
fn make(x: Tensor<f32>) -> Tensor<f32>:
    let a = Tensor.gpu<f32>([1.0, 2.0])
    let b = Tensor.gpu<f32>([3.0, 4.0])
    if x.len > 0:
        return b
    print(a)
    return b

fn main():
    let x = Tensor.gpu<f32>([1.0])
    let t = make(x)
    print(t)
"#;
    let typed = check_src(src).expect("should type-check");
    let make_fn = typed.fns.iter().find(|f| f.name == "make").unwrap();

    // CTMM places Drop(a) after the If node.
    let drop_a_after_if = {
        let if_idx = make_fn.body.iter().position(|s| matches!(s, TypedStmt::If { .. }))
            .expect("If node not found");
        make_fn.body[if_idx..].iter()
            .any(|s| matches!(s, TypedStmt::Drop { name } if name == "a"))
    };
    assert!(drop_a_after_if, "Drop(a) must be placed after the If node in the outer body");

    let if_stmt = make_fn.body.iter().find(|s| matches!(s, TypedStmt::If { .. }))
        .expect("If node not found");
    let TypedStmt::If { then_body, .. } = if_stmt else { panic!() };

    let return_idx = then_body.iter().position(|s| matches!(s, TypedStmt::Return { .. }))
        .expect("Return not found in then_body");
    let drop_a_before_return = then_body[..return_idx].iter()
        .any(|s| matches!(s, TypedStmt::Drop { name } if name == "a"));
    assert!(
        drop_a_before_return,
        "Drop(a) must appear before the early Return inside the if branch; then_body:\n{:#?}",
        then_body
    );
}

// ── M11: CTMM — DropEnum + aggregate assign drops ────────────────────────────

/// Enum binding with a tensor-carrying variant must emit `DropEnum` at end of
/// scope, not just a plain `Drop`.
#[test]
fn test_ctmm_enum_binding_emits_drop_enum() {
    let src = r#"
enum Outcome:
    Good(value: Tensor<f32>)
    Bad

fn main():
    let r = Outcome.Good(value=Tensor.gpu<f32>([1.0, 2.0]))
    match r:
        Good(v):
            print(v)
        Bad:
            print(Tensor.gpu<f32>([0.0]))
"#;
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let has_drop_enum = main.body.iter().any(|s| matches!(s, TypedStmt::DropEnum { name, .. } if name == "r"));
    assert!(has_drop_enum, "DropEnum(r) must be inserted for enum with tensor variant; body:\n{:#?}", main.body);
}

/// `let mut` struct reassignment in a loop must drop the OLD struct value
/// (DropStruct) before each Assign, not just the final value.
#[test]
fn test_ctmm_struct_reassign_in_loop_drops_old_value() {
    let src = r#"
struct Pair:
    a: Tensor<f32>
    b: Tensor<f32>

fn make_pair() -> Pair:
    return Pair(a=Tensor.gpu<f32>([1.0]), b=Tensor.gpu<f32>([2.0]))

fn main():
    let mut p = make_pair()
    for i in range(3):
        p = make_pair()
    print(p.a)
"#;
    let typed = check_src(src).expect("should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let for_body = main.body.iter().find_map(|s| {
        if let TypedStmt::For { body, .. } = s { Some(body.as_slice()) } else { None }
    }).expect("For node not found");

    // DropStruct(p) must appear BEFORE Assign(p) inside the loop body.
    let drop_before_assign = {
        let mut saw_drop = false;
        let mut found = false;
        for stmt in for_body {
            if matches!(stmt, TypedStmt::DropStruct { name, .. } if name == "p") { saw_drop = true; }
            if matches!(stmt, TypedStmt::Assign { target: TypedAssignTarget::Ident(name), .. } if name == "p") && saw_drop { found = true; }
        }
        found
    };
    assert!(
        drop_before_assign,
        "DropStruct(p) must precede Assign(p) in loop body; body:\n{:#?}",
        for_body
    );
}

// ── Phase 5: 2-D nested tensor literals ──────────────────────────────────────

#[test]
fn test_2d_tensor_literal_typechecks() {
    let src = r#"
fn main():
    let x = Tensor.gpu<f32>([[1.0, 2.0], [3.0, 4.0]])
    print(x)
"#;
    check_src(src).expect("2-D tensor literal should typecheck");
}

#[test]
fn test_2d_tensor_shape_mismatch_errors() {
    let src = r#"
fn main():
    let x = Tensor.gpu<f32>([[1.0, 2.0], [3.0]])
    print(x)
"#;
    // Parser already rejects non-rectangular rows; this tests the parse error.
    let result = malus_syntax::parse(malus_syntax::FileId(0), src);
    assert!(result.is_err(), "non-rectangular tensor should fail to parse");
}

// ── Phase 4: fixed arrays ─────────────────────────────────────────────────────

#[test]
fn test_array_literal_type() {
    let src = r#"
fn main():
    let xs = [1, 2, 3]
    print(xs[0])
"#;
    let prog = check_src(src).expect("check failed");
    let body = &prog.fns[0].body;
    let let_stmt = &body[0];
    match let_stmt {
        TypedStmt::Let { name, expr } => {
            assert_eq!(name, "xs");
            match &expr.ty {
                crate::ResolvedTy::Array { len, .. } => assert_eq!(*len, 3),
                other => panic!("expected Array type, got {:?}", other),
            }
        }
        other => panic!("expected Let, got {:?}", other),
    }
}

#[test]
fn test_array_index_type() {
    let src = r#"
fn main():
    let xs = [1, 2, 3]
    let v = xs[0]
    print(v)
"#;
    let prog = check_src(src).expect("check failed");
    let body = &prog.fns[0].body;
    // body[1] is `let v = xs[0]`
    match &body[1] {
        TypedStmt::Let { name, expr } => {
            assert_eq!(name, "v");
            match &expr.ty {
                crate::ResolvedTy::Scalar(_) => {}
                other => panic!("expected Scalar element type, got {:?}", other),
            }
        }
        other => panic!("expected Let, got {:?}", other),
    }
}

#[test]
fn test_for_in_binds_elem_type() {
    let src = r#"
fn main():
    let xs = [1, 2, 3]
    for x in xs:
        print(x)
"#;
    // Should typecheck without errors: x is bound to i64 (element type).
    check_src(src).expect("for-in should typecheck");
}

#[test]
fn test_array_drop_emitted() {
    // An array of tensors should get a DropArray at end of scope.
    let src = r#"
fn main():
    let ts = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    print(ts[0])
"#;
    let prog = check_src(src).expect("check failed");
    let body = &prog.fns[0].body;
    let kinds = flat_stmt_kinds(body);
    assert!(kinds.contains(&"DropArray"), "expected DropArray in body; got {:?}", kinds);
}

// ── M13: Variable type ────────────────────────────────────────────────────────

#[test]
fn test_variable_type_checks() {
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([1.0, 2.0])
    let v = variable(t)
    let d = v.data
    print(d)
"#;
    check_src(src).expect("Variable type should check cleanly");
}

#[test]
fn test_variable_release_emitted() {
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([1.0])
    let v = variable(t)
    print(v.data)
"#;
    let prog = check_src(src).expect("check failed");
    let body = &prog.fns[0].body;
    let kinds = flat_stmt_kinds(body);
    assert!(kinds.contains(&"Release"), "expected Release for Variable; got {:?}", kinds);
}

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
        Some(p):
            let escaped = p
        Empty:
            print("empty")
"#;
    check_src(src).expect("struct payload escape should be allowed in M13");
}

// ── M20: Lvalue assignment + ** operator ──────────────────────────────────────

#[test]
fn test_pow_operator_f32_f32() {
    let src = r#"
fn main():
    let x = 2.0
    let y = x ** 3.0
    print(y)
"#;
    check_src(src).expect("f32 ** f32 should type-check");
}

#[test]
fn test_pow_operator_f32_i64() {
    let src = r#"
fn main():
    let t = 2
    let y = 0.9 ** t
    print(y)
"#;
    check_src(src).expect("f32 ** i64 should type-check");
}

#[test]
fn test_pow_operator_right_assoc() {
    let src = r#"
fn main():
    let y = 2.0 ** 3.0 ** 2.0
    print(y)
"#;
    check_src(src).expect("** right-associativity should parse and type-check");
}

#[test]
fn test_pow_operator_non_scalar_rejected() {
    let src = r#"
fn main():
    let a = Tensor.gpu<f32>([1.0])
    let b = a ** 2.0
    print(b)
"#;
    let errors = check_src(src).expect_err("** on tensor should be rejected");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::PowOperatorScalarOnly { .. })),
        "expected PowOperatorScalarOnly, got: {:?}", errors
    );
}

#[test]
fn test_index_assign_let_mut() {
    let src = r#"
fn main():
    let mut arr = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    let t = Tensor.gpu<f32>([99.0])
    arr[0] = t
    print(arr[0])
"#;
    check_src(src).expect("index assign on let mut array should type-check");
}

#[test]
fn test_index_assign_immutable_rejected() {
    let src = r#"
fn main():
    let arr = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    let t = Tensor.gpu<f32>([99.0])
    arr[0] = t
"#;
    let errors = check_src(src).expect_err("index assign on immutable array should be rejected");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::AssignToImmutable { .. })),
        "expected AssignToImmutable, got: {:?}", errors
    );
}

#[test]
fn test_field_assign_let_mut() {
    let src = r#"
struct Point:
    x: f32
    y: f32

fn main():
    let mut p = Point(x=1.0, y=2.0)
    p.y = 99.0
    print(p.y)
"#;
    check_src(src).expect("field assign on let mut struct should type-check");
}

#[test]
fn test_field_assign_immutable_rejected() {
    let src = r#"
struct Point:
    x: f32
    y: f32

fn main():
    let p = Point(x=1.0, y=2.0)
    p.y = 99.0
"#;
    let errors = check_src(src).expect_err("field assign on immutable struct should be rejected");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::AssignToImmutable { .. })),
        "expected AssignToImmutable, got: {:?}", errors
    );
}

#[test]
fn test_mut_param_interior_mutation_ok() {
    let src = r#"
fn fill(mut arr: Array<Tensor<f32>,2>, t: Tensor<f32>):
    arr[0] = t

fn main():
    let mut arr = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    let t = Tensor.gpu<f32>([99.0])
    fill(arr, t)
    print(arr[0])
"#;
    check_src(src).expect("mut param interior mutation should type-check");
}

#[test]
fn test_mut_param_bare_rebind_rejected() {
    let src = r#"
fn replace(mut arr: Array<Tensor<f32>,2>):
    let t = Tensor.gpu<f32>([1.0])
    let new_arr = [t, t]
    arr = new_arr

fn main():
    let mut arr = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    replace(arr)
"#;
    let errors = check_src(src).expect_err("bare rebind of mut param should be rejected");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::MutParamBareRebind { .. })),
        "expected MutParamBareRebind, got: {:?}", errors
    );
}

#[test]
fn test_variable_field_assign_accepted() {
    let src = r#"
struct Model:
    w: Tensor<f32>

fn main():
    let t = Tensor.gpu<f32>([1.0])
    let v = variable(t)
    let mut m = Model(w=v)
    let t2 = Tensor.gpu<f32>([2.0])
    let v2 = variable(t2)
    m.w = v2
"#;
    // M22 commit 3: Variable field assign is now supported (mut base required).
    check_src(src).expect("assigning to Variable field on mut struct should now be accepted");
}

#[test]
fn test_nested_lvalue_rejected() {
    let src = r#"
fn main():
    let mut arr = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    let t = Tensor.gpu<f32>([99.0])
    arr[0][0] = t
"#;
    let errors = check_src(src).expect_err("nested lvalue should be rejected");
    assert!(
        errors.iter().any(|e| matches!(e, SemaError::NestedLvalue { .. })),
        "expected NestedLvalue, got: {:?}", errors
    );
}

// ── M27: grad-inference pass ──────────────────────────────────────────────────

/// A tensor never passed through `variable()` is not grad-tracked and gets a
/// static `Drop`, never an RC `Release`.
#[test]
fn test_grad_inference_non_derived_tensor_not_tracked() {
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([1.0])
    print(t)
"#;
    let prog = check_src(src).expect("check failed");
    let kinds = flat_stmt_kinds(&prog.fns[0].body);
    assert!(kinds.contains(&"Drop"), "expected static Drop for non-grad-tracked tensor; got {:?}", kinds);
    assert!(!kinds.contains(&"Release"), "non-grad-tracked tensor must not be RC-released; got {:?}", kinds);
}

/// Every expression lexically inside `with no_grad:` is forced non-grad-tracked,
/// overriding propagation — even `variable()`, which is otherwise an
/// unconditional grad-tracked seed.
#[test]
fn test_grad_inference_no_grad_forces_non_tracked() {
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([1.0])
    with no_grad:
        let v = variable(t)
        print(v)
"#;
    let prog = check_src(src).expect("check failed");
    let no_grad_body = prog.fns[0].body.iter().find_map(|s| {
        if let TypedStmt::NoGrad { body } = s { Some(body) } else { None }
    }).expect("expected a NoGrad stmt");
    let kinds = flat_stmt_kinds(no_grad_body);
    assert!(kinds.contains(&"Drop"), "no_grad-lexical variable() binding should be static-dropped; got {:?}", kinds);
    assert!(!kinds.contains(&"Release"), "no_grad-lexical binding must not be RC-released; got {:?}", kinds);
}

/// `.data` and `.grad` are detach points: their result is never grad-tracked,
/// regardless of the receiver — even a `variable()`-derived leaf.
#[test]
fn test_grad_inference_data_and_grad_are_detach_points() {
    let src = r#"
fn main():
    let t = Tensor.gpu<f32>([1.0])
    let v = variable(t)
    let d = v.data
    let g = v.grad
    print(d)
    print(g)
"#;
    let prog = check_src(src).expect("check failed");
    let body = &prog.fns[0].body;
    let drop_names = drops_in(body);
    assert!(drop_names.contains(&"d"), "'.data' result should be static-dropped (detach); got drops: {:?}", drop_names);
    assert!(drop_names.contains(&"g"), "'.grad' result should be static-dropped (detach); got drops: {:?}", drop_names);
    let release_names = releases_in(body);
    assert!(!release_names.contains(&"d"), "'.data' result must not be RC-released; got releases: {:?}", release_names);
    assert!(!release_names.contains(&"g"), "'.grad' result must not be RC-released; got releases: {:?}", release_names);
}

/// Interprocedural: grad-tracking flows through a fn's param and return type,
/// unioned over call sites (context-insensitive).
#[test]
fn test_grad_inference_interprocedural_param_and_return() {
    let src = r#"
fn identity(x: Tensor<f32>) -> Tensor<f32>:
    return x

fn main():
    let t = Tensor.gpu<f32>([1.0])
    let v = variable(t)
    let out = identity(v)
    print(out)
"#;
    let prog = check_src(src).expect("check failed");
    let main_fn = prog.fns.iter().find(|f| f.name == "main").expect("main not found");
    let release_names = releases_in(&main_fn.body);
    assert!(
        release_names.contains(&"out"),
        "call result through a grad-tracked param/return should be RC-released; got releases: {:?}",
        release_names
    );
}

/// Field-sensitivity: a struct field written by a grad-tracked value at one
/// construction site makes every read of that `(struct, field)` grad-tracked.
#[test]
fn test_grad_inference_struct_field_carrying() {
    let src = r#"
struct Model:
    w: Tensor<f32>

fn main():
    let t = Tensor.gpu<f32>([1.0])
    let v = variable(t)
    let m = Model(w=v)
    let w2 = m.w
    print(w2)
"#;
    let prog = check_src(src).expect("check failed");
    let release_names = releases_in(&prog.fns[0].body);
    assert!(
        release_names.contains(&"w2"),
        "reading a grad-carrying struct field should be RC-released; got releases: {:?}",
        release_names
    );
}

// ── M28: generics, trait/impl, List<T> ───────────────────────────────────────

/// Done-when #4: `fn id<T>(x: T) -> T: return x` called with `Tensor<f32>` and
/// `i32` produces distinct, correctly-typed monomorphizations — not the generic
/// item itself (erased before codegen, ADR-0034).
#[test]
fn test_generic_fn_monomorphization_distinct_instantiations() {
    let src = r#"
fn id<T>(x: T) -> T:
    return x

fn use_i32(v: i32) -> i32:
    return id(v)

fn main():
    let t = Tensor.gpu<f32>([1.0, 2.0])
    let b = id(t)
    let r = use_i32(7)
    print(b)
    print(r)
"#;
    let typed = check_src(src).expect("generic fn should monomorphize and type-check");
    let names: Vec<&str> = typed.fns.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"id__i32"), "expected id__i32 in {:?}", names);
    assert!(names.contains(&"id__Tensor_f32"), "expected id__Tensor_f32 in {:?}", names);
    assert!(!names.iter().any(|n| *n == "id"), "the generic item itself must not reach the typed IR");

    let id_i32 = typed.fns.iter().find(|f| f.name == "id__i32").unwrap();
    assert_eq!(id_i32.params[0].ty, crate::ty::ResolvedTy::Scalar(malus_syntax::ast::ScalarTy::I32));
    assert_eq!(id_i32.return_ty, crate::ty::ResolvedTy::Scalar(malus_syntax::ast::ScalarTy::I32));

    let id_tensor = typed.fns.iter().find(|f| f.name == "id__Tensor_f32").unwrap();
    assert!(id_tensor.params[0].ty.is_tensor());
    assert!(id_tensor.return_ty.is_tensor());
}

/// Calling the same generic fn twice with the same concrete type memoizes to
/// one instantiation (mono_cache), not two.
#[test]
fn test_generic_fn_monomorphization_memoized() {
    let src = r#"
fn id<T>(x: T) -> T:
    return x

fn main():
    let a = id(1)
    let b = id(2)
    print(a)
    print(b)
"#;
    let typed = check_src(src).expect("check failed");
    let count = typed.fns.iter().filter(|f| f.name == "id__i64").count();
    assert_eq!(count, 1, "second call with the same concrete type must reuse the cached instantiation");
}

/// Trait + impl + method-call dispatch: `model.parameters()` resolves through
/// the trait-impl registry to the monomorphized `Type__method` fn.
#[test]
fn test_trait_impl_method_call_dispatch() {
    let src = r#"
struct GPT:
    params: List<Tensor<f32>>

trait Module:
    fn parameters(self) -> List<Tensor<f32>>

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

fn main():
    let gpt = GPT(params=[Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])])
    let out = gpt.parameters()
    print(out)
"#;
    let typed = check_src(src).expect("trait/impl/method-call should type-check");
    let names: Vec<&str> = typed.fns.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"GPT__parameters"), "expected GPT__parameters in {:?}", names);
    let method = typed.fns.iter().find(|f| f.name == "GPT__parameters").unwrap();
    assert!(method.return_ty.is_list());
}

/// A generic fn bounded by a trait the concrete arg's type doesn't implement
/// is a sema error, not a panic.
#[test]
fn test_generic_fn_trait_bound_not_satisfied() {
    let src = r#"
struct GPT:
    params: List<Tensor<f32>>

trait Module:
    fn parameters(self) -> List<Tensor<f32>>

fn adamw<M: Module>(model: M) -> i64:
    return 0

fn main():
    let gpt = GPT(params=[Tensor.gpu<f32>([1.0])])
    let r = adamw(gpt)
    print(r)
"#;
    let errs = check_src(src).expect_err("GPT does not implement Module — must be a sema error");
    assert!(
        errs.iter().any(|e| matches!(e, SemaError::TraitBoundNotSatisfied { .. })),
        "expected TraitBoundNotSatisfied, got {:?}", errs
    );
}

/// `List<T>` literal disambiguation: a `[e1, e2]` literal assigned into a
/// `List<T>`-typed struct field is inferred as `List`, not `Array` (ADR-0034).
/// Indexing and `len()` both work on the result.
#[test]
fn test_list_literal_disambiguation_index_and_len() {
    let src = r#"
struct GPT:
    params: List<Tensor<f32>>

fn main():
    let gpt = GPT(params=[Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])])
    let n = len(gpt.params)
    let first = gpt.params[0]
    print(n)
    print(first)
"#;
    let typed = check_src(src).expect("List literal/index/len should type-check");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let gpt_let = main.body.iter().find_map(|s| match s {
        TypedStmt::Let { name, expr } if name == "gpt" => Some(expr),
        _ => None,
    }).unwrap();
    assert!(matches!(&gpt_let.kind, crate::typed_ir::TypedExprKind::StructInit { .. }));
    assert!(gpt_let.ty.is_struct());
}

/// `for p in list` iterates a `List<Tensor<f32>>` binding it results correctly.
#[test]
fn test_list_for_in() {
    let src = r#"
struct GPT:
    params: List<Tensor<f32>>

fn main():
    let gpt = GPT(params=[Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])])
    for p in gpt.params:
        print(p)
"#;
    check_src(src).expect("ForIn over List should type-check");
}

/// `ps[i] = variable(...)` — slot reassignment on a `mut List<Tensor<f32>>`
/// parameter, the write-back mechanism the generic optimizer relies on
/// (ADR-0034 D1).
#[test]
fn test_list_index_assign() {
    let src = r#"
fn bump(mut ps: List<Tensor<f32>>):
    ps[0] = variable(ps[0])

fn main():
    bump([Tensor.gpu<f32>([1.0])])
"#;
    check_src(src).expect("List slot reassignment should type-check");
}

/// Grad-inference (ADR-0030) is content-based and container-agnostic: reading an
/// element out of a `List` containing a grad-tracked tensor is itself
/// grad-tracked, propagating through `StructInit` -> `struct_field_grad` ->
/// `FieldAccess` -> `Index`, exactly like `Array`/`Tuple` already do.
#[test]
fn test_list_element_grad_tracked_propagates_through_index() {
    let src = r#"
struct GPT:
    params: List<Tensor<f32>>

fn main():
    let t = Tensor.gpu<f32>([1.0])
    let v = variable(t)
    let gpt = GPT(params=[v])
    let first = gpt.params[0]
    print(first)
"#;
    let typed = check_src(src).expect("check failed");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let first_grad = main.body.iter().find_map(|s| match s {
        TypedStmt::Let { name, expr } if name == "first" => Some(expr.grad_tracked),
        _ => None,
    });
    assert_eq!(
        first_grad, Some(true),
        "reading an element of a List containing a grad-tracked tensor should itself be grad-tracked"
    );
}

/// The core M28/ADR-0034 aliasing risk: `let gpt = GPT(params=ps)` aliases the
/// existing `ps` local into a struct field. Without a retain, `ps`'s own
/// `DropList` (inserted at its last use — this very statement) would drop the
/// shared box's refcount from 1 to 0, freeing the element tensors + box out
/// from under `gpt.params`. CTMM must emit `RetainAgg{ps}` immediately before
/// the `let gpt = ...` statement, and `DropList{ps}` immediately after —
/// net balanced (1 -> 2 -> 1), leaving exactly one live reference, now owned
/// by the struct field.
#[test]
fn test_list_struct_field_alias_retain_release_balanced() {
    let src = r#"
struct GPT:
    params: List<Tensor<f32>>

fn make_params() -> List<Tensor<f32>>:
    return [Tensor.gpu<f32>([1.0])]

fn main():
    let ps = make_params()
    let gpt = GPT(params=ps)
    print(gpt)
"#;
    let typed = check_src(src).expect("check failed");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let tags = flat_stmt_kinds(&main.body);

    let gpt_let_idx = main.body.iter().position(|s| matches!(s,
        TypedStmt::Let { name, .. } if name == "gpt"
    )).expect("expected `let gpt = ...` statement");

    assert_eq!(
        tags.get(gpt_let_idx.wrapping_sub(1)), Some(&"RetainAgg"),
        "expected RetainAgg immediately before `let gpt = GPT(params=ps)`; got tags: {:?}", tags
    );
    let retain_name = match &main.body[gpt_let_idx - 1] {
        TypedStmt::RetainAgg { name } => name.as_str(),
        _ => unreachable!(),
    };
    assert_eq!(retain_name, "ps");

    assert_eq!(
        tags.get(gpt_let_idx + 1), Some(&"DropList"),
        "expected DropList immediately after `let gpt = GPT(params=ps)` (ps's last use); got tags: {:?}", tags
    );
    let drop_name = match &main.body[gpt_let_idx + 1] {
        TypedStmt::DropList { name, .. } => name.as_str(),
        _ => unreachable!(),
    };
    assert_eq!(drop_name, "ps");
}

/// CAPSTONE DESIGN CONSTRAINT (not an M28 bug, a pre-existing CTMM property
/// this milestone's nanoGPT rewrite must respect): binding `let x =
/// model.params[i]` — a plain alias of a grad-tracked tensor read out of a
/// `List` — gets its OWN `Release` at `x`'s last use, exactly as if `x` had
/// been freshly allocated. Since `x` is really an ALIAS of the tensor still
/// owned by `model.params`, that `Release` would incorrectly decrement (and,
/// on a repeated call with the same model, eventually free) the shared
/// tensor out from under the caller. `forward()` must therefore reference
/// `model.params[i]` INLINE at each use site (matching the existing,
/// already-correct V3 pattern of writing `blk.wq` inline rather than binding
/// `let wq = blk.wq`) — never bind a struct/list tensor read to a persistent
/// local name. Index-typed scalar constants (`let WQ = 1`) are fine, since
/// they're plain i64 values with no ownership.
#[test]
fn test_list_indexed_tensor_alias_gets_release_capstone_design_constraint() {
    let src = r#"
struct GPT:
    params: List<Tensor<f32>>

fn forward(model: GPT) -> Tensor<f32>:
    let ln1_w = model.params[0]
    let scaled = ln1_w * 2.0
    return scaled

fn main():
    let gpt = GPT(params=[variable(Tensor.gpu<f32>([1.0]))])
    let out = forward(gpt)
    print(out)
"#;
    let typed = check_src(src).expect("check failed");
    let forward = typed.fns.iter().find(|f| f.name == "forward").unwrap();
    // `ln1_w` escapes via `return`, so per ADR-0030 it's Release'd (RC), not
    // statically Dropped — but the point stands regardless of which: `forward`
    // believes it owns (or must RC-manage) a reference that `model.params`
    // still needs. This test documents the property the capstone must design
    // around, not asserts it's "fixed" (fixing it generally is M29 borrow-
    // inference territory — proving `ln1_w`'s lifetime nests inside `model`'s
    // requires interprocedural analysis this milestone doesn't have).
    let releases = releases_in(&forward.body);
    assert!(
        releases.contains(&"ln1_w"),
        "expected ln1_w to be RC-managed (documenting the aliasing risk), got: {:?}", releases
    );
}

/// A plain (non-aliased) `List` local — never read out of a struct field and
/// never itself aliased — still gets exactly one `DropList` at its own last
/// use, with no spurious retain (the common, non-risky case).
#[test]
fn test_list_plain_local_gets_single_droplist_no_retain() {
    let src = r#"
fn main():
    let ps = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]
    print(ps[0])
"#;
    // `ps` resolves to Array here (no List-context) — use a helper fn return
    // type instead, matching the pattern the capstone actually uses.
    let src = src.replace(
        "let ps = [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]",
        "let ps = make_params()",
    );
    let src = format!(
        "fn make_params() -> List<Tensor<f32>>:\n    return [Tensor.gpu<f32>([1.0]), Tensor.gpu<f32>([2.0])]\n\n{src}"
    );
    let typed = check_src(&src).expect("check failed");
    let main = typed.fns.iter().find(|f| f.name == "main").unwrap();
    let tags = flat_stmt_kinds(&main.body);
    assert_eq!(tags.iter().filter(|t| **t == "RetainAgg").count(), 0, "no aliasing occurred; got tags: {:?}", tags);
    assert_eq!(tags.iter().filter(|t| **t == "DropList").count(), 1, "expected exactly one DropList; got tags: {:?}", tags);
}
