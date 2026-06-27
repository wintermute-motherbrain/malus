use crate::ast::*;
use std::fmt;
use std::fmt::Write;

// ── Public entry point ────────────────────────────────────────────────────────

/// Pretty-print a `Program` to a `String`. The output is valid malus source
/// that parses back to an equivalent AST (round-trip safe).
pub fn print_program(program: &Program) -> String {
    let mut out = String::new();
    let mut first = true;
    for item in &program.items {
        if !first {
            out.push('\n');
        }
        print_item(&mut out, item);
        first = false;
    }
    out
}

// ── Items ─────────────────────────────────────────────────────────────────────

fn print_item(out: &mut String, item: &Item) {
    match &item.kind {
        ItemKind::Import { path } => {
            writeln!(out, "import {}", path.segments.join(".")).unwrap();
        }
        ItemKind::FromImport { path, names } => {
            let name_list: Vec<&str> = names.iter().map(|(n, _)| n.as_str()).collect();
            writeln!(out, "from {} import {}", path.segments.join("."), name_list.join(", ")).unwrap();
        }
        ItemKind::Fn { name, params, return_ty, body } => {
            let params_str = params.iter().map(print_param).collect::<Vec<_>>().join(", ");
            let ret = return_ty.as_ref().map(|t| format!(" -> {}", print_ty(t))).unwrap_or_default();
            writeln!(out, "fn {name}({params_str}){ret}:").unwrap();
            for stmt in body {
                print_stmt(out, stmt, 1);
            }
        }
        ItemKind::Kernel { name, params, return_ty, body } => {
            let params_str = params.iter().map(print_kernel_param).collect::<Vec<_>>().join(", ");
            writeln!(out, "kernel {name}({params_str}) -> {}:", print_ty(return_ty)).unwrap();
            for stmt in body {
                print_stmt(out, stmt, 1);
            }
        }
    }
}

// ── Statements ────────────────────────────────────────────────────────────────

fn print_stmt(out: &mut String, stmt: &Stmt, depth: usize) {
    let indent = "    ".repeat(depth);
    match &stmt.kind {
        StmtKind::Let { name, expr } => {
            writeln!(out, "{indent}let {name} = {}", print_expr(expr)).unwrap();
        }
        StmtKind::LetMut { name, expr } => {
            writeln!(out, "{indent}let mut {name} = {}", print_expr(expr)).unwrap();
        }
        StmtKind::Assign { target, expr } => {
            writeln!(out, "{indent}{target} = {}", print_expr(expr)).unwrap();
        }
        StmtKind::Return { expr } => {
            writeln!(out, "{indent}return {}", print_expr(expr)).unwrap();
        }
        StmtKind::Expr(expr) => {
            writeln!(out, "{indent}{}", print_expr(expr)).unwrap();
        }
    }
}

// ── Expressions ───────────────────────────────────────────────────────────────

fn print_expr(expr: &Expr) -> String {
    match &expr.kind {
        ExprKind::Lit(lit) => print_lit(lit),
        ExprKind::Ident(name) => name.clone(),
        ExprKind::BinOp { op, lhs, rhs } => {
            let lhs_s = print_expr_parens(lhs, op);
            let rhs_s = print_expr_parens(rhs, op);
            format!("{} {} {}", lhs_s, print_binop(op), rhs_s)
        }
        ExprKind::Unary { op, operand } => {
            match op {
                UnaryOp::Neg => format!("-{}", print_expr(operand)),
                UnaryOp::Not => format!("not {}", print_expr(operand)),
            }
        }
        ExprKind::Call { callee, args } => {
            let args_str = args.iter().map(print_expr).collect::<Vec<_>>().join(", ");
            format!("{}({})", print_expr(callee), args_str)
        }
        ExprKind::Index { base, indices } => {
            let idx_str = indices.iter().map(print_expr).collect::<Vec<_>>().join(", ");
            format!("{}[{}]", print_expr(base), idx_str)
        }
        ExprKind::FieldAccess { base, field } => {
            format!("{}.{}", print_expr(base), field)
        }
        ExprKind::TensorLiteral { placement, dtype, elements } => {
            let place = match placement {
                Placement::Cpu => "cpu",
                Placement::Gpu => "gpu",
            };
            let elements_str = elements.iter().map(print_expr).collect::<Vec<_>>().join(", ");
            format!("Tensor.{}<{}>([{}])", place, print_scalar_ty(dtype), elements_str)
        }
    }
}

/// Wrap `child` in parens if its precedence is lower than the operator's expectation.
fn print_expr_parens(child: &Expr, parent_op: &BinOp) -> String {
    let needs_parens = match &child.kind {
        ExprKind::BinOp { op, .. } => binop_prec(op) < binop_prec(parent_op),
        _ => false,
    };
    if needs_parens {
        format!("({})", print_expr(child))
    } else {
        print_expr(child)
    }
}

fn binop_prec(op: &BinOp) -> u8 {
    match op {
        BinOp::Or              => 1,
        BinOp::And             => 3,
        BinOp::Eq | BinOp::NotEq
        | BinOp::Lt | BinOp::LtEq
        | BinOp::Gt | BinOp::GtEq => 5,
        BinOp::Add | BinOp::Sub   => 7,
        BinOp::Mul | BinOp::Div
        | BinOp::Matmul            => 9,
    }
}

fn print_binop(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add    => "+",
        BinOp::Sub    => "-",
        BinOp::Mul    => "*",
        BinOp::Div    => "/",
        BinOp::Matmul => "@",
        BinOp::Eq     => "==",
        BinOp::NotEq  => "!=",
        BinOp::Lt     => "<",
        BinOp::LtEq   => "<=",
        BinOp::Gt     => ">",
        BinOp::GtEq   => ">=",
        BinOp::And    => "and",
        BinOp::Or     => "or",
    }
}

// ── Literals ──────────────────────────────────────────────────────────────────

fn print_lit(lit: &Lit) -> String {
    match lit {
        Lit::Int(n)   => n.to_string(),
        Lit::Float(f) => {
            let s = format!("{}", f);
            // Ensure the output has a decimal point so it re-lexes as a float.
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{}.0", s)
            }
        }
        Lit::Bool(b)  => if *b { "true" } else { "false" }.to_string(),
        Lit::Str(s)   => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
    }
}

// ── Types ─────────────────────────────────────────────────────────────────────

fn print_ty(ty: &Ty) -> String {
    match ty {
        Ty::Tensor { dtype } => format!("Tensor<{}>", print_scalar_ty(dtype)),
        Ty::Scalar(s)        => print_scalar_ty(s).to_string(),
        Ty::Bool             => "bool".to_string(),
        Ty::Named(n)         => n.clone(),
        Ty::Tuple(types)     => {
            let inner = types.iter().map(print_ty).collect::<Vec<_>>().join(", ");
            format!("({})", inner)
        }
    }
}

fn print_scalar_ty(s: &ScalarTy) -> &'static str {
    match s {
        ScalarTy::F32  => "f32",
        ScalarTy::F16  => "f16",
        ScalarTy::Bf16 => "bf16",
        ScalarTy::I8   => "i8",
        ScalarTy::I16  => "i16",
        ScalarTy::I32  => "i32",
        ScalarTy::I64  => "i64",
        ScalarTy::U8   => "u8",
        ScalarTy::U16  => "u16",
        ScalarTy::U32  => "u32",
        ScalarTy::U64  => "u64",
    }
}

// ── Parameters ────────────────────────────────────────────────────────────────

fn print_param(p: &Param) -> String {
    format!("{}: {}", p.name, print_ty(&p.ty))
}

fn print_kernel_param(p: &KernelParam) -> String {
    let inout = if p.inout { "inout " } else { "" };
    format!("{}{}: {}", inout, p.name, print_ty(&p.ty))
}

// ── Display impls ─────────────────────────────────────────────────────────────

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", print_program(self))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, FileId, Span};

    // Byte offsets change when comments are stripped by the printer, so we
    // compare structural equality by zeroing all spans before asserting.
    fn erase_spans(prog: Program) -> Program {
        let z = Span::new(FileId(0), 0, 0);
        Program {
            items: prog.items.into_iter().map(|item| Item {
                span: z,
                kind: match item.kind {
                    ItemKind::Import { path } =>
                        ItemKind::Import { path: ModulePath { span: z, ..path } },
                    ItemKind::FromImport { path, names } =>
                        ItemKind::FromImport {
                            path: ModulePath { span: z, ..path },
                            names: names.into_iter().map(|(n, _)| (n, z)).collect(),
                        },
                    ItemKind::Fn { name, params, return_ty, body } =>
                        ItemKind::Fn {
                            name,
                            params: params.into_iter().map(|p| Param { span: z, ..p }).collect(),
                            return_ty,
                            body: body.into_iter().map(|s| erase_stmt(s, z)).collect(),
                        },
                    ItemKind::Kernel { name, params, return_ty, body } =>
                        ItemKind::Kernel {
                            name,
                            params: params.into_iter().map(|p| KernelParam { span: z, ..p }).collect(),
                            return_ty,
                            body: body.into_iter().map(|s| erase_stmt(s, z)).collect(),
                        },
                },
            }).collect(),
        }
    }

    fn erase_stmt(s: Stmt, z: Span) -> Stmt {
        Stmt {
            span: z,
            kind: match s.kind {
                StmtKind::Let { name, expr } =>
                    StmtKind::Let { name, expr: erase_expr(expr, z) },
                StmtKind::LetMut { name, expr } =>
                    StmtKind::LetMut { name, expr: erase_expr(expr, z) },
                StmtKind::Assign { target, expr } =>
                    StmtKind::Assign { target, expr: erase_expr(expr, z) },
                StmtKind::Return { expr } =>
                    StmtKind::Return { expr: erase_expr(expr, z) },
                StmtKind::Expr(expr) =>
                    StmtKind::Expr(erase_expr(expr, z)),
            },
        }
    }

    fn erase_expr(e: Expr, z: Span) -> Expr {
        let kind = match e.kind {
            ExprKind::Lit(_) | ExprKind::Ident(_) => e.kind,
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: Box::new(erase_expr(*lhs, z)),
                rhs: Box::new(erase_expr(*rhs, z)),
            },
            ExprKind::Unary { op, operand } => ExprKind::Unary {
                op,
                operand: Box::new(erase_expr(*operand, z)),
            },
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: Box::new(erase_expr(*callee, z)),
                args: args.into_iter().map(|a| erase_expr(a, z)).collect(),
            },
            ExprKind::Index { base, indices } => ExprKind::Index {
                base: Box::new(erase_expr(*base, z)),
                indices: indices.into_iter().map(|i| erase_expr(i, z)).collect(),
            },
            ExprKind::FieldAccess { base, field } => ExprKind::FieldAccess {
                base: Box::new(erase_expr(*base, z)),
                field,
            },
            ExprKind::TensorLiteral { placement, dtype, elements } => ExprKind::TensorLiteral {
                placement,
                dtype,
                elements: elements.into_iter().map(|el| erase_expr(el, z)).collect(),
            },
        };
        Expr { kind, span: z }
    }

    fn roundtrip(src: &str) {
        let prog1 = parse(FileId(0), src).unwrap_or_else(|e| {
            panic!("initial parse failed: {e}\nsource:\n{src}")
        });
        let printed = print_program(&prog1);
        let prog2 = parse(FileId(0), &printed).unwrap_or_else(|e| {
            panic!("round-trip parse failed: {e}\nprinted:\n{printed}")
        });
        assert_eq!(erase_spans(prog1), erase_spans(prog2),
            "ASTs differ after round-trip.\nOriginal:\n{src}\nPrinted:\n{printed}");
    }

    #[test]
    fn roundtrip_add_tensors_example() {
        roundtrip(include_str!("../../../examples/add_tensors.ml"));
    }

    #[test]
    fn roundtrip_fn_no_params() {
        roundtrip("fn foo():\n    return 0\n");
    }

    #[test]
    fn roundtrip_fn_with_params_and_return() {
        roundtrip("fn add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:\n    return a + b\n");
    }

    #[test]
    fn roundtrip_kernel() {
        roundtrip("kernel add(inout a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:\n    return a + b\n");
    }

    #[test]
    fn roundtrip_binop_precedence() {
        roundtrip("fn f():\n    return a + b * c\n");
    }

    #[test]
    fn roundtrip_parens_preserved() {
        roundtrip("fn f():\n    return (a + b) * c\n");
    }

    #[test]
    fn roundtrip_comparison() {
        roundtrip("fn f():\n    return a == b\n");
    }

    #[test]
    fn roundtrip_bool_ops() {
        roundtrip("fn f():\n    return a and b or c\n");
    }

    #[test]
    fn roundtrip_field_access_call() {
        roundtrip("fn f():\n    return ops.add(a, b)\n");
    }

    #[test]
    fn roundtrip_tensor_literal() {
        roundtrip("fn f():\n    let x = Tensor.gpu<f32>([1.0, 2.0, 3.0])\n    return x\n");
    }

    #[test]
    fn roundtrip_import() {
        roundtrip("import ops\n\nfn f():\n    return 0\n");
    }

    #[test]
    fn roundtrip_from_import() {
        roundtrip("from ops import add, mul\n\nfn f():\n    return 0\n");
    }

    #[test]
    fn roundtrip_string_literal() {
        roundtrip("fn f():\n    load(\"path/to/file.safetensors\")\n");
    }

    #[test]
    fn roundtrip_float_no_decimal_roundtrip() {
        // 1e4 parses as Float(10000.0); printed as "10000" which re-lexes as Int.
        // We ensure our printer always emits a decimal point.
        let prog = parse(FileId(0), "fn f():\n    return 1.0\n").unwrap();
        let printed = print_program(&prog);
        assert!(printed.contains("1.0"), "expected float with decimal: {printed}");
        roundtrip("fn f():\n    return 1.0\n");
    }

    #[test]
    fn print_program_display_impl() {
        let prog = parse(FileId(0), "fn f():\n    return 0\n").unwrap();
        let s = prog.to_string();
        assert!(s.contains("fn f()"));
        assert!(s.contains("return 0"));
    }
}
