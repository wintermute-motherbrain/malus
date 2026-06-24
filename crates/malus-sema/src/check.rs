use std::collections::{HashMap, HashSet};
use malus_syntax::ast::{
    BinOp, ExprKind, ItemKind, Lit, Placement, Program, ScalarTy, Ty, UnaryOp,
};
use malus_syntax::Span;
use crate::builtins::{BuiltinKind, register_builtins};
use crate::env::{Callee, Env, FnSig, KernelSig, KernelParamSig, ParamSig};
use crate::error::SemaError;
use crate::ty::{is_float_scalar, scalar_ty_name, ResolvedTy};
use crate::typed_ir::{
    TypedExpr, TypedExprKind, TypedFn, TypedKernel, TypedKernelParam, TypedParam, TypedProgram,
    TypedStmt,
};

// ── Context passed through body checking ──────────────────────────────────────

struct BodyCtx<'a> {
    env: &'a mut Env,
    errors: &'a mut Vec<SemaError>,
    return_ty: ResolvedTy,
    in_kernel: bool,
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn check(
    program: &Program,
    module_aliases: &HashMap<String, HashSet<String>>,
) -> Result<TypedProgram, Vec<SemaError>> {
    let builtins = register_builtins();
    let mut env = Env::new(builtins, module_aliases.clone());
    let mut errors: Vec<SemaError> = Vec::new();

    // ── Pass 1: collect all fn/kernel signatures ──────────────────────────────

    let mut has_main = false;

    for item in &program.items {
        match &item.kind {
            ItemKind::Fn { name, params, return_ty, .. } => {
                if let Some(existing) = env.functions.get(name) {
                    errors.push(SemaError::DuplicateDefinition {
                        name: name.clone(),
                        first: existing.defined_at,
                        second: item.span,
                    });
                    continue;
                }
                let resolved_params = match resolve_params(params, &mut errors) {
                    Some(p) => p,
                    None => continue,
                };
                let resolved_return = match return_ty {
                    Some(ty) => match resolve_ty(ty, item.span, &mut errors) {
                        Some(t) => t,
                        None => continue,
                    },
                    None => ResolvedTy::Unit,
                };
                if name == "main" {
                    has_main = true;
                }
                env.functions.insert(name.clone(), FnSig {
                    params: resolved_params,
                    return_ty: resolved_return,
                    defined_at: item.span,
                });
            }
            ItemKind::Kernel { name, params, return_ty, .. } => {
                if let Some(existing) = env.kernels.get(name) {
                    errors.push(SemaError::DuplicateDefinition {
                        name: name.clone(),
                        first: existing.defined_at,
                        second: item.span,
                    });
                    continue;
                }
                let resolved_params = match resolve_kernel_params(params, &mut errors) {
                    Some(p) => p,
                    None => continue,
                };
                let resolved_return = match resolve_ty(return_ty, item.span, &mut errors) {
                    Some(t) => t,
                    None => continue,
                };
                env.kernels.insert(name.clone(), KernelSig {
                    params: resolved_params,
                    return_ty: resolved_return,
                    defined_at: item.span,
                });
            }
            // Imports are already resolved by the loader — ignore them.
            ItemKind::Import { .. } | ItemKind::FromImport { .. } => {}
        }
    }

    if !has_main {
        errors.push(SemaError::MainNotFound);
    }

    // ── Pass 2: check bodies ──────────────────────────────────────────────────

    let mut typed_fns: Vec<TypedFn> = Vec::new();
    let mut typed_kernels: Vec<TypedKernel> = Vec::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Fn { name, params, return_ty: _, body } => {
                let sig = match env.functions.get(name) {
                    Some(s) => s.clone(),
                    None => continue, // had a signature error above
                };

                env.push_scope();
                for p in &sig.params {
                    env.bind(p.name.clone(), p.ty.clone(), None);
                }

                let mut body_errors: Vec<SemaError> = Vec::new();
                let mut ctx = BodyCtx {
                    env: &mut env,
                    errors: &mut body_errors,
                    return_ty: sig.return_ty.clone(),
                    in_kernel: false,
                };
                let typed_body = check_body(body, &mut ctx);
                env.pop_scope();

                errors.extend(body_errors);

                let typed_params = sig.params.iter().zip(params.iter()).map(|(s, _)| {
                    TypedParam { name: s.name.clone(), ty: s.ty.clone() }
                }).collect();

                typed_fns.push(TypedFn {
                    name: name.clone(),
                    params: typed_params,
                    return_ty: sig.return_ty.clone(),
                    body: typed_body,
                    span: item.span,
                });
            }
            ItemKind::Kernel { name, params, return_ty: _, body } => {
                let sig = match env.kernels.get(name) {
                    Some(s) => s.clone(),
                    None => continue,
                };

                env.push_scope();
                for p in &sig.params {
                    env.bind(p.name.clone(), p.ty.clone(), Some(Placement::Gpu));
                }

                let mut body_errors: Vec<SemaError> = Vec::new();
                let mut ctx = BodyCtx {
                    env: &mut env,
                    errors: &mut body_errors,
                    return_ty: sig.return_ty.clone(),
                    in_kernel: true,
                };
                let typed_body = check_body(body, &mut ctx);
                env.pop_scope();

                errors.extend(body_errors);

                let typed_kparams = sig.params.iter().zip(params.iter()).map(|(s, p)| {
                    TypedKernelParam { inout: p.inout, name: s.name.clone(), ty: s.ty.clone() }
                }).collect();

                typed_kernels.push(TypedKernel {
                    name: name.clone(),
                    params: typed_kparams,
                    return_ty: sig.return_ty.clone(),
                    body: typed_body,
                    span: item.span,
                });
            }
            _ => {}
        }
    }

    if errors.is_empty() {
        Ok(TypedProgram { fns: typed_fns, kernels: typed_kernels })
    } else {
        Err(errors)
    }
}

// ── Body checking ─────────────────────────────────────────────────────────────

fn check_body(
    stmts: &[malus_syntax::ast::Stmt],
    ctx: &mut BodyCtx<'_>,
) -> Vec<TypedStmt> {
    let mut typed: Vec<TypedStmt> = Vec::new();
    for stmt in stmts {
        match &stmt.kind {
            malus_syntax::ast::StmtKind::Let { name, expr } => {
                match check_expr(expr, None, ctx) {
                    Some(texpr) => {
                        let ty = texpr.ty.clone();
                        let placement = texpr.placement;
                        typed.push(TypedStmt::Let { name: name.clone(), expr: texpr });
                        ctx.env.bind(name.clone(), ty, placement);
                    }
                    None => return typed, // bail — can't reliably continue
                }
            }
            malus_syntax::ast::StmtKind::Return { expr } => {
                match check_expr(expr, Some(&ctx.return_ty.clone()), ctx) {
                    Some(texpr) => {
                        if texpr.ty != ctx.return_ty {
                            ctx.errors.push(SemaError::ReturnTypeMismatch {
                                expected: ctx.return_ty.clone(),
                                found: texpr.ty.clone(),
                                span: expr.span,
                            });
                        }
                        typed.push(TypedStmt::Return { expr: texpr });
                    }
                    None => return typed,
                }
            }
            malus_syntax::ast::StmtKind::Expr(expr) => {
                match check_expr(expr, None, ctx) {
                    Some(texpr) => typed.push(TypedStmt::Expr(texpr)),
                    None => return typed,
                }
            }
        }
    }
    typed
}

// ── Expression type synthesis ─────────────────────────────────────────────────

fn check_expr(
    expr: &malus_syntax::ast::Expr,
    expected: Option<&ResolvedTy>,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    match &expr.kind {
        ExprKind::Lit(lit) => check_lit(lit, expected, expr.span, ctx),
        ExprKind::Ident(name) => check_ident(name, expr.span, ctx),
        ExprKind::BinOp { op, lhs, rhs } => check_binop(op, lhs, rhs, expr.span, ctx),
        ExprKind::Unary { op, operand } => check_unary(op, operand, expr.span, ctx),
        ExprKind::Call { callee, args } => check_call(callee, args, expr.span, ctx),
        ExprKind::TensorLiteral { placement, dtype, elements } =>
            check_tensor_literal(placement, dtype, elements, expr.span, ctx),
        ExprKind::Index { base, indices } => check_index(base, indices, expr.span, ctx),
        ExprKind::FieldAccess { base, field } => check_field_access(base, field, expr.span, ctx),
    }
}

fn check_lit(
    lit: &Lit,
    expected: Option<&ResolvedTy>,
    span: Span,
    _ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let ty = match lit {
        Lit::Int(_) => {
            // Coerce to float if expected type is a float scalar — lossless widening.
            match expected {
                Some(ResolvedTy::Scalar(s)) if is_float_scalar(s) => ResolvedTy::Scalar(s.clone()),
                _ => ResolvedTy::Scalar(ScalarTy::I64),
            }
        }
        Lit::Float(_) => ResolvedTy::Scalar(ScalarTy::F32),
        Lit::Bool(_) => ResolvedTy::Bool,
        Lit::Str(_) => ResolvedTy::Unit, // string literals are only valid in print calls
    };
    Some(typed_expr(TypedExprKind::Lit(lit.clone()), ty, None, span))
}

fn check_ident(name: &str, span: Span, ctx: &mut BodyCtx<'_>) -> Option<TypedExpr> {
    match ctx.env.lookup_binding(name) {
        Some((ty, placement)) => {
            let ty = ty.clone();
            let placement = *placement;
            Some(typed_expr(TypedExprKind::Ident(name.to_string()), ty, placement, span))
        }
        None => {
            ctx.errors.push(SemaError::UnknownIdent { name: name.to_string(), span });
            None
        }
    }
}

fn check_binop(
    op: &BinOp,
    lhs: &malus_syntax::ast::Expr,
    rhs: &malus_syntax::ast::Expr,
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let tlhs = check_expr(lhs, None, ctx)?;
    let trhs = check_expr(rhs, None, ctx)?;

    // Placement check for tensor operands.
    if tlhs.ty.is_tensor() && trhs.ty.is_tensor() {
        match (tlhs.placement, trhs.placement) {
            (Some(Placement::Cpu), Some(Placement::Gpu)) |
            (Some(Placement::Gpu), Some(Placement::Cpu)) => {
                ctx.errors.push(SemaError::PlacementMismatch {
                    lhs: placement_name(tlhs.placement),
                    rhs: placement_name(trhs.placement),
                    span,
                });
                return None;
            }
            _ => {}
        }
        // Dtype check for tensor operands.
        let ldtype = tlhs.ty.tensor_dtype().unwrap();
        let rdtype = trhs.ty.tensor_dtype().unwrap();
        if ldtype != rdtype {
            ctx.errors.push(SemaError::DtypeMismatch {
                lhs: scalar_ty_name(ldtype).to_string(),
                rhs: scalar_ty_name(rdtype).to_string(),
                span,
            });
            return None;
        }
    } else if tlhs.ty != trhs.ty {
        ctx.errors.push(SemaError::TypeMismatch {
            expected: tlhs.ty.clone(),
            found: trhs.ty.clone(),
            span,
        });
        return None;
    }

    let result_ty = match op {
        BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => {
            ResolvedTy::Bool
        }
        BinOp::And | BinOp::Or => ResolvedTy::Bool,
        _ => tlhs.ty.clone(),
    };

    let placement = tlhs.placement;
    Some(typed_expr(
        TypedExprKind::BinOp {
            op: op.clone(),
            lhs: Box::new(tlhs),
            rhs: Box::new(trhs),
        },
        result_ty,
        placement,
        span,
    ))
}

fn check_unary(
    op: &UnaryOp,
    operand: &malus_syntax::ast::Expr,
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let t = check_expr(operand, None, ctx)?;
    let ty = t.ty.clone();
    let placement = t.placement;
    Some(typed_expr(
        TypedExprKind::Unary { op: op.clone(), operand: Box::new(t) },
        ty,
        placement,
        span,
    ))
}

fn check_call(
    callee_expr: &malus_syntax::ast::Expr,
    args: &[malus_syntax::ast::Expr],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    // Resolve the callee name — either bare `add` or qualified `ops.add`.
    // Returns an owned enum so we can release the borrow on ctx.env before
    // calling check_expr (which needs &mut ctx).
    let (callee_name, resolved) = resolve_callee_name(callee_expr, ctx)?;

    match resolved {
        ResolvedCallee::Kernel(sig) => {
            if ctx.in_kernel {
                ctx.errors.push(SemaError::KernelCalledFromKernel {
                    name: callee_name.clone(),
                    span,
                });
                return None;
            }
            if args.len() != sig.params.len() {
                ctx.errors.push(SemaError::ArgCountMismatch {
                    callee: callee_name.clone(),
                    expected: sig.params.len(),
                    found: args.len(),
                    span,
                });
                return None;
            }
            let mut typed_args: Vec<TypedExpr> = Vec::new();
            let mut in_flight: Vec<String> = Vec::new();
            for (arg, param) in args.iter().zip(sig.params.iter()) {
                let ta = check_expr(arg, Some(&param.ty), ctx)?;
                if ta.ty != param.ty {
                    ctx.errors.push(SemaError::TypeMismatch {
                        expected: param.ty.clone(),
                        found: ta.ty.clone(),
                        span: arg.span,
                    });
                    return None;
                }
                if ta.ty.is_tensor() {
                    if let TypedExprKind::Ident(ref name) = ta.kind {
                        in_flight.push(name.clone());
                    }
                }
                typed_args.push(ta);
            }
            let return_ty = sig.return_ty.clone();
            Some(typed_expr(
                TypedExprKind::KernelCall { callee: callee_name, args: typed_args, in_flight },
                return_ty,
                Some(Placement::Gpu),
                span,
            ))
        }
        ResolvedCallee::Fn(sig) => {
            if args.len() != sig.params.len() {
                ctx.errors.push(SemaError::ArgCountMismatch {
                    callee: callee_name.clone(),
                    expected: sig.params.len(),
                    found: args.len(),
                    span,
                });
                return None;
            }
            let mut typed_args: Vec<TypedExpr> = Vec::new();
            for (arg, param) in args.iter().zip(sig.params.iter()) {
                let ta = check_expr(arg, Some(&param.ty), ctx)?;
                if ta.ty != param.ty {
                    ctx.errors.push(SemaError::TypeMismatch {
                        expected: param.ty.clone(),
                        found: ta.ty.clone(),
                        span: arg.span,
                    });
                    return None;
                }
                typed_args.push(ta);
            }
            let return_ty = sig.return_ty.clone();
            let placement = if return_ty.is_tensor() { Some(Placement::Gpu) } else { None };
            Some(typed_expr(
                TypedExprKind::Call { callee: callee_name, args: typed_args },
                return_ty,
                placement,
                span,
            ))
        }
        ResolvedCallee::Builtin(sig) => {
            let typed_args: Vec<TypedExpr> = match &sig.kind {
                BuiltinKind::Variadic => {
                    let mut out = Vec::new();
                    for arg in args {
                        out.push(check_expr(arg, None, ctx)?);
                    }
                    out
                }
                BuiltinKind::ShapeArgs => {
                    let mut out = Vec::new();
                    for arg in args {
                        out.push(check_expr(arg, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?);
                    }
                    out
                }
                BuiltinKind::Fixed(params) => {
                    if args.len() != params.len() {
                        ctx.errors.push(SemaError::ArgCountMismatch {
                            callee: callee_name.clone(),
                            expected: params.len(),
                            found: args.len(),
                            span,
                        });
                        return None;
                    }
                    let mut out = Vec::new();
                    for (arg, param_ty) in args.iter().zip(params.iter()) {
                        out.push(check_expr(arg, Some(param_ty), ctx)?);
                    }
                    out
                }
            };
            let placement = sig.return_placement;
            let return_ty = sig.return_ty.clone();
            Some(typed_expr(
                TypedExprKind::Call { callee: callee_name, args: typed_args },
                return_ty,
                placement,
                span,
            ))
        }
    }
}

fn check_tensor_literal(
    placement: &Placement,
    dtype: &ScalarTy,
    elements: &[malus_syntax::ast::Expr],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let elem_ty = ResolvedTy::Scalar(dtype.clone());
    let mut typed_elements: Vec<TypedExpr> = Vec::new();

    for elem in elements {
        let te = check_expr(elem, Some(&elem_ty), ctx)?;
        // Check for lossy coercion: float literal into integer tensor.
        if let TypedExprKind::Lit(Lit::Float(_)) = &te.kind {
            if !is_float_scalar(dtype) {
                ctx.errors.push(SemaError::LossyCoercion {
                    from: "float".to_string(),
                    to: scalar_ty_name(dtype).to_string(),
                    span: elem.span,
                });
                return None;
            }
        }
        // Allow int literal into float tensor (lossless widening).
        typed_elements.push(te);
    }

    Some(typed_expr(
        TypedExprKind::TensorLiteral {
            placement: placement.clone(),
            dtype: dtype.clone(),
            elements: typed_elements,
        },
        ResolvedTy::Tensor { dtype: dtype.clone() },
        Some(placement.clone()),
        span,
    ))
}

fn check_index(
    base: &malus_syntax::ast::Expr,
    indices: &[malus_syntax::ast::Expr],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let tbase = check_expr(base, None, ctx)?;
    let ty = tbase.ty.clone();
    let placement = tbase.placement;
    let mut typed_indices: Vec<TypedExpr> = Vec::new();
    for idx in indices {
        typed_indices.push(check_expr(idx, None, ctx)?);
    }
    Some(typed_expr(
        TypedExprKind::Index { base: Box::new(tbase), indices: typed_indices },
        ty,
        placement,
        span,
    ))
}

fn check_field_access(
    base: &malus_syntax::ast::Expr,
    field: &str,
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    // Field access on a module alias: `ops.add` → callee resolution.
    // In the parser, `ops.add(a, b)` is parsed as Call { callee: FieldAccess(Ident("ops"), "add"), ... }
    // so this path handles the base expression of such a call.
    // For now, produce a placeholder — actual resolution happens in check_call via resolve_callee_expr.
    let tbase = check_expr(base, None, ctx)?;
    let ty = tbase.ty.clone();
    let placement = tbase.placement;
    Some(typed_expr(
        TypedExprKind::FieldAccess { base: Box::new(tbase), field: field.to_string() },
        ty,
        placement,
        span,
    ))
}

// ── Callee resolution from expression ────────────────────────────────────────

/// Owned callee — cloned eagerly so we release the borrow on env before
/// mutably borrowing ctx again for argument checking.
enum ResolvedCallee {
    Fn(FnSig),
    Kernel(KernelSig),
    Builtin(crate::builtins::BuiltinSig),
}

fn resolve_callee_name(
    callee_expr: &malus_syntax::ast::Expr,
    ctx: &mut BodyCtx<'_>,
) -> Option<(String, ResolvedCallee)> {
    match &callee_expr.kind {
        ExprKind::Ident(name) => {
            match ctx.env.resolve_callee(name) {
                Some(Callee::Fn(sig)) => Some((name.clone(), ResolvedCallee::Fn(sig.clone()))),
                Some(Callee::Kernel(sig)) => Some((name.clone(), ResolvedCallee::Kernel(sig.clone()))),
                Some(Callee::Builtin(sig)) => Some((name.clone(), ResolvedCallee::Builtin(sig.clone()))),
                None => {
                    ctx.errors.push(SemaError::NotAFunction { name: name.clone(), span: callee_expr.span });
                    None
                }
            }
        }
        ExprKind::FieldAccess { base, field } => {
            if let ExprKind::Ident(module) = &base.kind {
                match ctx.env.resolve_qualified(module, field) {
                    Some(Callee::Fn(sig)) => Some((field.clone(), ResolvedCallee::Fn(sig.clone()))),
                    Some(Callee::Kernel(sig)) => Some((field.clone(), ResolvedCallee::Kernel(sig.clone()))),
                    Some(Callee::Builtin(sig)) => Some((field.clone(), ResolvedCallee::Builtin(sig.clone()))),
                    None => {
                        ctx.errors.push(SemaError::UnknownIdent {
                            name: format!("{}.{}", module, field),
                            span: callee_expr.span,
                        });
                        None
                    }
                }
            } else {
                ctx.errors.push(SemaError::NotAFunction { name: "<expr>".to_string(), span: callee_expr.span });
                None
            }
        }
        _ => {
            ctx.errors.push(SemaError::NotAFunction { name: "<expr>".to_string(), span: callee_expr.span });
            None
        }
    }
}

// ── Type resolution helpers ───────────────────────────────────────────────────

pub fn resolve_ty(ty: &Ty, span: Span, errors: &mut Vec<SemaError>) -> Option<ResolvedTy> {
    match ty {
        Ty::Tensor { dtype } => Some(ResolvedTy::Tensor { dtype: dtype.clone() }),
        Ty::Scalar(s) => Some(ResolvedTy::Scalar(s.clone())),
        Ty::Bool => Some(ResolvedTy::Bool),
        Ty::Tuple(ts) => {
            let mut resolved = Vec::new();
            for t in ts {
                resolved.push(resolve_ty(t, span, errors)?);
            }
            Some(ResolvedTy::Tuple(resolved))
        }
        Ty::Named(name) if name == "None" => Some(ResolvedTy::Unit),
        Ty::Named(name) => {
            errors.push(SemaError::UnknownType { name: name.clone(), span });
            None
        }
    }
}

fn resolve_params(
    params: &[malus_syntax::ast::Param],
    errors: &mut Vec<SemaError>,
) -> Option<Vec<ParamSig>> {
    let mut out = Vec::new();
    for p in params {
        let ty = resolve_ty(&p.ty, p.span, errors)?;
        out.push(ParamSig { name: p.name.clone(), ty });
    }
    Some(out)
}

fn resolve_kernel_params(
    params: &[malus_syntax::ast::KernelParam],
    errors: &mut Vec<SemaError>,
) -> Option<Vec<KernelParamSig>> {
    let mut out = Vec::new();
    for p in params {
        let ty = resolve_ty(&p.ty, p.span, errors)?;
        out.push(KernelParamSig { inout: p.inout, name: p.name.clone(), ty });
    }
    Some(out)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn typed_expr(
    kind: TypedExprKind,
    ty: ResolvedTy,
    placement: Option<Placement>,
    span: Span,
) -> TypedExpr {
    TypedExpr { kind, ty, placement, span }
}

fn placement_name(p: Option<Placement>) -> String {
    match p {
        Some(Placement::Cpu) => "cpu".to_string(),
        Some(Placement::Gpu) => "gpu".to_string(),
        None => "unknown".to_string(),
    }
}
