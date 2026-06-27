use std::collections::{HashMap, HashSet};
use malus_syntax::ast::{
    BinOp, CallArg, ExprKind, ItemKind, Lit, Placement, Program, ScalarTy, StmtKind,
    Ty, UnaryOp,
};
use malus_syntax::Span;
use crate::builtins::{BuiltinKind, register_builtins};
use crate::env::{Callee, Env, EnumDef, FnSig, KernelSig, KernelParamSig, ParamSig, StructDef,
    VariantSig};
use crate::error::SemaError;
use crate::ty::{is_float_scalar, scalar_ty_name, ResolvedTy};
use crate::typed_ir::{
    TypedExpr, TypedExprKind, TypedFn, TypedKernel, TypedKernelParam, TypedMatchArm, TypedParam,
    TypedProgram, TypedStmt,
};

// ── Nominal maps (thread through resolve_ty without borrow conflicts) ─────────

pub(crate) struct NominalMaps<'a> {
    structs: &'a HashMap<String, StructDef>,
    enums: &'a HashMap<String, EnumDef>,
}

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

    // ── Pass 1a: collect struct/enum names ───────────────────────────────────
    //
    // We must do this before resolving fn/kernel signatures so that field and
    // return types referencing user types can resolve.  We also register names
    // before resolving field types so mutual/forward references work.

    let mut local_structs: HashMap<String, StructDef> = HashMap::new();
    let mut local_enums: HashMap<String, EnumDef> = HashMap::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Struct { name, .. } => {
                if local_structs.contains_key(name.as_str()) || local_enums.contains_key(name.as_str()) {
                    let first = local_structs.get(name.as_str())
                        .map(|d| d.defined_at)
                        .or_else(|| local_enums.get(name.as_str()).map(|d| d.defined_at))
                        .unwrap_or(item.span);
                    errors.push(SemaError::DuplicateTypeDefinition { name: name.clone(), first, second: item.span });
                    continue;
                }
                local_structs.insert(name.clone(), StructDef { fields: vec![], defined_at: item.span });
            }
            ItemKind::Enum { name, .. } => {
                if local_enums.contains_key(name.as_str()) || local_structs.contains_key(name.as_str()) {
                    let first = local_enums.get(name.as_str())
                        .map(|d| d.defined_at)
                        .or_else(|| local_structs.get(name.as_str()).map(|d| d.defined_at))
                        .unwrap_or(item.span);
                    errors.push(SemaError::DuplicateTypeDefinition { name: name.clone(), first, second: item.span });
                    continue;
                }
                local_enums.insert(name.clone(), EnumDef { variants: vec![], defined_at: item.span });
            }
            _ => {}
        }
    }

    // ── Pass 1b: resolve struct/enum field types ──────────────────────────────

    for item in &program.items {
        let nominals = NominalMaps { structs: &local_structs, enums: &local_enums };
        match &item.kind {
            ItemKind::Struct { name, fields } => {
                if !local_structs.contains_key(name.as_str()) { continue; }
                let mut resolved_fields = Vec::new();
                let mut ok = true;
                for f in fields {
                    match resolve_ty(&f.ty, f.span, &nominals, &mut errors) {
                        Some(ty) => resolved_fields.push((f.name.clone(), ty)),
                        None => { ok = false; }
                    }
                }
                if ok {
                    if let Some(def) = local_structs.get_mut(name.as_str()) {
                        def.fields = resolved_fields;
                    }
                }
            }
            ItemKind::Enum { name, variants } => {
                if !local_enums.contains_key(name.as_str()) { continue; }
                let mut resolved_variants = Vec::new();
                let mut ok = true;
                for v in variants {
                    let mut vfields = Vec::new();
                    for f in &v.fields {
                        match resolve_ty(&f.ty, f.span, &nominals, &mut errors) {
                            Some(ty) => vfields.push((f.name.clone(), ty)),
                            None => { ok = false; }
                        }
                    }
                    resolved_variants.push(VariantSig { name: v.name.clone(), fields: vfields });
                }
                if ok {
                    if let Some(def) = local_enums.get_mut(name.as_str()) {
                        def.variants = resolved_variants;
                    }
                }
            }
            _ => {}
        }
    }

    // ── Pass 1c: collect fn/kernel signatures ─────────────────────────────────

    let mut has_main = false;

    for item in &program.items {
        let nominals = NominalMaps { structs: &local_structs, enums: &local_enums };
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
                let resolved_params = match resolve_params(params, &nominals, &mut errors) {
                    Some(p) => p,
                    None => continue,
                };
                let resolved_return = match return_ty {
                    Some(ty) => match resolve_ty(ty, item.span, &nominals, &mut errors) {
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
                let resolved_params = match resolve_kernel_params(params, &nominals, &mut errors) {
                    Some(p) => p,
                    None => continue,
                };
                let resolved_return = match resolve_ty(return_ty, item.span, &nominals, &mut errors) {
                    Some(t) => t,
                    None => continue,
                };
                env.kernels.insert(name.clone(), KernelSig {
                    params: resolved_params,
                    return_ty: resolved_return,
                    defined_at: item.span,
                });
            }
            // Struct/Enum: already handled above.
            ItemKind::Struct { .. } | ItemKind::Enum { .. } => {}
            // Imports are already resolved by the loader — ignore them.
            ItemKind::Import { .. } | ItemKind::FromImport { .. } => {}
        }
    }

    // Move local nominal maps into env for pass-2 body checking.
    env.structs = local_structs;
    env.enums = local_enums;

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
                    // Element-space: inside the kernel body, tensor params are seen as
                    // their element scalar type. The external signature stays Tensor.
                    let elem_ty = match &p.ty {
                        ResolvedTy::Tensor { dtype } => ResolvedTy::Scalar(dtype.clone()),
                        other => other.clone(),
                    };
                    env.bind(p.name.clone(), elem_ty, Some(Placement::Gpu));
                }

                // Element-space: kernel body returns the element scalar type; the
                // external return type (Tensor<dtype>) is used for callers.
                let kernel_return_ty = match &sig.return_ty {
                    ResolvedTy::Tensor { dtype } => ResolvedTy::Scalar(dtype.clone()),
                    other => other.clone(),
                };

                let mut body_errors: Vec<SemaError> = Vec::new();
                let mut ctx = BodyCtx {
                    env: &mut env,
                    errors: &mut body_errors,
                    return_ty: kernel_return_ty,
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
            malus_syntax::ast::StmtKind::LetMut { name, expr } => {
                match check_expr(expr, None, ctx) {
                    Some(texpr) => {
                        let ty = texpr.ty.clone();
                        let placement = texpr.placement;
                        typed.push(TypedStmt::Let { name: name.clone(), expr: texpr });
                        ctx.env.bind_mutable(name.clone(), ty, placement);
                    }
                    None => return typed,
                }
            }
            malus_syntax::ast::StmtKind::Assign { target, expr } => {
                let target_ty = match ctx.env.lookup_binding(target) {
                    Some((ty, _)) => ty.clone(),
                    None => {
                        ctx.errors.push(SemaError::UnknownIdent {
                            name: target.clone(),
                            span: stmt.span,
                        });
                        return typed;
                    }
                };
                if !ctx.env.is_mutable(target) {
                    ctx.errors.push(SemaError::AssignToImmutable {
                        name: target.clone(),
                        span: stmt.span,
                    });
                    return typed;
                }
                match check_expr(expr, Some(&target_ty), ctx) {
                    Some(texpr) => {
                        if texpr.ty != target_ty {
                            ctx.errors.push(SemaError::TypeMismatch {
                                expected: target_ty,
                                found: texpr.ty.clone(),
                                span: expr.span,
                            });
                            return typed;
                        }
                        typed.push(TypedStmt::Assign { name: target.clone(), expr: texpr });
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
            StmtKind::If { condition, then_body, else_body } => {
                // Condition must be Bool.
                let tcond = match check_expr(condition, Some(&ResolvedTy::Bool), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                if tcond.ty != ResolvedTy::Bool {
                    ctx.errors.push(SemaError::TypeMismatch {
                        expected: ResolvedTy::Bool,
                        found: tcond.ty.clone(),
                        span: condition.span,
                    });
                    return typed;
                }
                // Each branch is checked in its own scope so bindings don't escape.
                ctx.env.push_scope();
                let tthen = check_body(then_body, ctx);
                ctx.env.pop_scope();
                let telse = if let Some(eb) = else_body {
                    ctx.env.push_scope();
                    let t = check_body(eb, ctx);
                    ctx.env.pop_scope();
                    Some(t)
                } else {
                    None
                };
                typed.push(TypedStmt::If { condition: tcond, then_body: tthen, else_body: telse });
            }
            StmtKind::For { var, start, end, body } => {
                // Loop bounds must be I64 (range() desugars to int literals or exprs).
                let i64_ty = ResolvedTy::Scalar(ScalarTy::I64);
                let tstart = match check_expr(start, Some(&i64_ty), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                let tend = match check_expr(end, Some(&i64_ty), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                // Loop variable is I64, visible only inside the body.
                ctx.env.push_scope();
                ctx.env.bind(var.clone(), i64_ty, None);
                let tbody = check_body(body, ctx);
                ctx.env.pop_scope();
                typed.push(TypedStmt::For { var: var.clone(), start: tstart, end: tend, body: tbody });
            }
            StmtKind::ForIn { var, iter, body } => {
                // `iter` must resolve to Array<T, N>; `var` is bound to T inside body.
                let titer = match check_expr(iter, None, ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                let (elem_ty, _len) = match &titer.ty {
                    ResolvedTy::Array { elem, len } => (*elem.clone(), *len),
                    other => {
                        ctx.errors.push(SemaError::TypeMismatch {
                            expected: ResolvedTy::Array {
                                elem: Box::new(ResolvedTy::Unit),
                                len: 0,
                            },
                            found: other.clone(),
                            span: iter.span,
                        });
                        return typed;
                    }
                };
                ctx.env.push_scope();
                ctx.env.bind(var.clone(), elem_ty, None);
                let tbody = check_body(body, ctx);
                ctx.env.pop_scope();
                typed.push(TypedStmt::ForIn { var: var.clone(), iter: titer, body: tbody });
            }
            StmtKind::While { condition, body } => {
                // Condition must be Bool.
                let tcond = match check_expr(condition, Some(&ResolvedTy::Bool), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                if tcond.ty != ResolvedTy::Bool {
                    ctx.errors.push(SemaError::TypeMismatch {
                        expected: ResolvedTy::Bool,
                        found: tcond.ty.clone(),
                        span: condition.span,
                    });
                    return typed;
                }
                ctx.env.push_scope();
                let tbody = check_body(body, ctx);
                ctx.env.pop_scope();
                typed.push(TypedStmt::While { condition: tcond, body: tbody });
            }
            StmtKind::Match { scrutinee, arms } => {
                let tscrutinee = match check_expr(scrutinee, None, ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                // Scrutinee must be an enum type.
                let (enum_name, variants) = match &tscrutinee.ty {
                    ResolvedTy::Enum { name, variants } => (name.clone(), variants.clone()),
                    other => {
                        ctx.errors.push(SemaError::MatchScrutineeNotEnum {
                            found: other.to_string(),
                            span: scrutinee.span,
                        });
                        return typed;
                    }
                };
                // Exhaustiveness and uniqueness check.
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                for arm in arms {
                    if arm.variant == "_" {
                        ctx.errors.push(SemaError::MatchWildcard { span: arm.span });
                        return typed;
                    }
                    if !seen.insert(arm.variant.clone()) {
                        ctx.errors.push(SemaError::DuplicateMatchArm { variant: arm.variant.clone(), span: arm.span });
                        return typed;
                    }
                    if !variants.iter().any(|(vn, _)| vn == &arm.variant) {
                        ctx.errors.push(SemaError::UnknownVariant {
                            enum_name: enum_name.clone(),
                            variant: arm.variant.clone(),
                            span: arm.span,
                        });
                        return typed;
                    }
                }
                let missing: Vec<String> = variants.iter()
                    .filter_map(|(vn, _)| if seen.contains(vn.as_str()) { None } else { Some(vn.clone()) })
                    .collect();
                if !missing.is_empty() {
                    ctx.errors.push(SemaError::NonExhaustiveMatch {
                        enum_name: enum_name.clone(),
                        missing,
                        span: scrutinee.span,
                    });
                    return typed;
                }
                // Type-check each arm.
                let mut typed_arms: Vec<TypedMatchArm> = Vec::new();
                for arm in arms {
                    let (variant_index, vfields) = variants.iter()
                        .enumerate()
                        .find(|(_, (vn, _))| vn == &arm.variant)
                        .map(|(i, (_, vf))| (i as u32, vf.clone()))
                        .unwrap();
                    // Arity check on bindings.
                    if arm.bindings.len() != vfields.len() {
                        ctx.errors.push(SemaError::MatchArmArityMismatch {
                            variant: arm.variant.clone(),
                            expected: vfields.len(),
                            found: arm.bindings.len(),
                            span: arm.span,
                        });
                        return typed;
                    }
                    ctx.env.push_scope();
                    let mut bindings_typed: Vec<(String, ResolvedTy)> = Vec::new();
                    for (bname, (_, fty)) in arm.bindings.iter().zip(vfields.iter()) {
                        let fpl = if fty.is_tensor() { Some(Placement::Gpu) } else { None };
                        ctx.env.bind(bname.clone(), fty.clone(), fpl);
                        bindings_typed.push((bname.clone(), fty.clone()));
                    }
                    let arm_body = check_body(&arm.body, ctx);
                    ctx.env.pop_scope();
                    typed_arms.push(TypedMatchArm {
                        variant: arm.variant.clone(),
                        variant_index,
                        bindings: bindings_typed,
                        body: arm_body,
                    });
                }
                typed.push(TypedStmt::Match { scrutinee: tscrutinee, arms: typed_arms });
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
        ExprKind::TensorLiteral { placement, dtype, elements, shape } =>
            check_tensor_literal(placement, dtype, elements, shape, expr.span, ctx),
        ExprKind::ArrayLiteral { elements } =>
            check_array_literal(elements, expr.span, ctx),
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
        // Allow scalar broadcast in fn bodies: Tensor<dtype> op Scalar(same dtype)
        // for arithmetic ops. Reject comparisons and matmul with mixed types.
        let is_broadcast = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
            && match (&tlhs.ty, &trhs.ty) {
                (ResolvedTy::Tensor { dtype: td }, ResolvedTy::Scalar(sd)) => td == sd,
                (ResolvedTy::Scalar(sd), ResolvedTy::Tensor { dtype: td }) => sd == td,
                _ => false,
            };
        if !is_broadcast {
            ctx.errors.push(SemaError::TypeMismatch {
                expected: tlhs.ty.clone(),
                found: trhs.ty.clone(),
                span,
            });
            return None;
        }
    }

    let result_ty = match op {
        BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => {
            if ctx.in_kernel {
                // Element-space: comparison yields the operand's scalar dtype (the mask).
                // MSL bool-to-float implicit; no Bool type inside kernel bodies.
                tlhs.ty.clone()
            } else {
                ResolvedTy::Bool
            }
        }
        BinOp::And | BinOp::Or => ResolvedTy::Bool,
        _ => {
            // For scalar broadcast, result is the tensor type regardless of operand order.
            match (&tlhs.ty, &trhs.ty) {
                (ResolvedTy::Scalar(_), ResolvedTy::Tensor { .. }) => trhs.ty.clone(),
                _ => tlhs.ty.clone(),
            }
        }
    };

    // Placement: prefer the tensor operand's placement for scalar-broadcast ops.
    let placement = match (&tlhs.placement, &trhs.placement) {
        (None, Some(p)) => Some(*p),
        (Some(p), _) => Some(*p),
        _ => None,
    };
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
    args: &[CallArg],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    // ── Struct constructor: Layer(weights=w, bias=b) ──────────────────────────
    if let ExprKind::Ident(type_name) = &callee_expr.kind {
        if let Some(sdef) = ctx.env.structs.get(type_name.as_str()).cloned() {
            let struct_name = type_name.clone();
            // Build name→arg map; reject positional args in struct constructors.
            let mut named: std::collections::HashMap<String, &malus_syntax::ast::Expr> = std::collections::HashMap::new();
            for arg in args {
                match &arg.name {
                    Some(n) => { named.insert(n.clone(), &arg.value); }
                    None => {
                        ctx.errors.push(SemaError::UnknownConstructorField {
                            struct_name: struct_name.clone(),
                            field: "<positional>".to_string(),
                            span,
                        });
                        return None;
                    }
                }
            }
            // Check for unknown fields.
            for arg in args {
                if let Some(n) = &arg.name {
                    if !sdef.fields.iter().any(|(f, _)| f == n) {
                        ctx.errors.push(SemaError::UnknownConstructorField {
                            struct_name: struct_name.clone(),
                            field: n.clone(),
                            span: arg.value.span,
                        });
                        return None;
                    }
                }
            }
            // Resolve fields in decl order.
            let mut fields_out: Vec<TypedExpr> = Vec::new();
            for (fname, fty) in &sdef.fields {
                match named.get(fname.as_str()) {
                    Some(arg_expr) => {
                        let ta = check_expr(arg_expr, Some(fty), ctx)?;
                        if ta.ty != *fty {
                            ctx.errors.push(SemaError::TypeMismatch {
                                expected: fty.clone(),
                                found: ta.ty.clone(),
                                span: arg_expr.span,
                            });
                            return None;
                        }
                        fields_out.push(ta);
                    }
                    None => {
                        ctx.errors.push(SemaError::MissingField {
                            struct_name: struct_name.clone(),
                            field: fname.clone(),
                            span,
                        });
                        return None;
                    }
                }
            }
            let ty = ResolvedTy::Struct { name: struct_name.clone(), fields: sdef.fields.clone() };
            return Some(typed_expr(
                TypedExprKind::StructInit { name: struct_name, fields: fields_out },
                ty,
                None,
                span,
            ));
        }
    }

    // ── Enum variant constructor: Activation.Relu(...) ───────────────────────
    if let ExprKind::FieldAccess { base, field: variant_name } = &callee_expr.kind {
        if let ExprKind::Ident(enum_name) = &base.kind {
            if let Some(edef) = ctx.env.enums.get(enum_name.as_str()).cloned() {
                let en = enum_name.clone();
                let vn = variant_name.clone();
                let variant_index = match edef.variants.iter().position(|v| v.name == vn) {
                    Some(i) => i as u32,
                    None => {
                        ctx.errors.push(SemaError::UnknownVariant {
                            enum_name: en.clone(),
                            variant: vn.clone(),
                            span,
                        });
                        return None;
                    }
                };
                let vsig = edef.variants[variant_index as usize].clone();
                let max_payload_slots = edef.variants.iter().map(|v| v.fields.len()).max().unwrap_or(0);
                let mut payload_out: Vec<TypedExpr> = Vec::new();
                if !args.is_empty() {
                    if args.iter().any(|a| a.name.is_some()) {
                        let mut named_map: std::collections::HashMap<String, &malus_syntax::ast::Expr> = std::collections::HashMap::new();
                        for arg in args {
                            if let Some(n) = &arg.name {
                                named_map.insert(n.clone(), &arg.value);
                            }
                        }
                        for (fname, fty) in &vsig.fields {
                            match named_map.get(fname.as_str()) {
                                Some(aexpr) => { payload_out.push(check_expr(aexpr, Some(fty), ctx)?); }
                                None => {
                                    ctx.errors.push(SemaError::MissingField {
                                        struct_name: format!("{}::{}", en, vn),
                                        field: fname.clone(),
                                        span,
                                    });
                                    return None;
                                }
                            }
                        }
                    } else {
                        if args.len() != vsig.fields.len() {
                            ctx.errors.push(SemaError::MatchArmArityMismatch {
                                variant: vn.clone(),
                                expected: vsig.fields.len(),
                                found: args.len(),
                                span,
                            });
                            return None;
                        }
                        for (arg, (_, fty)) in args.iter().zip(vsig.fields.iter()) {
                            payload_out.push(check_expr(&arg.value, Some(fty), ctx)?);
                        }
                    }
                }
                let variants_ty = edef.variants.iter()
                    .map(|v| (v.name.clone(), v.fields.clone()))
                    .collect();
                let ty = ResolvedTy::Enum { name: en.clone(), variants: variants_ty };
                return Some(typed_expr(
                    TypedExprKind::EnumInit {
                        enum_name: en,
                        variant: vn,
                        variant_index,
                        payload: payload_out,
                        max_payload_slots,
                    },
                    ty,
                    None,
                    span,
                ));
            }
        }
    }

    // ── Regular function/kernel/builtin call (positional) ─────────────────────
    // Returns an owned enum so we can release the borrow on ctx.env before
    // calling check_expr (which needs &mut ctx).
    let (callee_name, resolved) = resolve_callee_name(callee_expr, ctx)?;
    // Strip CallArg wrappers — regular calls are positional.
    let positional: Vec<&malus_syntax::ast::Expr> = args.iter().map(|a| &a.value).collect();

    match resolved {
        ResolvedCallee::Kernel(sig) => {
            if ctx.in_kernel {
                ctx.errors.push(SemaError::KernelCalledFromKernel {
                    name: callee_name.clone(),
                    span,
                });
                return None;
            }
            if positional.len() != sig.params.len() {
                ctx.errors.push(SemaError::ArgCountMismatch {
                    callee: callee_name.clone(),
                    expected: sig.params.len(),
                    found: positional.len(),
                    span,
                });
                return None;
            }
            let mut typed_args: Vec<TypedExpr> = Vec::new();
            let mut in_flight: Vec<String> = Vec::new();
            for (arg, param) in positional.iter().zip(sig.params.iter()) {
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
            if positional.len() != sig.params.len() {
                ctx.errors.push(SemaError::ArgCountMismatch {
                    callee: callee_name.clone(),
                    expected: sig.params.len(),
                    found: positional.len(),
                    span,
                });
                return None;
            }
            let mut typed_args: Vec<TypedExpr> = Vec::new();
            for (arg, param) in positional.iter().zip(sig.params.iter()) {
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
            let is_print_call = callee_name == "print" || callee_name == "println";
            let typed_args: Vec<TypedExpr> = match &sig.kind {
                BuiltinKind::Variadic => {
                    let mut out = Vec::new();
                    for (i, arg) in positional.iter().enumerate() {
                        let checked = check_expr(arg, None, ctx)?;
                        // String literals are only valid as the first arg of print/println.
                        if checked.ty == ResolvedTy::Unit {
                            if let TypedExprKind::Lit(Lit::Str(_)) = &checked.kind {
                                if !is_print_call || i > 0 {
                                    ctx.errors.push(SemaError::StringLiteralOutsidePrint { span: arg.span });
                                    return None;
                                }
                            }
                        }
                        out.push(checked);
                    }
                    out
                }
                BuiltinKind::ShapeArgs => {
                    let mut out = Vec::new();
                    for arg in positional.iter() {
                        out.push(check_expr(arg, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?);
                    }
                    out
                }
                BuiltinKind::Fixed(params) => {
                    if positional.len() != params.len() {
                        ctx.errors.push(SemaError::ArgCountMismatch {
                            callee: callee_name.clone(),
                            expected: params.len(),
                            found: positional.len(),
                            span,
                        });
                        return None;
                    }
                    let mut out = Vec::new();
                    for (arg, param_ty) in positional.iter().zip(params.iter()) {
                        out.push(check_expr(arg, Some(param_ty), ctx)?);
                    }
                    out
                }
            };
            // Validate format string arg count for print/println.
            if is_print_call {
                if let Some(first) = typed_args.first() {
                    if let TypedExprKind::Lit(Lit::Str(fmt)) = &first.kind {
                        let placeholders = fmt.matches("{}").count();
                        let value_args = typed_args.len() - 1;
                        if placeholders != value_args {
                            ctx.errors.push(SemaError::FormatArgCountMismatch {
                                callee: callee_name.clone(),
                                placeholders,
                                args: value_args,
                                span,
                            });
                            return None;
                        }
                    }
                }
            }
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
    shape: &[usize],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    // Validate product(shape) == elements.len().
    let expected_count: usize = shape.iter().product();
    if expected_count != elements.len() {
        ctx.errors.push(SemaError::TensorShapeMismatch {
            expected: expected_count,
            found: elements.len(),
            span,
        });
        return None;
    }

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
            shape: shape.to_vec(),
        },
        ResolvedTy::Tensor { dtype: dtype.clone() },
        Some(placement.clone()),
        span,
    ))
}

fn check_array_literal(
    elements: &[malus_syntax::ast::Expr],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    if elements.is_empty() {
        ctx.errors.push(SemaError::TypeMismatch {
            expected: ResolvedTy::Array { elem: Box::new(ResolvedTy::Unit), len: 0 },
            found: ResolvedTy::Unit,
            span,
        });
        return None;
    }
    let first = check_expr(&elements[0], None, ctx)?;
    let elem_ty = first.ty.clone();
    let placement = first.placement;
    let mut typed: Vec<TypedExpr> = vec![first];
    for elem in &elements[1..] {
        let te = check_expr(elem, Some(&elem_ty), ctx)?;
        if te.ty != elem_ty {
            ctx.errors.push(SemaError::TypeMismatch {
                expected: elem_ty.clone(),
                found: te.ty.clone(),
                span: elem.span,
            });
            return None;
        }
        typed.push(te);
    }
    let len = typed.len();
    Some(typed_expr(
        TypedExprKind::ArrayLiteral { elements: typed },
        ResolvedTy::Array { elem: Box::new(elem_ty), len },
        placement,
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
    // Array<T, N>[i] → T
    if let ResolvedTy::Array { elem, .. } = &tbase.ty.clone() {
        let elem_ty = *elem.clone();
        let placement = tbase.placement;
        let mut typed_indices: Vec<TypedExpr> = Vec::new();
        for idx in indices {
            typed_indices.push(check_expr(idx, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?);
        }
        return Some(typed_expr(
            TypedExprKind::Index { base: Box::new(tbase), indices: typed_indices },
            elem_ty,
            placement,
            span,
        ));
    }
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
    // Bare enum variant: `Activation.Relu` (no call) → EnumInit with empty payload.
    if let ExprKind::Ident(enum_name) = &base.kind {
        if let Some(edef) = ctx.env.enums.get(enum_name.as_str()).cloned() {
            let en = enum_name.clone();
            let vn = field.to_string();
            let variant_index = match edef.variants.iter().position(|v| v.name == vn) {
                Some(i) => i as u32,
                None => {
                    ctx.errors.push(SemaError::UnknownVariant {
                        enum_name: en.clone(),
                        variant: vn.clone(),
                        span,
                    });
                    return None;
                }
            };
            let vsig = &edef.variants[variant_index as usize];
            if !vsig.fields.is_empty() {
                // Data-carrying variant used without args — treat as constructor call arity error.
                ctx.errors.push(SemaError::MatchArmArityMismatch {
                    variant: vn.clone(),
                    expected: vsig.fields.len(),
                    found: 0,
                    span,
                });
                return None;
            }
            let max_payload_slots = edef.variants.iter().map(|v| v.fields.len()).max().unwrap_or(0);
            let variants_ty = edef.variants.iter().map(|v| (v.name.clone(), v.fields.clone())).collect();
            let ty = ResolvedTy::Enum { name: en.clone(), variants: variants_ty };
            return Some(typed_expr(
                TypedExprKind::EnumInit {
                    enum_name: en,
                    variant: vn,
                    variant_index,
                    payload: vec![],
                    max_payload_slots,
                },
                ty,
                None,
                span,
            ));
        }
    }

    // Field access on a module alias: `ops.add` → callee resolution.
    // In the parser, `ops.add(a, b)` is parsed as Call { callee: FieldAccess(Ident("ops"), "add"), ... }
    // so this path handles the base expression of such a call.
    let tbase = check_expr(base, None, ctx)?;

    // .len on a tensor returns the element count as i64.
    if field == "len" && tbase.ty.is_tensor() {
        return Some(typed_expr(
            TypedExprKind::FieldAccess { base: Box::new(tbase), field: field.to_string() },
            ResolvedTy::Scalar(ScalarTy::I64),
            None,
            span,
        ));
    }

    // Struct field access: `s.field` → type of that field.
    if let ResolvedTy::Struct { fields, .. } = &tbase.ty {
        if let Some((_, fty)) = fields.iter().find(|(n, _)| n == field) {
            let field_ty = fty.clone();
            let field_placement = if field_ty.is_tensor() { Some(malus_syntax::ast::Placement::Gpu) } else { None };
            return Some(typed_expr(
                TypedExprKind::FieldAccess { base: Box::new(tbase), field: field.to_string() },
                field_ty,
                field_placement,
                span,
            ));
        }
        // Report unknown field.
        let struct_name = if let ResolvedTy::Struct { name, .. } = &tbase.ty {
            name.clone()
        } else { unreachable!() };
        ctx.errors.push(SemaError::UnknownField {
            struct_name,
            field: field.to_string(),
            span,
        });
        return None;
    }

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

pub fn resolve_ty(
    ty: &Ty,
    span: Span,
    nominals: &NominalMaps<'_>,
    errors: &mut Vec<SemaError>,
) -> Option<ResolvedTy> {
    match ty {
        Ty::Tensor { dtype } => Some(ResolvedTy::Tensor { dtype: dtype.clone() }),
        Ty::Scalar(s) => Some(ResolvedTy::Scalar(s.clone())),
        Ty::Bool => Some(ResolvedTy::Bool),
        Ty::Tuple(ts) => {
            let mut resolved = Vec::new();
            for t in ts {
                resolved.push(resolve_ty(t, span, nominals, errors)?);
            }
            Some(ResolvedTy::Tuple(resolved))
        }
        Ty::Array { elem, len } => {
            let resolved_elem = resolve_ty(elem, span, nominals, errors)?;
            Some(ResolvedTy::Array { elem: Box::new(resolved_elem), len: *len })
        }
        Ty::Named(name) if name == "None" => Some(ResolvedTy::Unit),
        Ty::Named(name) => {
            if let Some(def) = nominals.structs.get(name.as_str()) {
                return Some(ResolvedTy::Struct {
                    name: name.clone(),
                    fields: def.fields.clone(),
                });
            }
            if let Some(def) = nominals.enums.get(name.as_str()) {
                let variants = def.variants.iter()
                    .map(|v| (v.name.clone(), v.fields.clone()))
                    .collect();
                return Some(ResolvedTy::Enum { name: name.clone(), variants });
            }
            errors.push(SemaError::UnknownType { name: name.clone(), span });
            None
        }
    }
}

fn resolve_params(
    params: &[malus_syntax::ast::Param],
    nominals: &NominalMaps<'_>,
    errors: &mut Vec<SemaError>,
) -> Option<Vec<ParamSig>> {
    let mut out = Vec::new();
    for p in params {
        let ty = resolve_ty(&p.ty, p.span, nominals, errors)?;
        out.push(ParamSig { name: p.name.clone(), ty });
    }
    Some(out)
}

fn resolve_kernel_params(
    params: &[malus_syntax::ast::KernelParam],
    nominals: &NominalMaps<'_>,
    errors: &mut Vec<SemaError>,
) -> Option<Vec<KernelParamSig>> {
    let mut out = Vec::new();
    for p in params {
        let ty = resolve_ty(&p.ty, p.span, nominals, errors)?;
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
