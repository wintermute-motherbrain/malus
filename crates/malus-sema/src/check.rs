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
    TypedAssignTarget, TypedExpr, TypedExprKind, TypedFn, TypedKernel, TypedKernelParam,
    TypedMatchArm, TypedParam, TypedProgram, TypedStmt,
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
    loop_depth: usize,
    no_grad_depth: usize,
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
                        Some(ty) => {
                            if ty.is_tuple() {
                                errors.push(SemaError::TupleInStructField {
                                    struct_name: name.clone(),
                                    field: f.name.clone(),
                                    span: f.span,
                                });
                                ok = false;
                            } else {
                                resolved_fields.push((f.name.clone(), ty));
                            }
                        }
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
            ItemKind::Fn { name, params: _, return_ty: _, body } => {
                let sig = match env.functions.get(name) {
                    Some(s) => s.clone(),
                    None => continue, // had a signature error above
                };

                env.push_scope();
                for p in &sig.params {
                    if p.is_mut {
                        env.bind_mut_param(p.name.clone(), p.ty.clone(), None);
                    } else {
                        env.bind(p.name.clone(), p.ty.clone(), None);
                    }
                }

                let mut body_errors: Vec<SemaError> = Vec::new();
                let mut ctx = BodyCtx {
                    env: &mut env,
                    errors: &mut body_errors,
                    return_ty: sig.return_ty.clone(),
                    in_kernel: false,
                    loop_depth: 0,
                    no_grad_depth: 0,
                };
                let typed_body = check_body(body, &mut ctx);
                env.pop_scope();

                errors.extend(body_errors);

                let typed_params = sig.params.iter().map(|s| {
                    TypedParam { name: s.name.clone(), ty: s.ty.clone(), is_mut: s.is_mut }
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

                let implicit = is_implicit_map_kernel(body);

                env.push_scope();
                if implicit {
                    // Legacy element-space rebinding: tensor params seen as scalar element type.
                    for p in &sig.params {
                        let elem_ty = match &p.ty {
                            ResolvedTy::Tensor { dtype } => ResolvedTy::Scalar(dtype.clone()),
                            other => other.clone(),
                        };
                        env.bind(p.name.clone(), elem_ty, Some(Placement::Gpu));
                    }
                } else {
                    // Explicit kernel: params keep their resolved types (indexable Tensors,
                    // pass-through scalars).  The implicit `out` binding is a mutable Tensor
                    // of the return dtype; body writes `out[expr] = val`.
                    for p in &sig.params {
                        let pl = if matches!(&p.ty, ResolvedTy::Tensor { .. }) {
                            Some(Placement::Gpu)
                        } else {
                            None
                        };
                        env.bind(p.name.clone(), p.ty.clone(), pl);
                    }
                    env.bind_mutable("out".to_string(), sig.return_ty.clone(), Some(Placement::Gpu));
                }

                let kernel_return_ty = if implicit {
                    // Element-space return: body returns scalar element type.
                    match &sig.return_ty {
                        ResolvedTy::Tensor { dtype } => ResolvedTy::Scalar(dtype.clone()),
                        other => other.clone(),
                    }
                } else {
                    // Explicit kernels write to `out`; no explicit return expected.
                    ResolvedTy::Unit
                };

                let mut body_errors: Vec<SemaError> = Vec::new();
                let mut ctx = BodyCtx {
                    env: &mut env,
                    errors: &mut body_errors,
                    return_ty: kernel_return_ty,
                    in_kernel: true,
                    loop_depth: 0,
                    no_grad_depth: 0,
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
                    is_implicit_map: implicit,
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

// ── Kernel form classifier ────────────────────────────────────────────────────

/// Returns `true` iff the kernel body is the legacy implicit-map form: only
/// `Let`/`LetMut` bindings followed by a single final `Return` expression, with
/// no thread intrinsics, indexing, shared memory, control flow, or `out` writes.
/// Codegen-gpu uses this to choose the old `out[tid]=expr` lowering.
fn is_implicit_map_kernel(body: &[malus_syntax::ast::Stmt]) -> bool {
    use malus_syntax::ast::StmtKind;
    if body.is_empty() {
        return false;
    }
    for stmt in body {
        match &stmt.kind {
            StmtKind::Let { .. } | StmtKind::LetMut { .. } => {}
            StmtKind::Return { .. } => {}
            _ => return false,
        }
    }
    matches!(body.last().map(|s| &s.kind), Some(StmtKind::Return { .. }))
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
                if ctx.no_grad_depth > 0 {
                    ctx.errors.push(SemaError::EarlyExitInNoGrad { span: stmt.span });
                    return typed;
                }
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
            malus_syntax::ast::StmtKind::LetTuple { names, mutable, expr } => {
                let texpr = match check_expr(expr, None, ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                let elem_tys = match &texpr.ty {
                    ResolvedTy::Tuple(ts) => ts.clone(),
                    other => {
                        ctx.errors.push(SemaError::TupleDestructureNotTuple {
                            found: other.to_string(),
                            span: stmt.span,
                        });
                        return typed;
                    }
                };
                if names.len() != elem_tys.len() {
                    ctx.errors.push(SemaError::TupleDestructureArity {
                        expected: elem_tys.len(),
                        found: names.len(),
                        span: stmt.span,
                    });
                    return typed;
                }
                let name_tys: Vec<(String, ResolvedTy)> = names.iter()
                    .zip(elem_tys.iter())
                    .map(|(n, t)| (n.clone(), t.clone()))
                    .collect();
                typed.push(TypedStmt::LetTuple { names: name_tys.clone(), expr: texpr });
                for (name, ty) in name_tys {
                    if *mutable {
                        ctx.env.bind_mutable(name, ty, None);
                    } else {
                        ctx.env.bind(name, ty, None);
                    }
                }
            }
            malus_syntax::ast::StmtKind::Assign { target, expr } => {
                match check_lvalue(target, expr, stmt.span, ctx) {
                    Some(stmt) => typed.push(stmt),
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
                // Condition must be Bool (or any Scalar in kernel bodies, where
                // comparisons yield a scalar mask that MSL treats as truthy).
                let tcond = match check_expr(condition, Some(&ResolvedTy::Bool), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                if !ctx.in_kernel && tcond.ty != ResolvedTy::Bool {
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
            StmtKind::Break => {
                if ctx.loop_depth == 0 {
                    ctx.errors.push(SemaError::BreakOutsideLoop { span: stmt.span });
                    return typed;
                }
                typed.push(TypedStmt::Break);
            }
            StmtKind::Continue => {
                if ctx.loop_depth == 0 {
                    ctx.errors.push(SemaError::ContinueOutsideLoop { span: stmt.span });
                    return typed;
                }
                typed.push(TypedStmt::Continue);
            }
            StmtKind::NoGrad { body } => {
                ctx.env.push_scope();
                ctx.no_grad_depth += 1;
                let tbody = check_body(body, ctx);
                ctx.no_grad_depth -= 1;
                ctx.env.pop_scope();
                typed.push(TypedStmt::NoGrad { body: tbody });
            }
            StmtKind::For { var, start, end, body } => {
                // In kernel bodies, loop variable and bounds are I32 (matching thread
                // intrinsics which return i32).  In fn bodies, I64 (integer default).
                let loop_var_ty = if ctx.in_kernel {
                    ResolvedTy::Scalar(ScalarTy::I32)
                } else {
                    ResolvedTy::Scalar(ScalarTy::I64)
                };
                let tstart = match check_expr(start, Some(&loop_var_ty), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                let tend = match check_expr(end, Some(&loop_var_ty), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                ctx.env.push_scope();
                ctx.env.bind(var.clone(), loop_var_ty, None);
                ctx.loop_depth += 1;
                let tbody = check_body(body, ctx);
                ctx.loop_depth -= 1;
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
                ctx.loop_depth += 1;
                let tbody = check_body(body, ctx);
                ctx.loop_depth -= 1;
                ctx.env.pop_scope();
                typed.push(TypedStmt::ForIn { var: var.clone(), iter: titer, body: tbody });
            }
            StmtKind::While { condition, body } => {
                // Condition must be Bool (or any Scalar in kernel bodies).
                let tcond = match check_expr(condition, Some(&ResolvedTy::Bool), ctx) {
                    Some(e) => e,
                    None => return typed,
                };
                if !ctx.in_kernel && tcond.ty != ResolvedTy::Bool {
                    ctx.errors.push(SemaError::TypeMismatch {
                        expected: ResolvedTy::Bool,
                        found: tcond.ty.clone(),
                        span: condition.span,
                    });
                    return typed;
                }
                ctx.env.push_scope();
                ctx.loop_depth += 1;
                let tbody = check_body(body, ctx);
                ctx.loop_depth -= 1;
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
                        let fpl = if fty.is_tensor() || fty.is_variable() { Some(Placement::Gpu) } else { None };
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
            // ── M24: threadgroup shared memory ────────────────────────────────────
            StmtKind::LetShared { name, elem_ty, size } => {
                if !ctx.in_kernel {
                    ctx.errors.push(SemaError::LetSharedOutsideKernel { span: stmt.span });
                    return typed;
                }
                // Bind as a mutable Array<Scalar(elem_ty), size> so `scratch[i]=val` works.
                let arr_ty = ResolvedTy::Array {
                    elem: Box::new(ResolvedTy::Scalar(elem_ty.clone())),
                    len: *size,
                };
                ctx.env.bind_mutable(name.clone(), arr_ty, None);
                typed.push(TypedStmt::LetShared { name: name.clone(), elem_ty: elem_ty.clone(), size: *size });
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
        ExprKind::Tuple(elements) => check_tuple(elements, expr.span, ctx),
        ExprKind::TupleIndex { base, index } => check_tuple_index(base, *index, expr.span, ctx),
        ExprKind::KernelLaunch { kernel, config, args } =>
            check_kernel_launch(kernel, config, args, expr.span, ctx),
    }
}

fn check_lit(
    lit: &Lit,
    expected: Option<&ResolvedTy>,
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let ty = match lit {
        Lit::Int(_) => {
            // Coerce to float if expected type is a float scalar — lossless widening.
            // In kernel bodies, integer literals default to I32 (matching thread intrinsics
            // and scalar uniform params declared as i32). In fn bodies, default to I64.
            match expected {
                Some(ResolvedTy::Scalar(s)) if is_float_scalar(s) => ResolvedTy::Scalar(s.clone()),
                Some(ResolvedTy::Scalar(ScalarTy::I32)) => ResolvedTy::Scalar(ScalarTy::I32),
                _ if ctx.in_kernel => ResolvedTy::Scalar(ScalarTy::I32),
                _ => ResolvedTy::Scalar(ScalarTy::I64),
            }
        }
        Lit::Float(_) => ResolvedTy::Scalar(ScalarTy::F32),
        Lit::Bool(_) => ResolvedTy::Bool,
        // String literals always have type Str.  When used as the first arg of
        // print/println they are also pattern-matched by codegen as a
        // compile-time format template (via TypedExprKind::Lit(Lit::Str(_))),
        // independently of their resolved type.
        Lit::Str(_) => ResolvedTy::Str,
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
    // `**` is scalar-only: f32 ** {f32|i32|i64} → f32.
    // Check this before the general Variable/Tensor path.
    if *op == BinOp::Pow {
        let tlhs = check_expr(lhs, Some(&ResolvedTy::Scalar(ScalarTy::F32)), ctx)?;
        let trhs = check_expr(rhs, None, ctx)?;
        let lhs_ok = matches!(&tlhs.ty, ResolvedTy::Scalar(ScalarTy::F32));
        let rhs_ok = matches!(&trhs.ty,
            ResolvedTy::Scalar(ScalarTy::F32) | ResolvedTy::Scalar(ScalarTy::I32)
            | ResolvedTy::Scalar(ScalarTy::I64));
        if !lhs_ok || !rhs_ok {
            ctx.errors.push(SemaError::PowOperatorScalarOnly { span });
            return None;
        }
        return Some(typed_expr(
            TypedExprKind::BinOp { op: BinOp::Pow, lhs: Box::new(tlhs), rhs: Box::new(trhs) },
            ResolvedTy::Scalar(ScalarTy::F32),
            None,
            span,
        ));
    }

    let tlhs = check_expr(lhs, None, ctx)?;
    let trhs = check_expr(rhs, None, ctx)?;

    // Variable⊗Variable → Variable for arithmetic and matmul ops.
    if tlhs.ty.is_variable() && trhs.ty.is_variable() {
        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Matmul) {
            ctx.errors.push(SemaError::TypeMismatch {
                expected: tlhs.ty.clone(),
                found: trhs.ty.clone(),
                span,
            });
            return None;
        }
        let ldtype = match &tlhs.ty { ResolvedTy::Variable { dtype } => dtype.clone(), _ => unreachable!() };
        let rdtype = match &trhs.ty { ResolvedTy::Variable { dtype } => dtype.clone(), _ => unreachable!() };
        if ldtype != rdtype {
            ctx.errors.push(SemaError::DtypeMismatch {
                lhs: scalar_ty_name(&ldtype).to_string(),
                rhs: scalar_ty_name(&rdtype).to_string(),
                span,
            });
            return None;
        }
        return Some(typed_expr(
            TypedExprKind::BinOp { op: op.clone(), lhs: Box::new(tlhs), rhs: Box::new(trhs) },
            ResolvedTy::Variable { dtype: ldtype },
            Some(Placement::Gpu),
            span,
        ));
    }

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
            // Reduction builtins (sum/mean/max/var) accept named args axis=/keepdim=
            // and return early via a dedicated checker.
            if let BuiltinKind::Reduction = &sig.kind {
                return check_reduction_call(&callee_name, args, span, ctx);
            }
            // TensorThenShapeArgs builtins (reshape/transpose/permute) return early
            // via a dedicated checker.
            if let BuiltinKind::TensorThenShapeArgs = &sig.kind {
                return check_shape_op_call(&callee_name, args, span, ctx);
            }
            // AxisOnly builtins (softmax/layernorm) accept named arg axis= (required).
            if let BuiltinKind::AxisOnly = &sig.kind {
                return check_axis_only_call(&callee_name, args, span, ctx);
            }
            // KernelOnly builtins (thread intrinsics, barrier, fmax/fmin, rsqrt).
            if let BuiltinKind::KernelOnly { params, ret } = &sig.kind {
                let params = params.clone();
                let ret = ret.clone();
                if !ctx.in_kernel {
                    ctx.errors.push(SemaError::KernelIntrinsicOutsideKernel {
                        name: callee_name.clone(),
                        span,
                    });
                    return None;
                }
                if positional.len() != params.len() {
                    ctx.errors.push(SemaError::ArgCountMismatch {
                        callee: callee_name.clone(),
                        expected: params.len(),
                        found: positional.len(),
                        span,
                    });
                    return None;
                }
                let mut typed_args = Vec::new();
                for (arg, param_ty) in positional.iter().zip(params.iter()) {
                    typed_args.push(check_expr(arg, Some(param_ty), ctx)?);
                }
                return Some(typed_expr(
                    TypedExprKind::Call { callee: callee_name, args: typed_args },
                    ret,
                    None,
                    span,
                ));
            }
            let is_print_call = callee_name == "print" || callee_name == "println";
            let typed_args: Vec<TypedExpr> = match &sig.kind {
                BuiltinKind::Variadic => {
                    let mut out = Vec::new();
                    for arg in positional.iter() {
                        let checked = check_expr(arg, None, ctx)?;
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
                BuiltinKind::VariadicTyped(expected) => {
                    let mut out = Vec::new();
                    for arg in positional.iter() {
                        let ta = check_expr(arg, Some(expected), ctx)?;
                        if ta.ty != *expected {
                            ctx.errors.push(SemaError::TypeMismatch {
                                expected: expected.clone(),
                                found: ta.ty.clone(),
                                span: arg.span,
                            });
                            return None;
                        }
                        out.push(ta);
                    }
                    out
                }
                BuiltinKind::Reduction => unreachable!("Reduction handled above"),
                BuiltinKind::TensorThenShapeArgs => unreachable!("TensorThenShapeArgs handled above"),
                BuiltinKind::AxisOnly => unreachable!("AxisOnly handled above"),
                BuiltinKind::KernelOnly { .. } => unreachable!("KernelOnly handled above"),
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
            // Unary-tensor builtins accept Variable<f32> and return Variable<f32>:
            // relu/sigmoid/tanh/exp/log/sqrt/abs/transpose/sum take Tensor<f32> in the
            // builtin table, but when the caller passes a Variable the VJP path needs
            // the return type to be Variable so codegen emits tape_record_unary.
            let return_ty = if let BuiltinKind::Fixed(params) = &sig.kind {
                if params.len() == 1
                    && sig.return_ty.is_tensor()
                    && typed_args.len() == 1
                    && typed_args[0].ty.is_variable()
                {
                    if let ResolvedTy::Variable { dtype } = &typed_args[0].ty {
                        ResolvedTy::Variable { dtype: dtype.clone() }
                    } else {
                        sig.return_ty.clone()
                    }
                } else if ctx.in_kernel
                    && sig.return_ty.is_tensor()
                    && typed_args.iter().all(|a| matches!(&a.ty, ResolvedTy::Scalar(_)))
                    && !typed_args.is_empty()
                {
                    // Scalar-math pass-through in explicit kernel bodies.
                    // `exp(a[i])` → `Scalar(F32)` not `Tensor<F32>`.
                    if let ResolvedTy::Tensor { dtype } = &sig.return_ty {
                        ResolvedTy::Scalar(dtype.clone())
                    } else {
                        sig.return_ty.clone()
                    }
                } else {
                    sig.return_ty.clone()
                }
            } else {
                sig.return_ty.clone()
            };
            Some(typed_expr(
                TypedExprKind::Call { callee: callee_name, args: typed_args },
                return_ty,
                placement,
                span,
            ))
        }
    }
}

// ── Shape-op call checking ────────────────────────────────────────────────────
//
// Handles reshape(t, d0..dn), transpose(t[, i, j]), permute(t, p0..pn).
// First arg must be Tensor<f32> or Variable<f32>; remaining args are i64 dims.
// Normalizes to positional [tensor, d0..dn].  Variable input propagates to
// Variable output (same as reduction checking).  No shape/count validation in
// sema — runtime panics on mismatch (ADR-0013).

fn check_shape_op_call(
    callee: &str,
    raw_args: &[CallArg],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let tensor_f32 = ResolvedTy::Tensor { dtype: ScalarTy::F32 };

    // Positional-only: named args are not allowed for shape ops.
    for arg in raw_args {
        if arg.name.is_some() {
            ctx.errors.push(SemaError::ArgCountMismatch {
                callee: callee.to_string(),
                expected: 0,
                found: 0,
                span,
            });
            return None;
        }
    }
    let positional: Vec<&malus_syntax::ast::Expr> =
        raw_args.iter().map(|a| &a.value).collect();

    if positional.is_empty() {
        ctx.errors.push(SemaError::ArgCountMismatch {
            callee: callee.to_string(),
            expected: 1,
            found: 0,
            span,
        });
        return None;
    }

    // Check the leading tensor/variable arg.
    let checked_tensor = check_expr(positional[0], Some(&tensor_f32), ctx)?;
    let is_variable = checked_tensor.ty.is_variable();

    // Check each remaining dim arg with an I64 hint (shape args are i64 scalars).
    let mut typed_args: Vec<TypedExpr> = vec![checked_tensor];
    for dim_arg in &positional[1..] {
        typed_args.push(check_expr(dim_arg, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?);
    }

    // Propagate Variable: if input is Variable<f32>, output is Variable<f32>.
    let return_ty = if is_variable {
        ResolvedTy::Variable { dtype: ScalarTy::F32 }
    } else {
        tensor_f32
    };

    Some(typed_expr(
        TypedExprKind::Call { callee: callee.to_string(), args: typed_args },
        return_ty,
        Some(Placement::Gpu),
        span,
    ))
}

// ── Reduction call checking ───────────────────────────────────────────────────
//
// Handles `sum`, `mean`, `max`, `var` with optional named args `axis=i32` and
// `keepdim=i32` (0/1).  Normalizes to positional [tensor, axis, keepdim] for
// axis reductions.  For `sum` with no `axis=`, falls through to the 1-arg
// whole-tensor path.  Variable<f32> input propagates to Variable<f32> output.

fn check_reduction_call(
    callee: &str,
    raw_args: &[CallArg],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let tensor_f32 = ResolvedTy::Tensor { dtype: ScalarTy::F32 };

    let mut tensor_raw: Option<&malus_syntax::ast::Expr> = None;
    let mut axis_raw: Option<&malus_syntax::ast::Expr> = None;
    let mut keepdim_raw: Option<&malus_syntax::ast::Expr> = None;

    for ca in raw_args {
        match ca.name.as_deref() {
            None => {
                if tensor_raw.is_some() {
                    let positional_count = raw_args.iter().filter(|a| a.name.is_none()).count();
                    ctx.errors.push(SemaError::ArgCountMismatch {
                        callee: callee.to_string(),
                        expected: 1,
                        found: positional_count,
                        span,
                    });
                    return None;
                }
                tensor_raw = Some(&ca.value);
            }
            Some("axis") => axis_raw = Some(&ca.value),
            Some("keepdim") => keepdim_raw = Some(&ca.value),
            Some(bad) => {
                ctx.errors.push(SemaError::UnknownReductionArg {
                    name: bad.to_string(),
                    span: ca.value.span,
                });
                return None;
            }
        }
    }

    let tensor_raw = match tensor_raw {
        Some(e) => e,
        None => {
            ctx.errors.push(SemaError::ArgCountMismatch {
                callee: callee.to_string(),
                expected: 1,
                found: 0,
                span,
            });
            return None;
        }
    };

    let checked_tensor = check_expr(tensor_raw, Some(&tensor_f32), ctx)?;
    let is_variable = checked_tensor.ty.is_variable();

    // sum(t) with no axis= — whole-tensor backward-compatible 1-arg form.
    if callee == "sum" && axis_raw.is_none() {
        let return_ty = if is_variable {
            if let ResolvedTy::Variable { dtype } = &checked_tensor.ty {
                ResolvedTy::Variable { dtype: dtype.clone() }
            } else {
                tensor_f32
            }
        } else {
            tensor_f32
        };
        return Some(typed_expr(
            TypedExprKind::Call { callee: callee.to_string(), args: vec![checked_tensor] },
            return_ty,
            Some(Placement::Gpu),
            span,
        ));
    }

    // mean/max/var (and axis-form sum) — axis= required.
    let axis_raw = match axis_raw {
        Some(e) => e,
        None => {
            ctx.errors.push(SemaError::MissingReductionAxis {
                callee: callee.to_string(),
                span,
            });
            return None;
        }
    };

    // Runtime ABI uses i64 for axis and keepdim; hint I64 so check_lit produces I64.
    let checked_axis = check_expr(axis_raw, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?;

    // keepdim defaults to 0 (false); I64 throughout to match runtime sig and the default path.
    let checked_keepdim = if let Some(kd) = keepdim_raw {
        check_expr(kd, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?
    } else {
        typed_expr(
            TypedExprKind::Lit(Lit::Int(0)),
            ResolvedTy::Scalar(ScalarTy::I64),
            None,
            span,
        )
    };

    let return_ty = if is_variable {
        if let ResolvedTy::Variable { dtype } = &checked_tensor.ty {
            ResolvedTy::Variable { dtype: dtype.clone() }
        } else {
            tensor_f32
        }
    } else {
        tensor_f32
    };

    Some(typed_expr(
        TypedExprKind::Call {
            callee: callee.to_string(),
            args: vec![checked_tensor, checked_axis, checked_keepdim],
        },
        return_ty,
        Some(Placement::Gpu),
        span,
    ))
}

// ── M18: AxisOnly builtins (softmax, layernorm) ──────────────────────────────
// One positional tensor/variable arg + required named `axis=N`.
// Normalizes to positional [tensor, axis].  Variable propagates to Variable output.

fn check_axis_only_call(
    callee: &str,
    raw_args: &[CallArg],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let tensor_f32 = ResolvedTy::Tensor { dtype: ScalarTy::F32 };

    let mut tensor_raw: Option<&malus_syntax::ast::Expr> = None;
    let mut axis_raw: Option<&malus_syntax::ast::Expr> = None;

    for ca in raw_args {
        match ca.name.as_deref() {
            None => {
                if tensor_raw.is_some() {
                    let positional_count = raw_args.iter().filter(|a| a.name.is_none()).count();
                    ctx.errors.push(SemaError::ArgCountMismatch {
                        callee: callee.to_string(),
                        expected: 1,
                        found: positional_count,
                        span,
                    });
                    return None;
                }
                tensor_raw = Some(&ca.value);
            }
            Some("axis") => axis_raw = Some(&ca.value),
            Some(bad) => {
                ctx.errors.push(SemaError::UnknownAxisArg {
                    callee: callee.to_string(),
                    name: bad.to_string(),
                    span: ca.value.span,
                });
                return None;
            }
        }
    }

    let tensor_raw = match tensor_raw {
        Some(e) => e,
        None => {
            ctx.errors.push(SemaError::ArgCountMismatch {
                callee: callee.to_string(),
                expected: 1,
                found: 0,
                span,
            });
            return None;
        }
    };

    let axis_raw = match axis_raw {
        Some(e) => e,
        None => {
            ctx.errors.push(SemaError::MissingAxisArg {
                callee: callee.to_string(),
                span,
            });
            return None;
        }
    };

    let checked_tensor = check_expr(tensor_raw, Some(&tensor_f32), ctx)?;
    let is_variable = checked_tensor.ty.is_variable();
    // Runtime ABI uses i64 for axis; hint I64 so check_lit produces I64.
    let checked_axis = check_expr(axis_raw, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?;

    let return_ty = if is_variable {
        if let ResolvedTy::Variable { dtype } = &checked_tensor.ty {
            ResolvedTy::Variable { dtype: dtype.clone() }
        } else {
            tensor_f32
        }
    } else {
        tensor_f32
    };

    Some(typed_expr(
        TypedExprKind::Call {
            callee: callee.to_string(),
            args: vec![checked_tensor, checked_axis],
        },
        return_ty,
        Some(Placement::Gpu),
        span,
    ))
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
    // Buffer<dtype>[i] → Scalar(I64)  (sign-extended element read)
    if let ResolvedTy::Buffer { .. } = &tbase.ty.clone() {
        let mut typed_indices: Vec<TypedExpr> = Vec::new();
        for idx in indices {
            typed_indices.push(check_expr(idx, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?);
        }
        return Some(typed_expr(
            TypedExprKind::Index { base: Box::new(tbase), indices: typed_indices },
            ResolvedTy::Scalar(ScalarTy::I64),
            None,
            span,
        ));
    }
    // Tensor<dtype>[i] → Scalar(dtype)  (flat row-major read; covers f32, i32, i64, etc.)
    if let ResolvedTy::Tensor { dtype } = &tbase.ty.clone() {
        let elem_ty = ResolvedTy::Scalar(dtype.clone());
        let mut typed_indices: Vec<TypedExpr> = Vec::new();
        for idx in indices {
            typed_indices.push(check_expr(idx, Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?);
        }
        return Some(typed_expr(
            TypedExprKind::Index { base: Box::new(tbase), indices: typed_indices },
            elem_ty,
            None,
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

    // .ndim on a tensor or variable → Scalar(I64) in fn bodies, Scalar(I32) in kernel bodies.
    // Kernel bodies use I32 to match thread intrinsics and scalar uniform params.
    if field == "ndim" && (tbase.ty.is_tensor() || tbase.ty.is_variable()) {
        let idx_ty = if ctx.in_kernel { ScalarTy::I32 } else { ScalarTy::I64 };
        return Some(typed_expr(
            TypedExprKind::FieldAccess { base: Box::new(tbase), field: field.to_string() },
            ResolvedTy::Scalar(idx_ty),
            None,
            span,
        ));
    }

    // .shape and .strides are valid only when immediately indexed (t.shape[i], t.strides[i]).
    // In fn bodies, element type is I64. In kernel bodies, I32 (matching thread intrinsics).
    if (field == "shape" || field == "strides") && (tbase.ty.is_tensor() || tbase.ty.is_variable()) {
        let placement = tbase.placement;
        let idx_ty = if ctx.in_kernel { ScalarTy::I32 } else { ScalarTy::I64 };
        return Some(typed_expr(
            TypedExprKind::FieldAccess { base: Box::new(tbase), field: field.to_string() },
            ResolvedTy::Array { elem: Box::new(ResolvedTy::Scalar(idx_ty)), len: 8 },
            placement,
            span,
        ));
    }

    // .data on a Variable returns the underlying Tensor.
    if field == "data" {
        if let ResolvedTy::Variable { dtype } = &tbase.ty.clone() {
            return Some(typed_expr(
                TypedExprKind::FieldAccess { base: Box::new(tbase), field: field.to_string() },
                ResolvedTy::Tensor { dtype: dtype.clone() },
                Some(malus_syntax::ast::Placement::Gpu),
                span,
            ));
        }
    }

    // .grad on a Variable returns an owned Tensor (retained by tape_get_grad, see D5).
    if field == "grad" {
        if let ResolvedTy::Variable { dtype } = &tbase.ty.clone() {
            return Some(typed_expr(
                TypedExprKind::FieldAccess { base: Box::new(tbase), field: field.to_string() },
                ResolvedTy::Tensor { dtype: dtype.clone() },
                Some(malus_syntax::ast::Placement::Gpu),
                span,
            ));
        }
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

// ── Tuple expressions (M13.5) ────────────────────────────────────────────────

fn check_tuple(
    elements: &[malus_syntax::ast::Expr],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    if elements.len() < 2 {
        ctx.errors.push(SemaError::TupleTooShort { span });
        return None;
    }
    let mut typed_elements = Vec::new();
    let mut elem_tys = Vec::new();
    for e in elements {
        let te = check_expr(e, None, ctx)?;
        if te.ty.is_tuple() {
            ctx.errors.push(SemaError::NestedTuple { span: e.span });
            return None;
        }
        elem_tys.push(te.ty.clone());
        typed_elements.push(te);
    }
    Some(typed_expr(
        TypedExprKind::TupleInit { elements: typed_elements },
        ResolvedTy::Tuple(elem_tys),
        None,
        span,
    ))
}

fn check_tuple_index(
    base: &malus_syntax::ast::Expr,
    index: usize,
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    let tbase = check_expr(base, None, ctx)?;
    match tbase.ty.clone() {
        ResolvedTy::Tuple(ref elems) => {
            if index >= elems.len() {
                ctx.errors.push(SemaError::TupleIndexOutOfRange {
                    len: elems.len(),
                    index,
                    span,
                });
                return None;
            }
            let elem_ty = elems[index].clone();
            Some(typed_expr(
                TypedExprKind::TupleIndex { base: Box::new(tbase), index },
                elem_ty,
                None,
                span,
            ))
        }
        other => {
            ctx.errors.push(SemaError::TupleIndexNotTuple { found: other.to_string(), span });
            None
        }
    }
}

// ── M25 kernel launch expression ─────────────────────────────────────────────

fn check_kernel_launch(
    kernel_name: &str,
    config: &[(String, malus_syntax::ast::Expr)],
    args: &[CallArg],
    span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedExpr> {
    // Kernel launches are only valid in fn bodies, not inside kernel bodies.
    if ctx.in_kernel {
        ctx.errors.push(SemaError::KernelLaunchInsideKernel { span });
        return None;
    }

    // Look up the kernel signature.
    let ksig = match ctx.env.kernels.get(kernel_name) {
        Some(s) => s.clone(),
        None => {
            ctx.errors.push(SemaError::UnknownKernel { name: kernel_name.to_string(), span });
            return None;
        }
    };

    // Extract and type-check config entries: grid (required), tg (required), out (optional).
    let mut grid_expr: Option<malus_syntax::ast::Expr> = None;
    let mut tg_expr: Option<malus_syntax::ast::Expr> = None;
    let mut out_expr: Option<malus_syntax::ast::Expr> = None;
    for (key, val) in config {
        match key.as_str() {
            "grid" => grid_expr = Some(val.clone()),
            "tg"   => tg_expr   = Some(val.clone()),
            "out"  => out_expr  = Some(val.clone()),
            other => {
                // Unknown key — ignore with a note (future-proofing); just emit a mismatch.
                ctx.errors.push(SemaError::UnknownReductionArg { name: other.to_string(), span });
            }
        }
    }
    let grid_ast = match grid_expr {
        Some(e) => e,
        None => {
            ctx.errors.push(SemaError::MissingLaunchConfig { key: "grid".to_string(), span });
            return None;
        }
    };
    let tg_ast = match tg_expr {
        Some(e) => e,
        None => {
            ctx.errors.push(SemaError::MissingLaunchConfig { key: "tg".to_string(), span });
            return None;
        }
    };

    // Type-check config expressions; they must produce Array<i64,3>.
    let array3_i64 = ResolvedTy::Array {
        elem: Box::new(ResolvedTy::Scalar(ScalarTy::I64)),
        len: 3,
    };
    let tgrid = check_expr(&grid_ast, Some(&array3_i64), ctx)?;
    let ttg   = check_expr(&tg_ast,   Some(&array3_i64), ctx)?;
    let tout_shape = if let Some(e) = out_expr.as_ref() {
        Some(check_expr(e, Some(&array3_i64), ctx)?)
    } else {
        None
    };

    // Partition kernel params into tensor params and scalar params (declaration order).
    let tensor_params: Vec<&KernelParamSig> = ksig.params.iter().filter(|p| p.ty.is_tensor() || p.ty.is_variable()).collect();
    let scalar_params: Vec<&KernelParamSig> = ksig.params.iter().filter(|p| matches!(p.ty, ResolvedTy::Scalar(_))).collect();

    // Check positional runtime args match the kernel params (tensors then scalars).
    if args.len() != tensor_params.len() + scalar_params.len() {
        ctx.errors.push(SemaError::ArgCountMismatch {
            callee: kernel_name.to_string(),
            expected: tensor_params.len() + scalar_params.len(),
            found: args.len(),
            span,
        });
        return None;
    }

    let mut tensor_args: Vec<TypedExpr> = Vec::new();
    let mut scalar_args: Vec<TypedExpr> = Vec::new();

    for (i, arg) in args.iter().enumerate() {
        if i < tensor_params.len() {
            let expected = &tensor_params[i].ty;
            let ta = check_expr(&arg.value, Some(expected), ctx)?;
            tensor_args.push(ta);
        } else {
            let j = i - tensor_params.len();
            let expected = &scalar_params[j].ty;
            let ta = check_expr(&arg.value, Some(expected), ctx)?;
            scalar_args.push(ta);
        }
    }

    // Result type = kernel return type as a GPU tensor.
    let result_dtype = match &ksig.return_ty {
        ResolvedTy::Tensor { dtype } => dtype.clone(),
        other => {
            ctx.errors.push(SemaError::TypeMismatch {
                expected: ResolvedTy::Tensor { dtype: ScalarTy::F32 },
                found: other.clone(),
                span,
            });
            return None;
        }
    };
    let result_ty = ResolvedTy::Tensor { dtype: result_dtype };

    Some(typed_expr(
        TypedExprKind::KernelLaunch {
            kernel: kernel_name.to_string(),
            grid: Box::new(tgrid),
            tg: Box::new(ttg),
            out_shape: tout_shape.map(Box::new),
            tensor_args,
            scalar_args,
        },
        result_ty,
        Some(Placement::Gpu),
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
        Ty::Variable { dtype } => Some(ResolvedTy::Variable { dtype: dtype.clone() }),
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
            if resolved_elem.is_tuple() {
                errors.push(SemaError::TupleInArrayElement { span });
                return None;
            }
            Some(ResolvedTy::Array { elem: Box::new(resolved_elem), len: *len })
        }
        Ty::Buffer { dtype } => Some(ResolvedTy::Buffer { dtype: dtype.clone() }),
        Ty::Named(name) if name == "None" => Some(ResolvedTy::Unit),
        Ty::Named(name) if name == "str" => Some(ResolvedTy::Str),
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

/// Validate and lower an lvalue assignment.
///
/// Valid targets (single-level only):
///   - `Ident(name)` — bare variable rebind; requires `let mut` local (not `mut` param).
///   - `Index { base: Ident(name), .. }` — indexed element; base must be mutable.
///   - `FieldAccess { base: Ident(name), field }` — struct field; base must be mutable;
///     field must not be `Variable` (post-V3, ADR-0016).
/// Nested targets (`a[i].f`, `a.b[j]`) are rejected with `NestedLvalue`.
fn check_lvalue(
    target: &malus_syntax::ast::Expr,
    rhs: &malus_syntax::ast::Expr,
    stmt_span: Span,
    ctx: &mut BodyCtx<'_>,
) -> Option<TypedStmt> {
    use malus_syntax::ast::ExprKind;

    match &target.kind {
        ExprKind::Ident(name) => {
            // Bare rebind. Requires the binding to be mutable.
            // `mut` params are mutable for interior mutation only — reject bare rebind.
            let (target_ty, is_param) = match ctx.env.lookup_binding(name) {
                Some((ty, _)) => (ty.clone(), false),
                None => {
                    ctx.errors.push(SemaError::UnknownIdent { name: name.clone(), span: stmt_span });
                    return None;
                }
            };
            // Detect mut param: it's in mutable_names but was registered without LetMut.
            // The only way to be mutable without being a `let mut` local is to be a `mut` param.
            // We detect this by checking the AST params of the current fn, but we don't have
            // that context here. Instead, track mut-param names in the env with a separate set.
            let _ = is_param;
            if !ctx.env.is_mutable(name) {
                ctx.errors.push(SemaError::AssignToImmutable { name: name.clone(), span: stmt_span });
                return None;
            }
            // Reject `mut` param bare rebind: a mut param is interior-only.
            if ctx.env.is_mut_param(name) {
                ctx.errors.push(SemaError::MutParamBareRebind { name: name.clone(), span: stmt_span });
                return None;
            }
            let texpr = check_expr(rhs, Some(&target_ty), ctx)?;
            if texpr.ty != target_ty {
                ctx.errors.push(SemaError::TypeMismatch {
                    expected: target_ty,
                    found: texpr.ty.clone(),
                    span: rhs.span,
                });
                return None;
            }
            Some(TypedStmt::Assign {
                target: TypedAssignTarget::Ident(name.clone()),
                expr: texpr,
            })
        }

        ExprKind::Index { base, indices } => {
            // Indexed element assignment: `base[i] = rhs`
            // Base must be an Ident (no nested lvalues in M20).
            let base_name = match &base.kind {
                ExprKind::Ident(n) => n.clone(),
                _ => {
                    ctx.errors.push(SemaError::NestedLvalue { span: target.span });
                    return None;
                }
            };
            // Base must be mutable.
            let base_ty = match ctx.env.lookup_binding(&base_name) {
                Some((ty, _)) => ty.clone(),
                None => {
                    ctx.errors.push(SemaError::UnknownIdent { name: base_name.clone(), span: target.span });
                    return None;
                }
            };
            if !ctx.env.is_mutable(&base_name) {
                ctx.errors.push(SemaError::AssignToImmutable { name: base_name.clone(), span: stmt_span });
                return None;
            }
            // Type-check indices (expect a single integer index).
            if indices.is_empty() {
                ctx.errors.push(SemaError::UnknownIdent { name: "[]".into(), span: target.span });
                return None;
            }
            // Base must be an array or buffer.
            match &base_ty.clone() {
                ResolvedTy::Array { elem, .. } => {
                    let elem_ty = *elem.clone();
                    let tidx = check_expr(&indices[0], Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?;
                    let texpr = check_expr(rhs, Some(&elem_ty), ctx)?;
                    if texpr.ty != elem_ty {
                        ctx.errors.push(SemaError::TypeMismatch {
                            expected: elem_ty.clone(),
                            found: texpr.ty.clone(),
                            span: rhs.span,
                        });
                        return None;
                    }
                    Some(TypedStmt::Assign {
                        target: TypedAssignTarget::Index { base: base_name, index: Box::new(tidx), elem_ty },
                        expr: texpr,
                    })
                }
                ResolvedTy::Buffer { dtype } => {
                    let dtype = dtype.clone();
                    let elem_ty = ResolvedTy::Scalar(ScalarTy::I64);
                    let tidx = check_expr(&indices[0], Some(&ResolvedTy::Scalar(ScalarTy::I64)), ctx)?;
                    let texpr = check_expr(rhs, Some(&elem_ty), ctx)?;
                    if texpr.ty != elem_ty {
                        ctx.errors.push(SemaError::TypeMismatch {
                            expected: elem_ty.clone(),
                            found: texpr.ty.clone(),
                            span: rhs.span,
                        });
                        return None;
                    }
                    Some(TypedStmt::Assign {
                        target: TypedAssignTarget::BufferIndex { base: base_name, index: Box::new(tidx), dtype },
                        expr: texpr,
                    })
                }
                // M24: `out[expr] = val` inside an explicit kernel body.
                // `out` is bound as a mutable Tensor<F32> by the kernel scope setup.
                ResolvedTy::Tensor { dtype } => {
                    if !ctx.in_kernel {
                        ctx.errors.push(SemaError::TensorIndexAssignOutsideKernel { span: target.span });
                        return None;
                    }
                    let dtype = dtype.clone();
                    let elem_ty = ResolvedTy::Scalar(dtype.clone());
                    let idx_hint = ResolvedTy::Scalar(ScalarTy::I32);
                    let tidx = check_expr(&indices[0], Some(&idx_hint), ctx)?;
                    let texpr = check_expr(rhs, Some(&elem_ty), ctx)?;
                    if texpr.ty != elem_ty {
                        ctx.errors.push(SemaError::TypeMismatch {
                            expected: elem_ty.clone(),
                            found: texpr.ty.clone(),
                            span: rhs.span,
                        });
                        return None;
                    }
                    Some(TypedStmt::Assign {
                        target: TypedAssignTarget::Index { base: base_name, index: Box::new(tidx), elem_ty },
                        expr: texpr,
                    })
                }
                other => {
                    ctx.errors.push(SemaError::TypeMismatch {
                        expected: ResolvedTy::Array { elem: Box::new(ResolvedTy::Unit), len: 0 },
                        found: other.clone(),
                        span: target.span,
                    });
                    None
                }
            }
        }

        ExprKind::FieldAccess { base, field } => {
            // Field assignment: `base.field = rhs`
            // Base must be an Ident (no nested lvalues in M20).
            let base_name = match &base.kind {
                ExprKind::Ident(n) => n.clone(),
                _ => {
                    ctx.errors.push(SemaError::NestedLvalue { span: target.span });
                    return None;
                }
            };
            let base_ty = match ctx.env.lookup_binding(&base_name) {
                Some((ty, _)) => ty.clone(),
                None => {
                    ctx.errors.push(SemaError::UnknownIdent { name: base_name.clone(), span: target.span });
                    return None;
                }
            };
            if !ctx.env.is_mutable(&base_name) {
                ctx.errors.push(SemaError::AssignToImmutable { name: base_name.clone(), span: stmt_span });
                return None;
            }
            // Base must be a struct.
            let fields = match &base_ty {
                ResolvedTy::Struct { fields, .. } => fields.clone(),
                other => {
                    ctx.errors.push(SemaError::TypeMismatch {
                        expected: ResolvedTy::Struct { name: "<struct>".into(), fields: vec![] },
                        found: other.clone(),
                        span: target.span,
                    });
                    return None;
                }
            };
            let (slot_idx, field_ty) = match fields.iter().enumerate()
                .find(|(_, (n, _))| n == field)
            {
                Some((i, (_, ty))) => (i, ty.clone()),
                None => {
                    let sname = match &base_ty { ResolvedTy::Struct { name, .. } => name.clone(), _ => "<struct>".into() };
                    ctx.errors.push(SemaError::UnknownField {
                        struct_name: sname,
                        field: field.clone(),
                        span: target.span,
                    });
                    return None;
                }
            };
            let texpr = check_expr(rhs, Some(&field_ty), ctx)?;
            if texpr.ty != field_ty {
                ctx.errors.push(SemaError::TypeMismatch {
                    expected: field_ty.clone(),
                    found: texpr.ty.clone(),
                    span: rhs.span,
                });
                return None;
            }
            Some(TypedStmt::Assign {
                target: TypedAssignTarget::Field { base: base_name, slot_idx, field_ty },
                expr: texpr,
            })
        }

        _ => {
            ctx.errors.push(SemaError::NestedLvalue { span: target.span });
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
        out.push(ParamSig { name: p.name.clone(), ty, is_mut: p.is_mut });
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
