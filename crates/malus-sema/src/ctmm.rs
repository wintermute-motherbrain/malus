use std::collections::{HashMap, HashSet};
use malus_syntax::ast::Placement;
use crate::typed_ir::{TypedExpr, TypedExprKind, TypedFn, TypedStmt};

/// CTMM: run last-use analysis on all fn bodies, injecting Drop and GpuBarrier nodes.
/// Kernel bodies are skipped — kernels borrow their inputs and return new owned tensors;
/// there is no local allocation to free inside a kernel body in v0.1.
pub fn annotate_fns(fns: &mut Vec<TypedFn>) {
    for f in fns.iter_mut() {
        annotate_body(&mut f.body);
    }
}

fn annotate_body(body: &mut Vec<TypedStmt>) {
    hoist_gpu_subexprs(body);
    hoist_gpu_producing_returns(body);
    let local_bindings = collect_local_bindings(body);
    let escaping = collect_escaping(body);
    let last_uses = find_last_uses(body, &local_bindings, &escaping);

    insert_drops(body, &last_uses);
    insert_barriers(body);
}

fn is_gpu_producing(expr: &TypedExpr) -> bool {
    matches!(&expr.kind, TypedExprKind::KernelCall { .. })
        || matches!(&expr.kind, TypedExprKind::BinOp { lhs, .. } if lhs.ty.is_tensor())
        || matches!(&expr.kind, TypedExprKind::Call { .. } if expr.ty.is_tensor() && expr.placement == Some(Placement::Gpu))
}

fn hoist_gpu_subexprs(body: &mut Vec<TypedStmt>) {
    let mut counter = 0u32;
    let mut result: Vec<TypedStmt> = Vec::with_capacity(body.len());
    for stmt in body.drain(..) {
        match stmt {
            TypedStmt::Let { name, expr } => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, &mut counter);
                result.extend(hoisted);
                result.push(TypedStmt::Let { name, expr });
            }
            TypedStmt::Return { expr } => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, &mut counter);
                result.extend(hoisted);
                result.push(TypedStmt::Return { expr });
            }
            TypedStmt::Expr(expr) => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, &mut counter);
                result.extend(hoisted);
                result.push(TypedStmt::Expr(expr));
            }
            other => result.push(other),
        }
    }
    *body = result;
}

fn hoist_gpu_in_expr(
    expr: TypedExpr,
    hoisted: &mut Vec<TypedStmt>,
    counter: &mut u32,
) -> TypedExpr {
    let span = expr.span;
    match expr.kind {
        TypedExprKind::Call { callee, args } => {
            let new_args = hoist_args(args, hoisted, counter);
            TypedExpr {
                kind: TypedExprKind::Call { callee, args: new_args },
                span,
                ..expr
            }
        }
        TypedExprKind::KernelCall { callee, args, in_flight } => {
            let new_args = hoist_args(args, hoisted, counter);
            TypedExpr {
                kind: TypedExprKind::KernelCall { callee, args: new_args, in_flight },
                span,
                ..expr
            }
        }
        TypedExprKind::BinOp { op, lhs, rhs } => {
            TypedExpr {
                kind: TypedExprKind::BinOp {
                    op,
                    lhs: Box::new(hoist_gpu_in_expr(*lhs, hoisted, counter)),
                    rhs: Box::new(hoist_gpu_in_expr(*rhs, hoisted, counter)),
                },
                span,
                ..expr
            }
        }
        TypedExprKind::Unary { op, operand } => {
            TypedExpr {
                kind: TypedExprKind::Unary {
                    op,
                    operand: Box::new(hoist_gpu_in_expr(*operand, hoisted, counter)),
                },
                span,
                ..expr
            }
        }
        TypedExprKind::TensorLiteral { placement, dtype, elements } => {
            let new_elements = elements
                .into_iter()
                .map(|e| hoist_gpu_in_expr(e, hoisted, counter))
                .collect();
            TypedExpr {
                kind: TypedExprKind::TensorLiteral { placement, dtype, elements: new_elements },
                span,
                ..expr
            }
        }
        TypedExprKind::Index { base, indices } => {
            let new_base = Box::new(hoist_gpu_in_expr(*base, hoisted, counter));
            let new_indices = indices
                .into_iter()
                .map(|i| hoist_gpu_in_expr(i, hoisted, counter))
                .collect();
            TypedExpr {
                kind: TypedExprKind::Index { base: new_base, indices: new_indices },
                span,
                ..expr
            }
        }
        TypedExprKind::FieldAccess { base, field } => {
            TypedExpr {
                kind: TypedExprKind::FieldAccess {
                    base: Box::new(hoist_gpu_in_expr(*base, hoisted, counter)),
                    field,
                },
                span,
                ..expr
            }
        }
        _ => expr,
    }
}

fn hoist_args(
    args: Vec<TypedExpr>,
    hoisted: &mut Vec<TypedStmt>,
    counter: &mut u32,
) -> Vec<TypedExpr> {
    let mut new_args = Vec::with_capacity(args.len());
    for arg in args {
        let arg = hoist_gpu_in_expr(arg, hoisted, counter);
        if is_gpu_producing(&arg) && arg.ty.is_tensor() {
            let name = format!("__malus_tmp_{}", counter);
            *counter += 1;
            let ty = arg.ty.clone();
            let placement = arg.placement;
            let span = arg.span;
            hoisted.push(TypedStmt::Let { name: name.clone(), expr: arg });
            new_args.push(TypedExpr {
                kind: TypedExprKind::Ident(name),
                ty,
                placement,
                span,
            });
        } else {
            new_args.push(arg);
        }
    }
    new_args
}

fn hoist_gpu_producing_returns(body: &mut Vec<TypedStmt>) {
    let mut i = 0;
    let mut counter = 0u32;
    while i < body.len() {
        if let TypedStmt::Return { expr } = &body[i] {
            if is_gpu_producing(expr) && expr.ty.is_tensor() {
                let expr = if let TypedStmt::Return { expr } = body.remove(i) { expr } else { unreachable!() };
                let name = format!("__malus_ret_{}", counter);
                counter += 1;
                let ret_ty = expr.ty.clone();
                let span = expr.span;
                body.insert(i, TypedStmt::Let { name: name.clone(), expr });
                body.insert(i + 1, TypedStmt::Return {
                    expr: TypedExpr {
                        kind: TypedExprKind::Ident(name.clone()),
                        ty: ret_ty,
                        placement: None,
                        span,
                    },
                });
                i += 2;
                continue;
            }
        }
        i += 1;
    }
}

// ── Phase 1: Drop insertion ───────────────────────────────────────────────────

fn insert_drops(body: &mut Vec<TypedStmt>, last_uses: &HashMap<String, usize>) {
    let mut by_idx: HashMap<usize, Vec<String>> = HashMap::new();
    for (name, last_idx) in last_uses {
        by_idx.entry(*last_idx).or_default().push(name.clone());
    }

    let mut indices: Vec<usize> = by_idx.keys().copied().collect();
    indices.sort_by(|a, b| b.cmp(a));

    for idx in indices {
        let mut names = by_idx.remove(&idx).unwrap();
        names.sort();
        let insert_pos = idx + 1;
        for (offset, name) in names.iter().enumerate() {
            body.insert(insert_pos + offset, TypedStmt::Drop { name: name.clone() });
        }
    }
}

// ── Phase 2: Barrier insertion (GPU-pending set) ──────────────────────────────

fn insert_barriers(body: &mut Vec<TypedStmt>) {
    let mut pending: HashSet<String> = HashSet::new();
    let mut i = 0;
    while i < body.len() {
        if matches!(body[i], TypedStmt::GpuBarrier) {
            pending.clear();
            i += 1;
            continue;
        }

        if let Some((in_flight, output_name)) = extract_gpu_producing_expr(&body[i]) {
            for n in &in_flight {
                pending.insert(n.clone());
            }
            if let Some(name) = output_name {
                pending.insert(name);
            }
            if matches!(body[i], TypedStmt::Return { .. }) {
                body.insert(i, TypedStmt::GpuBarrier);
                pending.clear();
                i += 1;
            }
            i += 1;
            continue;
        }

        let mut referenced = HashSet::new();
        collect_idents_in_stmt(&body[i], &mut referenced);
        if referenced.intersection(&pending).next().is_some() {
            body.insert(i, TypedStmt::GpuBarrier);
            pending.clear();
            i += 1;
        }
        i += 1;
    }
}

fn extract_gpu_producing_expr(stmt: &TypedStmt) -> Option<(Vec<String>, Option<String>)> {
    let (expr, output_name) = match stmt {
        TypedStmt::Let { name, expr } => (expr, Some(name.clone())),
        TypedStmt::Expr(expr) => (expr, None),
        TypedStmt::Return { expr } => (expr, None),
        TypedStmt::Drop { .. } | TypedStmt::GpuBarrier => return None,
    };
    match &expr.kind {
        TypedExprKind::KernelCall { in_flight, .. } => {
            Some((in_flight.clone(), output_name))
        }
        TypedExprKind::BinOp { lhs, .. } if lhs.ty.is_tensor() => {
            let mut idents = HashSet::new();
            collect_idents_in_expr(expr, &mut idents);
            Some((idents.into_iter().collect(), output_name))
        }
        TypedExprKind::Call { args, .. } if expr.ty.is_tensor() && expr.placement == Some(Placement::Gpu) => {
            let mut idents = HashSet::new();
            for a in args {
                collect_idents_in_expr(a, &mut idents);
            }
            Some((idents.into_iter().collect(), output_name))
        }
        _ => None,
    }
}

// ── Binding collection helpers ────────────────────────────────────────────────

fn collect_local_bindings(body: &[TypedStmt]) -> HashSet<String> {
    body.iter().filter_map(|s| {
        if let TypedStmt::Let { name, .. } = s { Some(name.clone()) } else { None }
    }).collect()
}

fn collect_escaping(body: &[TypedStmt]) -> HashSet<String> {
    let mut escaping = HashSet::new();
    for stmt in body {
        if let TypedStmt::Return { expr } = stmt {
            collect_idents_in_expr(expr, &mut escaping);
        }
    }
    escaping
}

fn find_last_uses(
    body: &[TypedStmt],
    locals: &HashSet<String>,
    escaping: &HashSet<String>,
) -> HashMap<String, usize> {
    let mut last: HashMap<String, usize> = HashMap::new();
    for (idx, stmt) in body.iter().enumerate() {
        let mut used = HashSet::new();
        collect_idents_in_stmt(stmt, &mut used);
        for name in &used {
            if locals.contains(name) && !escaping.contains(name) {
                last.insert(name.clone(), idx);
            }
        }
    }
    last
}

// ── Identifier collectors ─────────────────────────────────────────────────────

fn collect_idents_in_stmt(stmt: &TypedStmt, out: &mut HashSet<String>) {
    match stmt {
        TypedStmt::Let { expr, .. } => collect_idents_in_expr(expr, out),
        TypedStmt::Return { expr } => collect_idents_in_expr(expr, out),
        TypedStmt::Expr(expr) => collect_idents_in_expr(expr, out),
        TypedStmt::Drop { .. } | TypedStmt::GpuBarrier => {}
    }
}

fn collect_idents_in_expr(expr: &crate::typed_ir::TypedExpr, out: &mut HashSet<String>) {
    match &expr.kind {
        TypedExprKind::Ident(name) => { out.insert(name.clone()); }
        TypedExprKind::BinOp { lhs, rhs, .. } => {
            collect_idents_in_expr(lhs, out);
            collect_idents_in_expr(rhs, out);
        }
        TypedExprKind::Unary { operand, .. } => collect_idents_in_expr(operand, out),
        TypedExprKind::Call { args, .. } => {
            for a in args { collect_idents_in_expr(a, out); }
        }
        TypedExprKind::KernelCall { args, .. } => {
            for a in args { collect_idents_in_expr(a, out); }
        }
        TypedExprKind::TensorLiteral { elements, .. } => {
            for e in elements { collect_idents_in_expr(e, out); }
        }
        TypedExprKind::Index { base, indices } => {
            collect_idents_in_expr(base, out);
            for i in indices { collect_idents_in_expr(i, out); }
        }
        TypedExprKind::FieldAccess { base, .. } => collect_idents_in_expr(base, out),
        TypedExprKind::Lit(_) => {}
    }
}
