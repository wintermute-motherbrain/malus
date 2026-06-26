use std::collections::{HashMap, HashSet};
use crate::typed_ir::{TypedExprKind, TypedFn, TypedStmt};

/// CTMM: run last-use analysis on all fn bodies, injecting Drop and GpuBarrier nodes.
/// Kernel bodies are skipped — kernels borrow their inputs and return new owned tensors;
/// there is no local allocation to free inside a kernel body in v0.1.
pub fn annotate_fns(fns: &mut Vec<TypedFn>) {
    for f in fns.iter_mut() {
        annotate_body(&mut f.body);
    }
}

fn annotate_body(body: &mut Vec<TypedStmt>) {
    let local_bindings = collect_local_bindings(body);
    let escaping = collect_escaping(body);
    let last_uses = find_last_uses(body, &local_bindings, &escaping);

    insert_drops(body, &last_uses);
    insert_barriers(body);
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

        if let Some((in_flight, output_name)) = extract_kernel_call(&body[i]) {
            for n in in_flight {
                pending.insert(n.clone());
            }
            if let Some(name) = output_name {
                pending.insert(name.to_string());
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

fn extract_kernel_call<'a>(stmt: &'a TypedStmt) -> Option<(&'a Vec<String>, Option<&'a str>)> {
    match stmt {
        TypedStmt::Let { name, expr } => {
            if let TypedExprKind::KernelCall { in_flight, .. } = &expr.kind {
                Some((in_flight, Some(name)))
            } else {
                None
            }
        }
        TypedStmt::Expr(expr) => {
            if let TypedExprKind::KernelCall { in_flight, .. } = &expr.kind {
                Some((in_flight, None))
            } else {
                None
            }
        }
        TypedStmt::Return { expr } => {
            if let TypedExprKind::KernelCall { in_flight, .. } = &expr.kind {
                Some((in_flight, None))
            } else {
                None
            }
        }
        TypedStmt::Drop { .. } | TypedStmt::GpuBarrier => None,
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
