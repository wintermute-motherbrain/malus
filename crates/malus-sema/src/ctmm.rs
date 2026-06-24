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

/// For each local tensor binding in the body, find its last use, then inject
/// a Drop (and GpuBarrier if in-flight) immediately after that statement.
fn annotate_body(body: &mut Vec<TypedStmt>) {
    // Determine which bindings are defined locally (excluding params — those are
    // injected by the caller and tracked separately).
    // We only inject Drop for `let` bindings within this body.
    let local_bindings = collect_local_bindings(body);

    // Collect the set of bindings that escape (appear in a Return expr).
    let escaping = collect_escaping(body);

    // For each non-escaping local binding, find the index of its last use.
    let last_uses = find_last_uses(body, &local_bindings, &escaping);

    // Track which bindings were passed to a KernelCall (in-flight).
    let in_flight_bindings = collect_in_flight(body);

    // Group drops by their last-use statement index.
    // For each group, emit one GpuBarrier (if any drop is in-flight) then all Drops.
    let mut by_idx: HashMap<usize, (bool, Vec<String>)> = HashMap::new();
    for (name, last_idx) in &last_uses {
        let entry = by_idx.entry(*last_idx).or_insert((false, Vec::new()));
        if in_flight_bindings.contains(name) {
            entry.0 = true; // needs a barrier
        }
        entry.1.push(name.clone());
    }

    // Sort indices descending so insertions don't shift earlier positions.
    let mut indices: Vec<usize> = by_idx.keys().copied().collect();
    indices.sort_by(|a, b| b.cmp(a));

    for idx in indices {
        let (needs_barrier, mut names) = by_idx.remove(&idx).unwrap();
        names.sort(); // deterministic ordering
        let insert_pos = idx + 1;
        let mut offset = 0;
        if needs_barrier {
            body.insert(insert_pos, TypedStmt::GpuBarrier);
            offset += 1;
        }
        for name in names {
            body.insert(insert_pos + offset, TypedStmt::Drop { name });
            offset += 1;
        }
    }
}

/// Collect the names of all `let` bindings defined in the body.
fn collect_local_bindings(body: &[TypedStmt]) -> HashSet<String> {
    body.iter().filter_map(|s| {
        if let TypedStmt::Let { name, .. } = s { Some(name.clone()) } else { None }
    }).collect()
}

/// Collect binding names that appear in a Return expression.
fn collect_escaping(body: &[TypedStmt]) -> HashSet<String> {
    let mut escaping = HashSet::new();
    for stmt in body {
        if let TypedStmt::Return { expr } = stmt {
            collect_idents_in_expr(expr, &mut escaping);
        }
    }
    escaping
}

/// Find the last statement index at which each (non-escaping) local binding is used.
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

/// Collect all binding names passed to a KernelCall anywhere in the body.
fn collect_in_flight(body: &[TypedStmt]) -> HashSet<String> {
    let mut in_flight = HashSet::new();
    for stmt in body {
        collect_in_flight_stmt(stmt, &mut in_flight);
    }
    in_flight
}

fn collect_in_flight_stmt(stmt: &TypedStmt, out: &mut HashSet<String>) {
    match stmt {
        TypedStmt::Let { expr, .. } => collect_in_flight_expr(expr, out),
        TypedStmt::Return { expr } => collect_in_flight_expr(expr, out),
        TypedStmt::Expr(expr) => collect_in_flight_expr(expr, out),
        TypedStmt::Drop { .. } | TypedStmt::GpuBarrier => {}
    }
}

fn collect_in_flight_expr(expr: &crate::typed_ir::TypedExpr, out: &mut HashSet<String>) {
    match &expr.kind {
        TypedExprKind::KernelCall { in_flight, args, .. } => {
            out.extend(in_flight.iter().cloned());
            for arg in args { collect_in_flight_expr(arg, out); }
        }
        TypedExprKind::BinOp { lhs, rhs, .. } => {
            collect_in_flight_expr(lhs, out);
            collect_in_flight_expr(rhs, out);
        }
        TypedExprKind::Unary { operand, .. } => collect_in_flight_expr(operand, out),
        TypedExprKind::Call { args, .. } => {
            for arg in args { collect_in_flight_expr(arg, out); }
        }
        TypedExprKind::TensorLiteral { elements, .. } => {
            for e in elements { collect_in_flight_expr(e, out); }
        }
        TypedExprKind::Index { base, indices } => {
            collect_in_flight_expr(base, out);
            for i in indices { collect_in_flight_expr(i, out); }
        }
        TypedExprKind::FieldAccess { base, .. } => collect_in_flight_expr(base, out),
        TypedExprKind::Lit(_) | TypedExprKind::Ident(_) => {}
    }
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
