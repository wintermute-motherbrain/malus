// CTMM: hierarchical last-use analysis for `fn` bodies.
//
// Injects `Drop` and `GpuBarrier` nodes so tensor allocations are freed
// statically without reference counting.  Kernel bodies are skipped — they
// borrow their inputs and return a new owned tensor; no local allocations
// exist inside a kernel body in V1.
//
// ## Hierarchical pass order (M9, see ADR-0014)
//
// Control flow (`if`/`for`/`while`) created a problem for the original flat
// linear scan: a tensor whose last use was inside a branch would get its `Drop`
// placed at the wrong position (or skipped).  The hierarchical analysis fixes
// this by recursing into inner scopes *first*:
//
//   1. `hoist_gpu_subexprs`          — outer body, control-flow nodes passed through unchanged
//   2. `hoist_gpu_producing_returns` — outer body, control-flow nodes passed through unchanged
//   3. `recurse_into_inner_scopes`   — full `annotate_body` on each if/for/while body
//   4. `collect_local_bindings`      — outer body only (inner bindings handled in step 3)
//   5. `collect_escaping`            — recurses into inner bodies (return inside branch)
//   6. `insert_assign_drops`         — recurses into inner bodies; `locals` guard removed
//   7. `find_last_uses`              — `collect_idents_in_stmt` recurses, so outer analysis
//                                      sees inner references and records last-use at the
//                                      control-flow node's position in the outer body
//   8. `insert_drops`                — outer body only
//   9. `insert_barriers`             — outer body with precise recursive boundary check

use std::collections::{HashMap, HashSet};
use malus_syntax::ast::Placement;
use crate::typed_ir::{TypedExpr, TypedExprKind, TypedFn, TypedStmt};

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the CTMM analysis on all `fn` bodies, injecting `Drop` and `GpuBarrier`
/// nodes.  Kernel bodies are skipped.
pub fn annotate_fns(fns: &mut Vec<TypedFn>) {
    for f in fns.iter_mut() {
        annotate_body(&mut f.body);
    }
}

// ── Core analysis ─────────────────────────────────────────────────────────────

fn annotate_body(body: &mut Vec<TypedStmt>) {
    // Steps 1-2: hoist GPU subexpressions and GPU-producing returns in the outer
    // body.  Control-flow nodes are passed through unchanged — their inner bodies
    // will be hoisted in step 3 when `annotate_body` recurses into them.
    hoist_gpu_subexprs(body);
    hoist_gpu_producing_returns(body);

    // Step 3: recurse into each inner scope *before* running the outer passes.
    // This gives inner bindings their own `Drop` and `GpuBarrier` nodes, and
    // means the outer passes can treat `If`/`For`/`While` as opaque use sites.
    recurse_into_inner_scopes(body);

    // Steps 4-9: outer-scope analysis.
    let locals = collect_local_bindings(body);
    let escaping = collect_escaping(body);
    insert_assign_drops(body, &escaping);
    let last_uses = find_last_uses(body, &locals, &escaping);
    insert_drops(body, &last_uses);
    insert_barriers(body);
}

/// Step 3: call `annotate_body` on each inner scope so inner bindings get
/// their own `Drop`/`GpuBarrier` nodes.
fn recurse_into_inner_scopes(body: &mut Vec<TypedStmt>) {
    for stmt in body.iter_mut() {
        match stmt {
            TypedStmt::If { then_body, else_body, .. } => {
                annotate_body(then_body);
                if let Some(eb) = else_body { annotate_body(eb); }
            }
            TypedStmt::For { body, .. } | TypedStmt::While { body, .. } => {
                annotate_body(body);
            }
            _ => {}
        }
    }
}

// ── GPU-producing predicate ───────────────────────────────────────────────────

fn is_gpu_producing(expr: &TypedExpr) -> bool {
    matches!(&expr.kind, TypedExprKind::KernelCall { .. })
        || matches!(&expr.kind, TypedExprKind::BinOp { lhs, .. } if lhs.ty.is_tensor())
        || matches!(&expr.kind, TypedExprKind::Call { .. }
                    if expr.ty.is_tensor() && expr.placement == Some(Placement::Gpu))
}

// ── Step 1: GPU subexpression hoisting ───────────────────────────────────────

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
            TypedStmt::Assign { name, expr } => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, &mut counter);
                // D6 guard: if the RHS is still GPU-producing and yields a tensor,
                // hoist it into a temp so the old binding can be safely dropped before
                // the Assign writes the new value. This prevents use-after-free when
                // the RHS references the Assign target (e.g. `acc = acc + delta`).
                let expr = if is_gpu_producing(&expr) && expr.ty.is_tensor() {
                    let tmp_name = format!("__malus_tmp_{}", counter);
                    counter += 1;
                    let ty = expr.ty.clone();
                    let placement = expr.placement;
                    let span = expr.span;
                    hoisted.push(TypedStmt::Let { name: tmp_name.clone(), expr });
                    TypedExpr {
                        kind: TypedExprKind::Ident(tmp_name),
                        ty,
                        placement,
                        span,
                    }
                } else {
                    expr
                };
                result.extend(hoisted);
                result.push(TypedStmt::Assign { name, expr });
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
            // Control-flow nodes are passed through unchanged; their inner bodies
            // are hoisted when `annotate_body` recurses in step 3.
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

// ── Step 2: GPU-producing return hoisting ─────────────────────────────────────

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

// ── Phase 0: Assign old-value drops ──────────────────────────────────────────

/// Insert `Drop{name}` immediately before each `Assign` to a tensor binding.
/// Frees the old tensor allocation before the Assign writes the new value.
///
/// The `locals.contains(name)` guard from the original code is intentionally
/// absent (see ADR-0014 §4): the type checker guarantees only `let mut` bindings
/// appear as Assign targets (never parameters), so checking against the outer
/// `locals` set is unnecessary and would incorrectly suppress drops for outer
/// `let mut` tensors reassigned inside a loop body.
///
/// Must run after `hoist_gpu_subexprs` (which ensures the Assign RHS no longer
/// references the target via the D6 guard, making the early Drop safe).
fn insert_assign_drops(body: &mut Vec<TypedStmt>, escaping: &HashSet<String>) {
    let mut i = 0;
    while i < body.len() {
        match &body[i] {
            TypedStmt::Assign { name, expr }
                if !escaping.contains(name) && expr.ty.is_tensor() =>
            {
                let name = name.clone();
                body.insert(i, TypedStmt::Drop { name });
                i += 1; // skip past the Drop we just inserted
            }
            _ => {}
        }
        // Recurse into inner bodies so outer-scope `let mut` tensors reassigned
        // inside a loop get a Drop before each inner Assign.
        match &mut body[i] {
            TypedStmt::If { then_body, else_body, .. } => {
                insert_assign_drops(then_body, escaping);
                if let Some(eb) = else_body { insert_assign_drops(eb, escaping); }
            }
            TypedStmt::For { body: inner, .. } | TypedStmt::While { body: inner, .. } => {
                insert_assign_drops(inner, escaping);
            }
            _ => {}
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

        // Drop of an in-flight name requires a barrier before freeing.
        if let TypedStmt::Drop { name } = &body[i] {
            if pending.contains(name.as_str()) {
                body.insert(i, TypedStmt::GpuBarrier);
                pending.clear();
                i += 1; // skip barrier
            }
            i += 1;
            continue;
        }

        // Control-flow nodes: if any pending tensor is referenced anywhere inside
        // the node, emit a barrier before it and clear pending.  The inner bodies
        // already received their own barrier pass in `annotate_body` step 3.
        if matches!(&body[i], TypedStmt::If { .. } | TypedStmt::For { .. } | TypedStmt::While { .. }) {
            if !pending.is_empty() {
                let mut referenced = HashSet::new();
                collect_idents_in_stmt(&body[i], &mut referenced);
                if referenced.intersection(&pending).next().is_some() {
                    body.insert(i, TypedStmt::GpuBarrier);
                    pending.clear();
                    i += 1;
                }
            }
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
        TypedStmt::Drop { .. }
        | TypedStmt::GpuBarrier
        | TypedStmt::Assign { .. }
        | TypedStmt::If { .. }
        | TypedStmt::For { .. }
        | TypedStmt::While { .. }
        | TypedStmt::Retain { .. }
        | TypedStmt::Release { .. } => return None,
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
        TypedExprKind::Call { args, .. }
            if expr.ty.is_tensor() && expr.placement == Some(Placement::Gpu) =>
        {
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

/// Collect `let` bindings introduced directly in `body` (not in inner scopes).
/// Inner bindings are handled by the recursive `annotate_body` in step 3.
fn collect_local_bindings(body: &[TypedStmt]) -> HashSet<String> {
    body.iter().filter_map(|s| {
        if let TypedStmt::Let { name, .. } = s { Some(name.clone()) } else { None }
    }).collect()
}

/// Collect names that must not be statically dropped:
/// - tensors that appear in a `return` expr (they escape to the caller)
/// - tensor idents that are the RHS of an `Assign` (moved into the target;
///   the target's own `Drop` will free it — dropping the source would double-free)
///
/// Recurses into inner bodies so a `return` inside a branch is accounted for.
fn collect_escaping(body: &[TypedStmt]) -> HashSet<String> {
    let mut escaping = HashSet::new();
    collect_escaping_in(body, &mut escaping);
    escaping
}

fn collect_escaping_in(body: &[TypedStmt], out: &mut HashSet<String>) {
    for stmt in body {
        match stmt {
            TypedStmt::Return { expr } => collect_idents_in_expr(expr, out),
            TypedStmt::Assign { expr, .. } if expr.ty.is_tensor() => {
                collect_idents_in_expr(expr, out);
            }
            TypedStmt::If { then_body, else_body, .. } => {
                collect_escaping_in(then_body, out);
                if let Some(eb) = else_body { collect_escaping_in(eb, out); }
            }
            TypedStmt::For { body, .. } | TypedStmt::While { body, .. } => {
                collect_escaping_in(body, out);
            }
            _ => {}
        }
    }
}

fn find_last_uses(
    body: &[TypedStmt],
    locals: &HashSet<String>,
    escaping: &HashSet<String>,
) -> HashMap<String, usize> {
    let mut last: HashMap<String, usize> = HashMap::new();
    for (idx, stmt) in body.iter().enumerate() {
        let mut used = HashSet::new();
        // `collect_idents_in_stmt` recurses into control-flow bodies so outer
        // bindings referenced *inside* an If/For/While are recorded at the
        // index of the control-flow node in the outer body.  This causes `Drop`
        // to be inserted *after* the node, which is always correct.
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
        TypedStmt::Let { expr, .. }
        | TypedStmt::Assign { expr, .. }
        | TypedStmt::Return { expr } => collect_idents_in_expr(expr, out),
        TypedStmt::Expr(expr) => collect_idents_in_expr(expr, out),
        TypedStmt::Drop { .. }
        | TypedStmt::GpuBarrier
        | TypedStmt::Retain { .. }
        | TypedStmt::Release { .. } => {}
        TypedStmt::If { condition, then_body, else_body } => {
            collect_idents_in_expr(condition, out);
            for s in then_body { collect_idents_in_stmt(s, out); }
            if let Some(eb) = else_body {
                for s in eb { collect_idents_in_stmt(s, out); }
            }
        }
        TypedStmt::For { start, end, body, .. } => {
            collect_idents_in_expr(start, out);
            collect_idents_in_expr(end, out);
            for s in body { collect_idents_in_stmt(s, out); }
        }
        TypedStmt::While { condition, body } => {
            collect_idents_in_expr(condition, out);
            for s in body { collect_idents_in_stmt(s, out); }
        }
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
