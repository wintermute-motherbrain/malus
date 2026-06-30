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
use crate::ty::ResolvedTy;
use crate::typed_ir::{TypedExpr, TypedExprKind, TypedFn, TypedStmt};

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the CTMM analysis on all `fn` bodies, injecting `Drop` and `GpuBarrier`
/// nodes.  Kernel bodies are skipped.
pub fn annotate_fns(fns: &mut Vec<TypedFn>) {
    for f in fns.iter_mut() {
        let var_params: Vec<(String, ResolvedTy)> = f.params.iter()
            .filter(|p| p.ty.is_variable())
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        if var_params.is_empty() {
            annotate_body(&mut f.body);
        } else {
            annotate_body_seeded(&mut f.body, &var_params);
        }
    }
}

// ── Core analysis ─────────────────────────────────────────────────────────────

fn annotate_body(body: &mut Vec<TypedStmt>) {
    annotate_body_seeded(body, &[]);
}

/// Core CTMM pass.  `seed` carries tensor payload bindings from a surrounding
/// match arm so they are treated as arm-scoped locals (retain-on-bind, M12).
fn annotate_body_seeded(body: &mut Vec<TypedStmt>, seed: &[(String, ResolvedTy)]) {
    // Steps 1-2: hoist GPU subexpressions and GPU-producing returns in the outer
    // body.  Control-flow nodes are passed through unchanged — their inner bodies
    // will be hoisted in step 3 when `annotate_body` recurses into them.
    hoist_gpu_subexprs(body);
    insert_variable_arc_retains(body);
    hoist_gpu_producing_returns(body);

    // Step 3: recurse into each inner scope *before* running the outer passes.
    // This gives inner bindings their own `Drop` and `GpuBarrier` nodes, and
    // means the outer passes can treat `If`/`For`/`While` as opaque use sites.
    recurse_into_inner_scopes(body);

    // Step 3b (M12): retain tensor match-arm payload bindings + recurse into arms.
    annotate_match_arms(body);

    // Steps 4-9: outer-scope analysis.
    let mut locals = collect_local_bindings(body);
    let mut local_types = collect_local_types(body);
    // Seed tensor payload bindings from the enclosing arm (tensor-only; aggregate
    // bindings are arm-local borrows freed by DropEnum on the scrutinee).
    for (n, t) in seed {
        locals.insert(n.clone());
        local_types.insert(n.clone(), t.clone());
    }
    let escaping = collect_escaping(body);
    insert_assign_drops(body, &escaping);
    let mut last_uses = find_last_uses(body, &locals, &escaping);
    // Step 7b (M12): ensure every seeded (retained) tensor payload binding has a
    // matching Drop, even if it is never used locally and does not escape.
    seed_unused_floor(&mut last_uses, seed, &escaping);
    insert_drops(body, &last_uses, &local_types);
    // Step 8.5: inject Drop/DropStruct before early Returns inside nested scopes
    // for bindings whose end-of-scope Drop is after the control-flow node.
    inject_early_return_unwinds(body, &local_types);
    // Step 8.6 (M12): inject drops before Break/Continue jumps.
    inject_break_continue_unwinds(body, &local_types);
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
            TypedStmt::For { body, .. }
            | TypedStmt::While { body, .. }
            | TypedStmt::ForIn { body, .. }
            | TypedStmt::NoGrad { body } => {
                annotate_body(body);
            }
            _ => {}
        }
    }
}

/// Step 3b (M12): For each match arm, prepend `Retain` for every tensor payload
/// binding and recurse `annotate_body_seeded` into the arm body.
///
/// Aggregate (struct/enum) bindings are neither seeded nor retained: they are
/// arm-local borrows freed by `DropEnum` on the scrutinee, and sema has already
/// rejected any escape of such a binding.
///
/// Nested matches inside an arm are handled automatically because
/// `annotate_body_seeded` calls `annotate_match_arms` again on the arm body.
fn annotate_match_arms(body: &mut Vec<TypedStmt>) {
    for stmt in body.iter_mut() {
        if let TypedStmt::Match { arms, .. } = stmt {
            for arm in arms.iter_mut() {
                let tensor_binds: Vec<(String, ResolvedTy)> = arm.bindings.iter()
                    .filter(|(_, t)| t.is_tensor() || t.is_variable())
                    .cloned()
                    .collect();
                // Prepend Retain for each tensor payload in field-declaration order.
                for (name, _) in tensor_binds.iter().rev() {
                    arm.body.insert(0, TypedStmt::Retain { name: name.clone() });
                }
                annotate_body_seeded(&mut arm.body, &tensor_binds);
            }
        }
    }
}

/// Step 7b (M12): A seeded tensor payload that is neither used locally nor
/// escaping would leave its prepended `Retain` unbalanced (leak).  Force its
/// `last_uses` entry to index 0 so `insert_drops` places a `Drop` immediately
/// after the `Retain` — an inert retain/release pair that keeps the refcount
/// balanced on every path.
fn seed_unused_floor(
    last_uses: &mut HashMap<String, usize>,
    seed: &[(String, ResolvedTy)],
    escaping: &HashSet<String>,
) {
    for (name, _) in seed {
        if !last_uses.contains_key(name.as_str()) && !escaping.contains(name.as_str()) {
            last_uses.insert(name.clone(), 0);
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
            TypedStmt::Assign { target, expr } => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, &mut counter);
                // D6 guard: if the RHS is still GPU-producing and yields a tensor or
                // variable, hoist it into a temp so the old slot can be safely released
                // before the Assign writes the new value. For Index/Field targets this
                // is always needed (element release happens in codegen after RHS is fully
                // evaluated). For Ident targets the existing self-reference guard applies.
                let needs_hoist = is_gpu_producing(&expr)
                    && (expr.ty.is_tensor() || expr.ty.is_variable());
                let expr = if needs_hoist {
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
                result.push(TypedStmt::Assign { target, expr });
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
            // Recurse into operands first, then hoist any GPU-producing operand
            // into a __malus_tmp so the existing Drop machinery can free it.
            let lhs = hoist_gpu_in_expr(*lhs, hoisted, counter);
            let rhs = hoist_gpu_in_expr(*rhs, hoisted, counter);
            let lhs = hoist_if_gpu_tensor(lhs, hoisted, counter);
            let rhs = hoist_if_gpu_tensor(rhs, hoisted, counter);
            TypedExpr {
                kind: TypedExprKind::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span,
                ..expr
            }
        }
        TypedExprKind::Unary { op, operand } => {
            let operand = hoist_gpu_in_expr(*operand, hoisted, counter);
            let operand = hoist_if_gpu_tensor(operand, hoisted, counter);
            TypedExpr {
                kind: TypedExprKind::Unary { op, operand: Box::new(operand) },
                span,
                ..expr
            }
        }
        TypedExprKind::TensorLiteral { placement, dtype, elements, shape } => {
            let new_elements = elements
                .into_iter()
                .map(|e| hoist_gpu_in_expr(e, hoisted, counter))
                .collect();
            TypedExpr {
                kind: TypedExprKind::TensorLiteral { placement, dtype, elements: new_elements, shape },
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
        TypedExprKind::ArrayLiteral { elements } => {
            let new_elements = elements
                .into_iter()
                .map(|e| hoist_gpu_in_expr(e, hoisted, counter))
                .collect();
            TypedExpr {
                kind: TypedExprKind::ArrayLiteral { elements: new_elements },
                span,
                ..expr
            }
        }
        TypedExprKind::TupleInit { elements } => {
            let new_elements = elements
                .into_iter()
                .map(|e| hoist_gpu_in_expr(e, hoisted, counter))
                .collect();
            TypedExpr {
                kind: TypedExprKind::TupleInit { elements: new_elements },
                span,
                ..expr
            }
        }
        TypedExprKind::TupleIndex { base, index } => {
            TypedExpr {
                kind: TypedExprKind::TupleIndex {
                    base: Box::new(hoist_gpu_in_expr(*base, hoisted, counter)),
                    index,
                },
                span,
                ..expr
            }
        }
        _ => expr,
    }
}

/// Hoist `operand` into a `__malus_tmp` Let if it is GPU-producing and
/// tensor-typed, so the existing last-use/Drop machinery can free it.
/// Mirrors the logic in `hoist_args` but for BinOp/Unary operands.
fn hoist_if_gpu_tensor(
    operand: TypedExpr,
    hoisted: &mut Vec<TypedStmt>,
    counter: &mut u32,
) -> TypedExpr {
    if is_gpu_producing(&operand) && operand.ty.is_tensor() {
        let name = format!("__malus_tmp_{}", counter);
        *counter += 1;
        let ty = operand.ty.clone();
        let placement = operand.placement;
        let span = operand.span;
        hoisted.push(TypedStmt::Let { name: name.clone(), expr: operand });
        TypedExpr { kind: TypedExprKind::Ident(name), ty, placement, span }
    } else {
        operand
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

/// Insert a Drop/DropStruct/DropEnum before each `Assign` to an owned binding.
/// Frees the old allocation before the Assign writes the new value.
///
/// Extended in M11 to cover Struct and Enum targets in addition to tensors.
/// The `locals.contains(name)` guard is intentionally absent (see ADR-0014 §4).
///
/// Must run after `hoist_gpu_subexprs` (D6 guard ensures the RHS no longer
/// references the target, making the early Drop safe).
fn insert_assign_drops(body: &mut Vec<TypedStmt>, escaping: &HashSet<String>) {
    use crate::typed_ir::TypedAssignTarget;
    let mut i = 0;
    while i < body.len() {
        match &body[i] {
            // Ident target: drop the old binding value (same as before).
            TypedStmt::Assign { target: TypedAssignTarget::Ident(name), expr }
                if !escaping.contains(name) =>
            {
                if let Some(drop_stmt) = make_drop_stmt_for_ty(name, &expr.ty) {
                    body.insert(i, drop_stmt);
                    i += 1;
                }
            }
            // Index/Field targets: no whole-binding drop here.
            // The codegen handles the old-element-release inline (load old slot →
            // tensor_release/tensor_free → store new), after the RHS has been fully
            // evaluated into a temp by `hoist_gpu_subexprs`.
            TypedStmt::Assign { target: TypedAssignTarget::Index { .. } | TypedAssignTarget::Field { .. } | TypedAssignTarget::BufferIndex { .. }, .. } => {}
            _ => {}
        }
        // Recurse into inner bodies so outer-scope `let mut` bindings reassigned
        // inside a loop get a Drop before each inner Assign.
        match &mut body[i] {
            TypedStmt::If { then_body, else_body, .. } => {
                insert_assign_drops(then_body, escaping);
                if let Some(eb) = else_body { insert_assign_drops(eb, escaping); }
            }
            TypedStmt::For { body: inner, .. }
            | TypedStmt::While { body: inner, .. }
            | TypedStmt::ForIn { body: inner, .. }
            | TypedStmt::NoGrad { body: inner } => {
                insert_assign_drops(inner, escaping);
            }
            _ => {}
        }
        i += 1;
    }
}

/// Build the appropriate drop statement for a named binding of the given type.
/// Returns `None` for types that own no heap resources (scalar, bool, unit).
fn make_drop_stmt_for_ty(name: &str, ty: &ResolvedTy) -> Option<TypedStmt> {
    match ty {
        ResolvedTy::Variable { .. } => Some(TypedStmt::Release { name: name.to_string() }),
        ResolvedTy::Tensor { .. } => Some(TypedStmt::Drop { name: name.to_string() }),
        // Str is a leaked whole-program-lifetime buffer (ADR-0018). No drop needed.
        ResolvedTy::Str => None,
        ResolvedTy::Struct { fields, .. } => {
            let droppable_fields = droppable_struct_fields(fields);
            Some(TypedStmt::DropStruct { name: name.to_string(), droppable_fields, retained_agg_slots: vec![] })
        }
        ResolvedTy::Enum { variants, .. } => {
            let drop_variants = droppable_enum_variants(variants);
            Some(TypedStmt::DropEnum { name: name.to_string(), variants: drop_variants })
        }
        // All arrays own heap memory (the box itself), so always emit DropArray.
        // Codegen handles whether to loop and release elements (only for owned types).
        ResolvedTy::Array { elem, len } => {
            Some(TypedStmt::DropArray { name: name.to_string(), elem_ty: *elem.clone(), len: *len })
        }
        ResolvedTy::Tuple(elements) => {
            let droppable_fields: Vec<(usize, ResolvedTy)> = elements.iter()
                .enumerate()
                .filter_map(|(i, ty)| {
                    if ty.is_tensor() || ty.is_variable() {
                        Some((i, ty.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            Some(TypedStmt::DropTuple { name: name.to_string(), droppable_fields })
        }
        ResolvedTy::Buffer { .. } => Some(TypedStmt::DropBuffer { name: name.to_string() }),
        _ => None,
    }
}

fn droppable_struct_fields(fields: &[(String, ResolvedTy)]) -> Vec<(usize, ResolvedTy)> {
    fields.iter()
        .enumerate()
        .filter_map(|(i, (_, ty))| {
            if ty.is_tensor() || ty.is_variable() || ty.is_struct() || ty.is_enum() {
                Some((i, ty.clone()))
            } else {
                None
            }
        })
        .collect()
}

fn droppable_enum_variants(
    variants: &[(String, Vec<(String, ResolvedTy)>)],
) -> Vec<(u32, Vec<(usize, ResolvedTy)>, Vec<usize>)> {
    variants.iter()
        .enumerate()
        .map(|(tag, (_, fields))| {
            let droppable = fields.iter()
                .enumerate()
                .filter_map(|(i, (_, ty))| {
                    if ty.is_tensor() || ty.is_variable() || ty.is_struct() || ty.is_enum() {
                        Some((i, ty.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            (tag as u32, droppable, vec![])
        })
        .collect()
}

// ── Phase 1: Drop insertion ───────────────────────────────────────────────────

fn insert_drops(
    body: &mut Vec<TypedStmt>,
    last_uses: &HashMap<String, usize>,
    local_types: &HashMap<String, ResolvedTy>,
) {
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
        let mut offset = 0;
        for name in &names {
            let ty = match local_types.get(name.as_str()) {
                Some(t) => t,
                None => continue,
            };
            let stmt = match make_drop_stmt_for_ty(name, ty) {
                Some(s) => s,
                None => continue,
            };
            body.insert(insert_pos + offset, stmt);
            offset += 1;
        }
    }
}

// ── Phase 1.5: Early-return unwind ───────────────────────────────────────────
//
// After `insert_drops` places Drop nodes in the outer body, scan for Returns
// nested inside If/For/While bodies.  Any outer binding whose Drop is placed
// AFTER a control-flow node containing an early Return would be leaked.  Inject
// Drop/DropStruct before the Return so the binding is freed on every exit path.

fn inject_early_return_unwinds(
    body: &mut Vec<TypedStmt>,
    local_types: &HashMap<String, ResolvedTy>,
) {
    // Map name → position of its Drop/DropStruct/DropEnum/DropArray in the outer body.
    let mut drop_positions: HashMap<String, usize> = HashMap::new();
    for (i, stmt) in body.iter().enumerate() {
        match stmt {
            TypedStmt::Drop { name }
            | TypedStmt::DropStruct { name, .. }
            | TypedStmt::DropEnum { name, .. }
            | TypedStmt::DropArray { name, .. }
            | TypedStmt::DropTuple { name, .. }
            | TypedStmt::DropBuffer { name }
            | TypedStmt::Release { name } => {
                drop_positions.insert(name.clone(), i);
            }
            _ => {}
        }
    }
    if drop_positions.is_empty() {
        return;
    }

    // For each control-flow node at outer position i, any binding whose Drop
    // is after i is "live" there and needs unwinding at any early Return inside.
    for i in 0..body.len() {
        let live_here: HashSet<String> = drop_positions.iter()
            .filter(|(_, &pos)| pos > i)
            .map(|(name, _)| name.clone())
            .collect();
        if live_here.is_empty() {
            continue;
        }
        // We can't hold a mutable borrow to body[i] via index while also
        // having live_here referencing drop_positions — use an index match.
        let is_cf = matches!(&body[i],
            TypedStmt::If { .. } | TypedStmt::For { .. }
            | TypedStmt::While { .. } | TypedStmt::ForIn { .. }
            | TypedStmt::Match { .. } | TypedStmt::NoGrad { .. });
        if !is_cf {
            continue;
        }
        match &mut body[i] {
            TypedStmt::If { then_body, else_body, .. } => {
                inject_unwind_in_body(then_body, &live_here, local_types);
                if let Some(eb) = else_body {
                    inject_unwind_in_body(eb, &live_here, local_types);
                }
            }
            TypedStmt::For { body: inner, .. }
            | TypedStmt::While { body: inner, .. }
            | TypedStmt::ForIn { body: inner, .. }
            | TypedStmt::NoGrad { body: inner } => {
                inject_unwind_in_body(inner, &live_here, local_types);
            }
            TypedStmt::Match { arms, .. } => {
                for arm in arms.iter_mut() {
                    inject_unwind_in_body(&mut arm.body, &live_here, local_types);
                }
            }
            _ => {}
        }
    }
}

fn inject_unwind_in_body(
    body: &mut Vec<TypedStmt>,
    live_outer: &HashSet<String>,
    local_types: &HashMap<String, ResolvedTy>,
) {
    let mut i = 0;
    while i < body.len() {
        match &body[i] {
            TypedStmt::Return { expr } => {
                let mut keep = HashSet::new();
                collect_idents_in_expr(expr, &mut keep);
                let mut to_drop: Vec<String> = live_outer
                    .iter()
                    .filter(|n| !keep.contains(*n) && local_types.contains_key(n.as_str()))
                    .cloned()
                    .collect();
                to_drop.sort();
                for name in &to_drop {
                    if let Some(stmt) = make_unwind_drop(name, local_types) {
                        body.insert(i, stmt);
                        i += 1;
                    }
                }
                // Skip past the Return (no inner scopes inside it).
            }
            TypedStmt::If { .. } | TypedStmt::For { .. }
            | TypedStmt::While { .. } | TypedStmt::ForIn { .. }
            | TypedStmt::Match { .. } | TypedStmt::NoGrad { .. } => {
                // Recurse into nested control flow.
                match &mut body[i] {
                    TypedStmt::If { then_body, else_body, .. } => {
                        inject_unwind_in_body(then_body, live_outer, local_types);
                        if let Some(eb) = else_body {
                            inject_unwind_in_body(eb, live_outer, local_types);
                        }
                    }
                    TypedStmt::For { body: inner, .. }
                    | TypedStmt::While { body: inner, .. }
                    | TypedStmt::ForIn { body: inner, .. }
                    | TypedStmt::NoGrad { body: inner } => {
                        inject_unwind_in_body(inner, live_outer, local_types);
                    }
                    TypedStmt::Match { arms, .. } => {
                        for arm in arms.iter_mut() {
                            inject_unwind_in_body(&mut arm.body, live_outer, local_types);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        i += 1;
    }
}

fn make_unwind_drop(name: &str, local_types: &HashMap<String, ResolvedTy>) -> Option<TypedStmt> {
    local_types.get(name).and_then(|ty| make_drop_stmt_for_ty(name, ty))
}

// ── Phase 1.6 (M12): Break/Continue unwind ────────────────────────────────────
//
// `break`/`continue` jump out of the loop body, skipping any `Drop` nodes that
// would fire at the normal end of the iteration.  For each jump we inject drops
// for all loop-body-level locals whose `Drop` sits after the jump point (i.e. is
// live there).
//
// This pass runs at every `annotate_body_seeded` level so multi-level scopes
// compose: a binding declared inside an `if` inside a loop gets its drop injected
// by the if-branch level; the loop-body level separately drops loop-body-top-level
// locals (descending through the if without descending into nested loops).
//
// Descends into: If / Match (same live_here set).
// Does NOT descend into: For / While / ForIn — a Break/Continue inside a nested
// loop belongs to that loop and was already handled when its body was annotated.

fn inject_break_continue_unwinds(body: &mut Vec<TypedStmt>, local_types: &HashMap<String, ResolvedTy>) {
    // Map name → position of its Drop/DropStruct/DropEnum/DropArray in this body.
    let mut drop_positions: HashMap<String, usize> = HashMap::new();
    for (i, stmt) in body.iter().enumerate() {
        match stmt {
            TypedStmt::Drop { name }
            | TypedStmt::DropStruct { name, .. }
            | TypedStmt::DropEnum { name, .. }
            | TypedStmt::DropArray { name, .. }
            | TypedStmt::DropTuple { name, .. }
            | TypedStmt::DropBuffer { name }
            | TypedStmt::Release { name } => {
                drop_positions.insert(name.clone(), i);
            }
            _ => {}
        }
    }
    if drop_positions.is_empty() {
        return;
    }

    let mut i = 0;
    while i < body.len() {
        // Compute the set of locals whose Drop is after position i (live at i).
        let live_here: HashSet<String> = drop_positions.iter()
            .filter(|(_, &pos)| pos > i)
            .map(|(name, _)| name.clone())
            .collect();

        if live_here.is_empty() {
            i += 1;
            continue;
        }

        match &body[i] {
            TypedStmt::Break | TypedStmt::Continue => {
                // Insert drops for all live locals immediately before the jump.
                let mut to_drop: Vec<String> = live_here.into_iter()
                    .filter(|n| local_types.contains_key(n.as_str()))
                    .collect();
                to_drop.sort();
                let mut inserted = 0;
                for name in &to_drop {
                    if let Some(stmt) = make_unwind_drop(name, local_types) {
                        body.insert(i + inserted, stmt);
                        inserted += 1;
                    }
                }
                i += inserted + 1; // skip past the inserted drops + the jump itself
                continue;
            }
            TypedStmt::If { .. } | TypedStmt::Match { .. } | TypedStmt::NoGrad { .. } => {
                // Descend; do NOT descend into nested loops.
                inject_bc_unwind_in_body_mut(body, i, &live_here, local_types);
            }
            _ => {}
        }
        i += 1;
    }
}

/// Recurse into If/Match arms looking for Break/Continue, injecting drops of
/// `live_outer` before each one.  Stops at nested For/While/ForIn.
fn inject_bc_unwind_in_body(
    inner: &mut Vec<TypedStmt>,
    live_outer: &HashSet<String>,
    local_types: &HashMap<String, ResolvedTy>,
) {
    let mut i = 0;
    while i < inner.len() {
        match &inner[i] {
            TypedStmt::Break | TypedStmt::Continue => {
                let mut to_drop: Vec<String> = live_outer.iter()
                    .filter(|n| local_types.contains_key(n.as_str()))
                    .cloned()
                    .collect();
                to_drop.sort();
                let mut inserted = 0;
                for name in &to_drop {
                    if let Some(stmt) = make_unwind_drop(name, local_types) {
                        inner.insert(i + inserted, stmt);
                        inserted += 1;
                    }
                }
                i += inserted + 1;
                continue;
            }
            TypedStmt::If { .. } | TypedStmt::Match { .. } | TypedStmt::NoGrad { .. } => {
                match &mut inner[i] {
                    TypedStmt::If { then_body, else_body, .. } => {
                        inject_bc_unwind_in_body(then_body, live_outer, local_types);
                        if let Some(eb) = else_body {
                            inject_bc_unwind_in_body(eb, live_outer, local_types);
                        }
                    }
                    TypedStmt::Match { arms, .. } => {
                        for arm in arms.iter_mut() {
                            inject_bc_unwind_in_body(&mut arm.body, live_outer, local_types);
                        }
                    }
                    TypedStmt::NoGrad { body: inner_body } => {
                        inject_bc_unwind_in_body(inner_body, live_outer, local_types);
                    }
                    _ => {}
                }
            }
            // Do NOT descend into nested loops — their Break/Continue is theirs.
            _ => {}
        }
        i += 1;
    }
}

/// Helper: perform the mutable descent for inject_break_continue_unwinds without
/// holding a simultaneous borrow on body through the live_here set.
fn inject_bc_unwind_in_body_mut(
    body: &mut Vec<TypedStmt>,
    idx: usize,
    live_here: &HashSet<String>,
    local_types: &HashMap<String, ResolvedTy>,
) {
    match &mut body[idx] {
        TypedStmt::If { then_body, else_body, .. } => {
            inject_bc_unwind_in_body(then_body, live_here, local_types);
            if let Some(eb) = else_body {
                inject_bc_unwind_in_body(eb, live_here, local_types);
            }
        }
        TypedStmt::Match { arms, .. } => {
            for arm in arms.iter_mut() {
                inject_bc_unwind_in_body(&mut arm.body, live_here, local_types);
            }
        }
        TypedStmt::NoGrad { body: inner } => {
            inject_bc_unwind_in_body(inner, live_here, local_types);
        }
        _ => {}
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
        // M12: also guard DropEnum/DropStruct/DropArray — their payload tensor
        // fields may still be in-flight when the box is released.
        let drop_name = match &body[i] {
            TypedStmt::Drop { name }
            | TypedStmt::DropStruct { name, .. }
            | TypedStmt::DropEnum { name, .. }
            | TypedStmt::DropArray { name, .. }
            | TypedStmt::DropTuple { name, .. }
            | TypedStmt::DropBuffer { name }
            | TypedStmt::Release { name } => Some(name.as_str()),
            _ => None,
        };
        if let Some(name) = drop_name {
            if pending.contains(name) {
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
        if matches!(&body[i], TypedStmt::If { .. } | TypedStmt::For { .. }
                | TypedStmt::While { .. } | TypedStmt::ForIn { .. }
                | TypedStmt::NoGrad { .. }) {
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
        | TypedStmt::DropStruct { .. }
        | TypedStmt::DropEnum { .. }
        | TypedStmt::DropArray { .. }
        | TypedStmt::DropTuple { .. }
        | TypedStmt::DropBuffer { .. }
        | TypedStmt::GpuBarrier
        | TypedStmt::Assign { .. }
        | TypedStmt::If { .. }
        | TypedStmt::For { .. }
        | TypedStmt::While { .. }
        | TypedStmt::Match { .. }
        | TypedStmt::Retain { .. }
        | TypedStmt::Release { .. }
        | TypedStmt::RetainAgg { .. }
        | TypedStmt::ReleaseAgg { .. }
        | TypedStmt::ForIn { .. }
        | TypedStmt::LetTuple { .. }
        | TypedStmt::Break
        | TypedStmt::Continue
        | TypedStmt::NoGrad { .. } => return None,
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
    let mut out = HashSet::new();
    for s in body {
        match s {
            TypedStmt::Let { name, .. } => { out.insert(name.clone()); }
            TypedStmt::LetTuple { names, .. } => {
                for (n, _) in names { out.insert(n.clone()); }
            }
            _ => {}
        }
    }
    out
}

/// Collect the resolved type of each `let` binding in `body` (outer scope only).
fn collect_local_types(body: &[TypedStmt]) -> HashMap<String, ResolvedTy> {
    let mut out = HashMap::new();
    for s in body {
        match s {
            TypedStmt::Let { name, expr } => { out.insert(name.clone(), expr.ty.clone()); }
            TypedStmt::LetTuple { names, .. } => {
                for (n, ty) in names { out.insert(n.clone(), ty.clone()); }
            }
            _ => {}
        }
    }
    out
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
            TypedStmt::Assign { expr, .. } if expr.ty.is_tensor() || expr.ty.is_variable() => {
                collect_idents_in_expr(expr, out);
            }
            TypedStmt::If { then_body, else_body, .. } => {
                collect_escaping_in(then_body, out);
                if let Some(eb) = else_body { collect_escaping_in(eb, out); }
            }
            TypedStmt::For { body, .. }
            | TypedStmt::While { body, .. }
            | TypedStmt::ForIn { body, .. }
            | TypedStmt::NoGrad { body } => {
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
        | TypedStmt::DropStruct { .. }
        | TypedStmt::DropEnum { .. }
        | TypedStmt::DropArray { .. }
        | TypedStmt::DropTuple { .. }
        | TypedStmt::DropBuffer { .. }
        | TypedStmt::GpuBarrier
        | TypedStmt::Retain { .. }
        | TypedStmt::Release { .. }
        | TypedStmt::RetainAgg { .. }
        | TypedStmt::ReleaseAgg { .. }
        | TypedStmt::Break
        | TypedStmt::Continue => {}
        TypedStmt::LetTuple { expr, .. } => collect_idents_in_expr(expr, out),
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
        TypedStmt::ForIn { iter, body, .. } => {
            collect_idents_in_expr(iter, out);
            for s in body { collect_idents_in_stmt(s, out); }
        }
        TypedStmt::Match { scrutinee, arms } => {
            collect_idents_in_expr(scrutinee, out);
            for arm in arms {
                for s in &arm.body { collect_idents_in_stmt(s, out); }
            }
        }
        TypedStmt::NoGrad { body } => {
            for s in body { collect_idents_in_stmt(s, out); }
        }
    }
}

/// Insert `Retain { name }` immediately before any `Let`/`Assign`/`Expr`/`Return`
/// statement whose top-level expression is a `Call` with Variable-typed Ident arguments.
/// This ensures every Variable passed to a function has a matching Retain+Release pair:
/// the caller retains before the call, the callee releases at last use (via CTMM seeding),
/// and the caller releases its own reference after the call.
fn insert_variable_arc_retains(body: &mut Vec<TypedStmt>) {
    let mut i = 0;
    while i < body.len() {
        let retain_names: Vec<String> = match &body[i] {
            TypedStmt::Let { expr, .. }
            | TypedStmt::LetTuple { expr, .. } => {
                variable_arc_retains_for_expr(expr)
            }
            TypedStmt::Assign { expr, .. } => {
                variable_arc_retains_for_expr(expr)
            }
            TypedStmt::Expr(expr) | TypedStmt::Return { expr } => {
                // For non-binding positions only emit retains for call args.
                if let TypedExprKind::Call { args, .. } = &expr.kind {
                    args.iter()
                        .filter(|a| a.ty.is_variable())
                        .filter_map(|a| {
                            if let TypedExprKind::Ident(n) = &a.kind { Some(n.clone()) } else { None }
                        })
                        .collect()
                } else {
                    vec![]
                }
            }
            _ => vec![],
        };
        for name in retain_names.into_iter().rev() {
            body.insert(i, TypedStmt::Retain { name });
            i += 1;
        }
        i += 1;
    }
}

fn variable_arc_retains_for_expr(expr: &TypedExpr) -> Vec<String> {
    match &expr.kind {
        // let b = a — Variable alias: retain a so b is a genuine co-owner.
        TypedExprKind::Ident(name) if expr.ty.is_variable() => vec![name.clone()],
        // let t = v.data — .data let-bind: retain v's handle so t is a genuine Tensor owner.
        TypedExprKind::FieldAccess { base, field }
            if field == "data" && base.ty.is_variable() =>
        {
            if let TypedExprKind::Ident(n) = &base.kind {
                vec![n.clone()]
            } else {
                vec![]
            }
        }
        // fn call: retain any Variable ident args (caller-retains ARC).
        TypedExprKind::Call { args, .. } => args
            .iter()
            .filter(|a| a.ty.is_variable())
            .filter_map(|a| {
                if let TypedExprKind::Ident(n) = &a.kind { Some(n.clone()) } else { None }
            })
            .collect(),
        // Array literal: retain Variable ident elements so the slot is a genuine
        // co-owner alongside any binding that still references the same handle.
        TypedExprKind::ArrayLiteral { elements } => elements
            .iter()
            .filter(|e| e.ty.is_variable())
            .filter_map(|e| {
                if let TypedExprKind::Ident(n) = &e.kind { Some(n.clone()) } else { None }
            })
            .collect(),
        // Struct literal: retain Variable ident fields so the struct slot is a genuine
        // co-owner alongside any binding that still references the same handle.
        // (variable() calls self-retain, so only Ident sources need an extra retain here.)
        TypedExprKind::StructInit { fields, .. } => fields
            .iter()
            .filter(|f| f.ty.is_variable())
            .filter_map(|f| {
                if let TypedExprKind::Ident(n) = &f.kind { Some(n.clone()) } else { None }
            })
            .collect(),
        _ => vec![],
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
        TypedExprKind::StructInit { fields, .. } => {
            for f in fields { collect_idents_in_expr(f, out); }
        }
        TypedExprKind::EnumInit { payload, .. } => {
            for p in payload { collect_idents_in_expr(p, out); }
        }
        TypedExprKind::ArrayLiteral { elements } => {
            for e in elements { collect_idents_in_expr(e, out); }
        }
        TypedExprKind::TupleInit { elements } => {
            for e in elements { collect_idents_in_expr(e, out); }
        }
        TypedExprKind::TupleIndex { base, .. } => collect_idents_in_expr(base, out),
        TypedExprKind::Lit(_) => {}
    }
}
