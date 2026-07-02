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

use crate::ty::ResolvedTy;
use crate::typed_ir::{TypedExpr, TypedExprKind, TypedProgram, TypedStmt};
use malus_syntax::ast::Placement;
use std::collections::{HashMap, HashSet};

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the CTMM analysis on all `fn` bodies, injecting `Drop` and `GpuBarrier`
/// nodes.  Kernel bodies are skipped.
///
/// M29 (ADR-0026, D2): every tensor param is a zero-cost borrow, uniformly,
/// regardless of grad-tracking. Pre-M29, a grad-tracked tensor param was
/// seeded here so it would draw its own `Release` at last use, balancing a
/// caller-side `Retain` inserted by `insert_variable_arc_retains` before the
/// call. Both sides of that pair are gone: the callee never independently
/// owns a reference to a param (so it never drops one), and the caller never
/// retains an argument before a call — the caller's own binding is still
/// dropped at its own true last use, which the call itself extends past this
/// synchronous callee's entire execution. The one case a plain borrow can't
/// cover — the callee returning the exact handle it borrowed, handing the
/// caller a reference it must independently own — needs a `Retain` on that
/// return path; `insert_param_return_retains` below covers it before the
/// general `annotate_body` walk runs (a returned *alias* of a param, e.g.
/// `let y = x; return y`, is still covered by the pre-M29
/// `insert_variable_arc_retains` alias-retain path, untouched by this change).
pub fn annotate_fns(program: &mut TypedProgram) {
    let _ = &program.fn_param_grad; // still consumed by grad_inference's own callers, not CTMM
    for f in program.fns.iter_mut() {
        let param_names: HashSet<String> = f
            .params
            .iter()
            .filter(|p| p.ty.is_tensor())
            .map(|p| p.name.clone())
            .collect();
        if !param_names.is_empty() {
            insert_param_return_retains(&mut f.body, &param_names);
        }
        // One hoist-temp counter per *function*, threaded through every nested
        // scope. A per-body counter would mint the same `__malus_tmp_0` in the
        // outer body and inside a loop; the outer last-use scan (which recurses
        // into control-flow nodes to treat them as opaque use sites) then sees
        // the inner temp's uses under the outer temp's name and emits a stray
        // Drop after the loop — codegen's flat per-fn variable map resolves it
        // to the *inner* temp, double-releasing the final iteration's value
        // (the M32-addendum loop-carried-reassignment over-release).
        let mut tmp_counter = 0u32;
        annotate_body(&mut f.body, &mut tmp_counter);
        // M29 (ADR-0026, D3): remove provably-redundant Retain+Drop/Release
        // pairs left by the (still-correct, still-conservative) machinery
        // above — a post-process cleanup, not a replacement analysis. See
        // `borrow_inference::demote_safe_borrows`.
        crate::borrow_inference::demote_safe_borrows(&mut f.body, &param_names);
    }
}

/// M29 (ADR-0026, D2): a tensor param carries no owned reference (uniform
/// borrow ABI), so directly returning it -- `return x` or `return x.data` --
/// must retain first: the caller's fresh result binding needs its own
/// reference distinct from whatever the original owner (in some ancestor
/// caller) still holds and will independently drop. Recurses into nested
/// control flow so a `return` inside an `if`/`for`/`while`/`match` is covered
/// too -- mirrors `collect_escaping_in`'s recursion.
fn insert_param_return_retains(body: &mut Vec<TypedStmt>, param_names: &HashSet<String>) {
    let mut i = 0;
    while i < body.len() {
        let retain_name: Option<String> = match &body[i] {
            TypedStmt::Return { expr } => match &expr.kind {
                TypedExprKind::Ident(name) if param_names.contains(name.as_str()) => {
                    Some(name.clone())
                }
                TypedExprKind::FieldAccess { base, field } if field == "data" => {
                    match &base.kind {
                        TypedExprKind::Ident(name) if param_names.contains(name.as_str()) => {
                            Some(name.clone())
                        }
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(name) = retain_name {
            body.insert(i, TypedStmt::Retain { name });
            i += 1;
        }
        match &mut body[i] {
            TypedStmt::If {
                then_body,
                else_body,
                ..
            } => {
                insert_param_return_retains(then_body, param_names);
                if let Some(eb) = else_body {
                    insert_param_return_retains(eb, param_names);
                }
            }
            TypedStmt::For { body: inner, .. }
            | TypedStmt::While { body: inner, .. }
            | TypedStmt::ForIn { body: inner, .. }
            | TypedStmt::NoGrad { body: inner } => {
                insert_param_return_retains(inner, param_names);
            }
            TypedStmt::Match { arms, .. } => {
                for arm in arms.iter_mut() {
                    insert_param_return_retains(&mut arm.body, param_names);
                }
            }
            _ => {}
        }
        i += 1;
    }
}

// ── Core analysis ─────────────────────────────────────────────────────────────

fn annotate_body(body: &mut Vec<TypedStmt>, tmp_counter: &mut u32) {
    annotate_body_seeded(body, &[], tmp_counter);
}

/// Core CTMM pass.  `seed` carries tensor payload bindings from a surrounding
/// match arm so they are treated as arm-scoped locals (retain-on-bind, M12).
/// `tmp_counter` is function-unique (see `annotate_fns`) so hoist temps never
/// collide across nesting levels.
fn annotate_body_seeded(
    body: &mut Vec<TypedStmt>,
    seed: &[(String, ResolvedTy)],
    tmp_counter: &mut u32,
) {
    // Steps 1-2: hoist GPU subexpressions and GPU-producing returns in the outer
    // body.  Control-flow nodes are passed through unchanged — their inner bodies
    // will be hoisted in step 3 when `annotate_body` recurses into them.
    hoist_gpu_subexprs(body, tmp_counter);
    insert_container_read_retains(body, tmp_counter);
    insert_variable_arc_retains(body);
    insert_list_retains(body);
    hoist_gpu_producing_returns(body, tmp_counter);

    // Step 3: recurse into each inner scope *before* running the outer passes.
    // This gives inner bindings their own `Drop` and `GpuBarrier` nodes, and
    // means the outer passes can treat `If`/`For`/`While` as opaque use sites.
    recurse_into_inner_scopes(body, tmp_counter);

    // Step 3b (M12): retain tensor match-arm payload bindings + recurse into arms.
    annotate_match_arms(body, tmp_counter);

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
    // M31 (ADR-0035): static barrier insertion is demoted to an opt-in
    // optimization lever. Read safety is the runtime's per-buffer pending
    // tracking + auto-flush; drops of pending tensors are memory-safe because
    // Metal command buffers retain referenced resources.
    if static_barriers_enabled() {
        insert_barriers(body);
    }
}

// Thread-local (not a process global) so parallel test threads can hold
// different settings; `check_with_options` sets it before annotate_fns runs
// on the same thread.
thread_local! {
    static STATIC_BARRIERS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_static_barriers(on: bool) {
    STATIC_BARRIERS.with(|c| c.set(on));
}

fn static_barriers_enabled() -> bool {
    STATIC_BARRIERS.with(|c| c.get())
}

/// Step 3: call `annotate_body` on each inner scope so inner bindings get
/// their own `Drop`/`GpuBarrier` nodes.
fn recurse_into_inner_scopes(body: &mut Vec<TypedStmt>, tmp_counter: &mut u32) {
    for stmt in body.iter_mut() {
        match stmt {
            TypedStmt::If {
                then_body,
                else_body,
                ..
            } => {
                annotate_body(then_body, tmp_counter);
                if let Some(eb) = else_body {
                    annotate_body(eb, tmp_counter);
                }
            }
            TypedStmt::For { body, .. }
            | TypedStmt::While { body, .. }
            | TypedStmt::ForIn { body, .. }
            | TypedStmt::NoGrad { body } => {
                annotate_body(body, tmp_counter);
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
fn annotate_match_arms(body: &mut Vec<TypedStmt>, tmp_counter: &mut u32) {
    for stmt in body.iter_mut() {
        if let TypedStmt::Match { arms, .. } = stmt {
            for arm in arms.iter_mut() {
                let tensor_binds: Vec<(String, ResolvedTy)> = arm
                    .bindings
                    .iter()
                    .filter(|(_, t)| t.is_tensor())
                    .cloned()
                    .collect();
                // Prepend Retain for each tensor payload in field-declaration order.
                for (name, _) in tensor_binds.iter().rev() {
                    arm.body.insert(0, TypedStmt::Retain { name: name.clone() });
                }
                annotate_body_seeded(&mut arm.body, &tensor_binds, tmp_counter);
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

fn hoist_gpu_subexprs(body: &mut Vec<TypedStmt>, counter: &mut u32) {
    let mut result: Vec<TypedStmt> = Vec::with_capacity(body.len());
    for stmt in body.drain(..) {
        match stmt {
            TypedStmt::Let { name, expr } => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, counter);
                result.extend(hoisted);
                result.push(TypedStmt::Let { name, expr });
            }
            TypedStmt::Assign { target, expr } => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, counter);
                // D6 guard: if the RHS is still GPU-producing and yields a tensor or
                // variable, hoist it into a temp so the old slot can be safely released
                // before the Assign writes the new value. For Index/Field targets this
                // is always needed (element release happens in codegen after RHS is fully
                // evaluated). For Ident targets the existing self-reference guard applies.
                let needs_hoist = is_gpu_producing(&expr) && expr.ty.is_tensor();
                let expr = if needs_hoist {
                    let tmp_name = format!("__malus_tmp_{}", counter);
                    *counter += 1;
                    let ty = expr.ty.clone();
                    let placement = expr.placement;
                    let span = expr.span;
                    let grad_tracked = expr.grad_tracked;
                    hoisted.push(TypedStmt::Let {
                        name: tmp_name.clone(),
                        expr,
                    });
                    TypedExpr {
                        kind: TypedExprKind::Ident(tmp_name),
                        ty,
                        placement,
                        span,
                        grad_tracked,
                    }
                } else {
                    expr
                };
                result.extend(hoisted);
                result.push(TypedStmt::Assign { target, expr });
            }
            TypedStmt::Return { expr } => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, counter);
                result.extend(hoisted);
                result.push(TypedStmt::Return { expr });
            }
            TypedStmt::Expr(expr) => {
                let mut hoisted = Vec::new();
                let expr = hoist_gpu_in_expr(expr, &mut hoisted, counter);
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
                kind: TypedExprKind::Call {
                    callee,
                    args: new_args,
                },
                span,
                ..expr
            }
        }
        TypedExprKind::KernelCall {
            callee,
            args,
            in_flight,
        } => {
            let new_args = hoist_args(args, hoisted, counter);
            TypedExpr {
                kind: TypedExprKind::KernelCall {
                    callee,
                    args: new_args,
                    in_flight,
                },
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
                kind: TypedExprKind::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
                ..expr
            }
        }
        TypedExprKind::Unary { op, operand } => {
            let operand = hoist_gpu_in_expr(*operand, hoisted, counter);
            let operand = hoist_if_gpu_tensor(operand, hoisted, counter);
            TypedExpr {
                kind: TypedExprKind::Unary {
                    op,
                    operand: Box::new(operand),
                },
                span,
                ..expr
            }
        }
        TypedExprKind::TensorLiteral {
            placement,
            dtype,
            elements,
            shape,
        } => {
            let new_elements = elements
                .into_iter()
                .map(|e| hoist_gpu_in_expr(e, hoisted, counter))
                .collect();
            TypedExpr {
                kind: TypedExprKind::TensorLiteral {
                    placement,
                    dtype,
                    elements: new_elements,
                    shape,
                },
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
                kind: TypedExprKind::Index {
                    base: new_base,
                    indices: new_indices,
                },
                span,
                ..expr
            }
        }
        TypedExprKind::FieldAccess { base, field } => TypedExpr {
            kind: TypedExprKind::FieldAccess {
                base: Box::new(hoist_gpu_in_expr(*base, hoisted, counter)),
                field,
            },
            span,
            ..expr
        },
        TypedExprKind::ArrayLiteral { elements } => {
            let new_elements = elements
                .into_iter()
                .map(|e| hoist_gpu_in_expr(e, hoisted, counter))
                .collect();
            TypedExpr {
                kind: TypedExprKind::ArrayLiteral {
                    elements: new_elements,
                },
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
                kind: TypedExprKind::TupleInit {
                    elements: new_elements,
                },
                span,
                ..expr
            }
        }
        TypedExprKind::TupleIndex { base, index } => TypedExpr {
            kind: TypedExprKind::TupleIndex {
                base: Box::new(hoist_gpu_in_expr(*base, hoisted, counter)),
                index,
            },
            span,
            ..expr
        },
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
        let grad_tracked = operand.grad_tracked;
        hoisted.push(TypedStmt::Let {
            name: name.clone(),
            expr: operand,
        });
        TypedExpr {
            kind: TypedExprKind::Ident(name),
            ty,
            placement,
            span,
            grad_tracked,
        }
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
            let grad_tracked = arg.grad_tracked;
            hoisted.push(TypedStmt::Let {
                name: name.clone(),
                expr: arg,
            });
            new_args.push(TypedExpr {
                kind: TypedExprKind::Ident(name),
                ty,
                placement,
                span,
                grad_tracked,
            });
        } else {
            new_args.push(arg);
        }
    }
    new_args
}

// ── Step 2: GPU-producing return hoisting ─────────────────────────────────────

fn hoist_gpu_producing_returns(body: &mut Vec<TypedStmt>, counter: &mut u32) {
    let mut i = 0;
    while i < body.len() {
        if let TypedStmt::Return { expr } = &body[i] {
            if is_gpu_producing(expr) && expr.ty.is_tensor() {
                let expr = if let TypedStmt::Return { expr } = body.remove(i) {
                    expr
                } else {
                    unreachable!()
                };
                let name = format!("__malus_ret_{}", counter);
                *counter += 1;
                let ret_ty = expr.ty.clone();
                let span = expr.span;
                let grad_tracked = expr.grad_tracked;
                body.insert(
                    i,
                    TypedStmt::Let {
                        name: name.clone(),
                        expr,
                    },
                );
                body.insert(
                    i + 1,
                    TypedStmt::Return {
                        expr: TypedExpr {
                            kind: TypedExprKind::Ident(name.clone()),
                            ty: ret_ty,
                            placement: None,
                            span,
                            grad_tracked,
                        },
                    },
                );
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
/// The `locals.contains(name)` guard is intentionally absent (see ADR-0014 §4)
/// — this is what lets a single-level scan catch a reassignment of an
/// outer-scope `let mut` (like `cur` in `__sum_to_shape_bwd`) even though
/// `cur` isn't `body`'s own local.
///
/// Outer body only — does NOT recurse into If/For/While/NoGrad bodies.
///
/// M29 bugfix (ADR-0026 investigation): this function used to also recurse
/// into every nested control-flow body itself. That was always redundant
/// with `annotate_body_seeded`'s own `recurse_into_inner_scopes` step (which
/// runs *earlier*, step 3, and gives every nested scope a full, independent
/// `annotate_body_seeded` call — including that scope's own
/// `insert_assign_drops` over its own body). The redundant self-recursion
/// therefore ran `insert_assign_drops` a second time over an already-fully-
/// processed inner body, inserting a *second* Drop before every reassignment
/// found there — an unconditional double-free the moment that binding's
/// refcount reached zero from the first (correct) Drop. Confirmed with real
/// Metal handles on the simplest possible repro (`let mut acc = ...; for i in
/// range(3): acc = add(acc, delta)` — exactly `test_let_mut_in_loop`'s
/// source) once the runtime's new over-release guard (this milestone) made
/// the fault deterministic instead of a silent refcount underflow/leak.
/// Pre-existing since `let mut` + reassignment shipped in V1; unrelated to
/// the D1-D7 design change and present with zero M29 sema edits applied.
///
/// Must run after `hoist_gpu_subexprs` (D6 guard ensures the RHS no longer
/// references the target, making the early Drop safe).
fn insert_assign_drops(
    body: &mut Vec<TypedStmt>,
    escaping: &HashSet<String>,
) {
    use crate::typed_ir::TypedAssignTarget;
    let mut i = 0;
    while i < body.len() {
        match &body[i] {
            // Ident target: drop the old binding value (same as before).
            TypedStmt::Assign {
                target: TypedAssignTarget::Ident(name),
                expr,
            } if !escaping.contains(name) => {
                if let Some(drop_stmt) =
                    make_drop_stmt_for_ty(name, &expr.ty)
                {
                    body.insert(i, drop_stmt);
                    i += 1;
                }
            }
            // Index/Field targets: no whole-binding drop here.
            // The codegen handles the old-element-release inline (load old slot →
            // tensor_release/tensor_free → store new), after the RHS has been fully
            // evaluated into a temp by `hoist_gpu_subexprs`.
            TypedStmt::Assign {
                target:
                    TypedAssignTarget::Index { .. }
                    | TypedAssignTarget::Field { .. }
                    | TypedAssignTarget::BufferIndex { .. },
                ..
            } => {}
            _ => {}
        }
        i += 1;
    }
}

/// Build the appropriate drop statement for a named binding of the given type.
/// Returns `None` for types that own no heap resources (scalar, bool, unit).
///
/// M29 (ADR-0026, D6): the top-level `Tensor` case is always a static `Drop`,
/// never an RC `Release`. Pre-M29, a grad-tracked tensor got `Release` instead
/// (ADR-0030) on the theory that it might be tape-saved and so needed RC to
/// survive past this binding's last use. That's unnecessary: every
/// `tape_record_*` fn (`malus-runtime/src/tape.rs`) retains its own saved
/// operands synchronously, before control ever returns to a point where CTMM
/// could drop them — the tape's self-retain, not this binding's Release, is
/// what keeps a tape-saved tensor alive. `Drop` and the old `Release` already
/// lowered to the identical runtime op (`tensor_free` delegates to
/// `tensor_release`, `malus-runtime/src/metal.rs`), so this is not a behavior
/// change for the decrement itself — it only stops counting these as RC ops in
/// the M29 compile-time reduction-ratio gate. Struct/enum/tuple/array/List
/// fields are unaffected: always RC-managed regardless of grad-tracking (see
/// `emit_drop_field` in codegen-cpu, and `List`'s ADR-0034 carve-out below).
fn make_drop_stmt_for_ty(name: &str, ty: &ResolvedTy) -> Option<TypedStmt> {
    match ty {
        ResolvedTy::Tensor { .. } => Some(TypedStmt::Drop {
            name: name.to_string(),
        }),
        // Str is a leaked whole-program-lifetime buffer (ADR-0018). No drop needed.
        ResolvedTy::Str => None,
        ResolvedTy::Struct { fields, .. } => {
            let droppable_fields = droppable_struct_fields(fields);
            Some(TypedStmt::DropStruct {
                name: name.to_string(),
                droppable_fields,
                retained_agg_slots: vec![],
            })
        }
        ResolvedTy::Enum { variants, .. } => {
            let drop_variants = droppable_enum_variants(variants);
            Some(TypedStmt::DropEnum {
                name: name.to_string(),
                variants: drop_variants,
            })
        }
        // All arrays own heap memory (the box itself), so always emit DropArray.
        // Codegen handles whether to loop and release elements (only for owned types).
        ResolvedTy::Array { elem, len } => Some(TypedStmt::DropArray {
            name: name.to_string(),
            elem_ty: *elem.clone(),
            len: *len,
        }),
        ResolvedTy::Tuple(elements) => {
            let droppable_fields: Vec<(usize, ResolvedTy)> = elements
                .iter()
                .enumerate()
                .filter_map(|(i, ty)| {
                    if ty.owns_heap_resources() {
                        Some((i, ty.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            Some(TypedStmt::DropTuple {
                name: name.to_string(),
                droppable_fields,
            })
        }
        ResolvedTy::Buffer { .. } => Some(TypedStmt::DropBuffer {
            name: name.to_string(),
        }),
        // M28: `List<T>` is ALWAYS reference-counted (ADR-0034) — never a static
        // Drop, regardless of `grad_tracked` (that flag is about element content,
        // orthogonal to the container's own lifetime; see the ADR-0030 gotcha).
        // Identity-returning a struct's List field (e.g. `Module::parameters`)
        // creates aliasing across a call boundary that neither this pass nor
        // M29's (intraprocedural-only) borrow-inference can prove safe to
        // statically free — RC is the sound fallback.
        ResolvedTy::List { elem } => Some(TypedStmt::DropList {
            name: name.to_string(),
            elem_ty: *elem.clone(),
        }),
        _ => None,
    }
}

// M34: field/element droppability is the shared `owns_heap_resources`
// predicate (ty.rs) — pre-M34 these filters skipped `List` (and `Array`,
// `Tuple`) fields entirely, so a struct holding a `List<Tensor<f32>>` leaked
// the whole list on drop.
fn droppable_struct_fields(fields: &[(String, ResolvedTy)]) -> Vec<(usize, ResolvedTy)> {
    fields
        .iter()
        .enumerate()
        .filter_map(|(i, (_, ty))| {
            if ty.owns_heap_resources() {
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
    variants
        .iter()
        .enumerate()
        .map(|(tag, (_, fields))| {
            let droppable = fields
                .iter()
                .enumerate()
                .filter_map(|(i, (_, ty))| {
                    if ty.owns_heap_resources() {
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
        let live_here: HashSet<String> = drop_positions
            .iter()
            .filter(|(_, &pos)| pos > i)
            .map(|(name, _)| name.clone())
            .collect();
        if live_here.is_empty() {
            continue;
        }
        // We can't hold a mutable borrow to body[i] via index while also
        // having live_here referencing drop_positions — use an index match.
        let is_cf = matches!(
            &body[i],
            TypedStmt::If { .. }
                | TypedStmt::For { .. }
                | TypedStmt::While { .. }
                | TypedStmt::ForIn { .. }
                | TypedStmt::Match { .. }
                | TypedStmt::NoGrad { .. }
        );
        if !is_cf {
            continue;
        }
        match &mut body[i] {
            TypedStmt::If {
                then_body,
                else_body,
                ..
            } => {
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
            TypedStmt::If { .. }
            | TypedStmt::For { .. }
            | TypedStmt::While { .. }
            | TypedStmt::ForIn { .. }
            | TypedStmt::Match { .. }
            | TypedStmt::NoGrad { .. } => {
                // Recurse into nested control flow.
                match &mut body[i] {
                    TypedStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
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
                            inject_unwind_in_body(
                                &mut arm.body,
                                live_outer,
                                local_types,
                            );
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

fn make_unwind_drop(
    name: &str,
    local_types: &HashMap<String, ResolvedTy>,
) -> Option<TypedStmt> {
    local_types
        .get(name)
        .and_then(|ty| make_drop_stmt_for_ty(name, ty))
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

fn inject_break_continue_unwinds(
    body: &mut Vec<TypedStmt>,
    local_types: &HashMap<String, ResolvedTy>,
) {
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
        let live_here: HashSet<String> = drop_positions
            .iter()
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
                let mut to_drop: Vec<String> = live_here
                    .into_iter()
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
                let mut to_drop: Vec<String> = live_outer
                    .iter()
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
                    TypedStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        inject_bc_unwind_in_body(then_body, live_outer, local_types);
                        if let Some(eb) = else_body {
                            inject_bc_unwind_in_body(eb, live_outer, local_types);
                        }
                    }
                    TypedStmt::Match { arms, .. } => {
                        for arm in arms.iter_mut() {
                            inject_bc_unwind_in_body(
                                &mut arm.body,
                                live_outer,
                                local_types,
                            );
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
        TypedStmt::If {
            then_body,
            else_body,
            ..
        } => {
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
        if matches!(
            &body[i],
            TypedStmt::If { .. }
                | TypedStmt::For { .. }
                | TypedStmt::While { .. }
                | TypedStmt::ForIn { .. }
                | TypedStmt::NoGrad { .. }
        ) {
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
        | TypedStmt::DropList { .. }
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
        | TypedStmt::NoGrad { .. }
        | TypedStmt::LetShared { .. } => return None,
    };
    match &expr.kind {
        TypedExprKind::KernelCall { in_flight, .. } => Some((in_flight.clone(), output_name)),
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
            TypedStmt::Let { name, .. } => {
                out.insert(name.clone());
            }
            TypedStmt::LetTuple { names, .. } => {
                for (n, _) in names {
                    out.insert(n.clone());
                }
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
            TypedStmt::Let { name, expr } => {
                out.insert(name.clone(), expr.ty.clone());
            }
            TypedStmt::LetTuple { names, .. } => {
                for (n, ty) in names {
                    out.insert(n.clone(), ty.clone());
                }
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
            TypedStmt::Assign { expr, .. } if expr.ty.is_tensor() => {
                collect_idents_in_expr(expr, out);
            }
            TypedStmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_escaping_in(then_body, out);
                if let Some(eb) = else_body {
                    collect_escaping_in(eb, out);
                }
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
        | TypedStmt::DropList { .. }
        | TypedStmt::DropBuffer { .. }
        | TypedStmt::GpuBarrier
        | TypedStmt::Retain { .. }
        | TypedStmt::Release { .. }
        | TypedStmt::RetainAgg { .. }
        | TypedStmt::ReleaseAgg { .. }
        | TypedStmt::Break
        | TypedStmt::Continue => {}
        TypedStmt::LetTuple { expr, .. } => collect_idents_in_expr(expr, out),
        TypedStmt::If {
            condition,
            then_body,
            else_body,
        } => {
            collect_idents_in_expr(condition, out);
            for s in then_body {
                collect_idents_in_stmt(s, out);
            }
            if let Some(eb) = else_body {
                for s in eb {
                    collect_idents_in_stmt(s, out);
                }
            }
        }
        TypedStmt::For {
            start, end, body, ..
        } => {
            collect_idents_in_expr(start, out);
            collect_idents_in_expr(end, out);
            for s in body {
                collect_idents_in_stmt(s, out);
            }
        }
        TypedStmt::While { condition, body } => {
            collect_idents_in_expr(condition, out);
            for s in body {
                collect_idents_in_stmt(s, out);
            }
        }
        TypedStmt::ForIn { iter, body, .. } => {
            collect_idents_in_expr(iter, out);
            for s in body {
                collect_idents_in_stmt(s, out);
            }
        }
        TypedStmt::Match { scrutinee, arms } => {
            collect_idents_in_expr(scrutinee, out);
            for arm in arms {
                for s in &arm.body {
                    collect_idents_in_stmt(s, out);
                }
            }
        }
        TypedStmt::NoGrad { body } => {
            for s in body {
                collect_idents_in_stmt(s, out);
            }
        }
        TypedStmt::LetShared { .. } => {}
    }
}

/// Insert `Retain { name }` immediately before any `Let`/`Assign`/`LetTuple`
/// statement whose top-level expression is a same-scope tensor *alias*
/// (`let b = a` or `let t = v.data`) or stores a tensor ident into an
/// aggregate literal (`ArrayLiteral`/`StructInit`).
///
/// M29 (ADR-0026, D2): the *function-call-argument* case this used to also
/// cover — the caller retaining a grad-tracked ident before passing it, to
/// balance a `Release` the callee drew from its own seeded param — is
/// removed. Tensor params are now a uniform zero-cost borrow ABI (see
/// `annotate_fns`): the caller never retains before a call, and the callee
/// never independently owns (and so never drops) a param. The alias and
/// aggregate-literal cases below are unrelated to that call-boundary protocol
/// — they guard against two *same-function* bindings independently dropping
/// the identical handle — and are unaffected by this change.
///
/// Shape recognition delegates to `retain_sites` (`retain_sites.rs`) — the
/// single source of truth for "which alias shapes are retain-worthy," also
/// consulted by `borrow_inference` when deciding which of these retains are
/// safe to demote.
fn insert_variable_arc_retains(body: &mut Vec<TypedStmt>) {
    let mut i = 0;
    while i < body.len() {
        let retain_names: Vec<String> = match &body[i] {
            TypedStmt::Let { expr, .. } | TypedStmt::LetTuple { expr, .. } => {
                tensor_retain_names(expr)
            }
            TypedStmt::Assign { expr, .. } => tensor_retain_names(expr),
            _ => vec![],
        };
        for name in retain_names.into_iter().rev() {
            body.insert(i, TypedStmt::Retain { name });
            i += 1;
        }
        i += 1;
    }
}

fn tensor_retain_names(expr: &TypedExpr) -> Vec<String> {
    crate::retain_sites::retain_sites(expr)
        .into_iter()
        .filter(|s| s.kind == crate::retain_sites::RetainKind::Tensor)
        .filter_map(|s| match s.target {
            crate::retain_sites::RetainTarget::Source(name) => Some(name),
            crate::retain_sites::RetainTarget::Binding => None,
        })
        .collect()
}

/// M28: insert `RetainAgg { name }` before any `Let`/`LetMut`/`Assign`/`Return`
/// statement whose top-level expression aliases an existing `List<T>` binding —
/// either a bare `let b = a` rebind, or an ident used as a `List`-typed field in
/// a `StructInit` (e.g. `GPT(params=params)`). Mirrors
/// `insert_variable_arc_retains` above but is unconditional on type (`List` RC
/// is structural, not content/grad-tracked-based — ADR-0034) and emits
/// `RetainAgg` (the existing, already-wired struct/tuple/enum aggregate-RC
/// primitive) rather than the tensor-specific `Retain`.
///
/// Without this, `let gpt = GPT(params=params)` would leave `params`'s
/// `DropList` (inserted at its last-use — this very statement) as the box's
/// *only* release, dropping the refcount from 1 to 0 and freeing the element
/// tensors + box out from under `gpt.params`, which now holds the identical
/// pointer. The retain here (1→2) plus `params`'s own release (2→1) leaves
/// exactly one live reference, now owned by the struct field — the same
/// retain-before/release-after dance `insert_variable_arc_retains` already uses
/// for grad-tracked tensor fields.
///
/// Scope note: only `Ident`-shaped aliasing sites are handled (matching every
/// List construction site in the V4 capstone). A `List` passed as a plain
/// (non-`mut`) call argument would need the same treatment if ever added —
/// `mut` params are unaffected (borrows, never separately dropped; ADR-0025).
///
/// Shape recognition delegates to `retain_sites` (`retain_sites.rs`) — see
/// `insert_variable_arc_retains`'s doc comment for the single-source-of-truth
/// rationale shared with `borrow_inference`.
fn insert_list_retains(body: &mut Vec<TypedStmt>) {
    let mut i = 0;
    while i < body.len() {
        let retain_names: Vec<String> = match &body[i] {
            TypedStmt::Let { expr, .. } | TypedStmt::LetTuple { expr, .. } => {
                list_retain_names(expr)
            }
            TypedStmt::Assign { expr, .. } => list_retain_names(expr),
            // Return only recognizes the bare-Ident shape, not e.g. a
            // StructInit field — a narrower scope than Let/Assign, matching
            // the original hand-written arm this replaces.
            TypedStmt::Return { expr } => list_retain_names(expr)
                .into_iter()
                .filter(|_| matches!(expr.kind, TypedExprKind::Ident(_)))
                .collect(),
            _ => vec![],
        };
        for name in retain_names.into_iter().rev() {
            body.insert(i, TypedStmt::RetainAgg { name });
            i += 1;
        }
        i += 1;
    }
}

fn list_retain_names(expr: &TypedExpr) -> Vec<String> {
    crate::retain_sites::retain_sites(expr)
        .into_iter()
        .filter(|s| s.kind == crate::retain_sites::RetainKind::Agg)
        .filter_map(|s| match s.target {
            crate::retain_sites::RetainTarget::Source(name) => Some(name),
            crate::retain_sites::RetainTarget::Binding => None,
        })
        .collect()
}

/// M34 (done-when #0, bug (b)): a container-element read (`base[i]` on a
/// `List` or `Array`) that gets BOUND to a name steals the container's
/// reference — CTMM emits a static Drop (or RC Release) for the new binding,
/// but the handle it loads is the one the container itself owns and will
/// independently release when the container is dropped. Every such bind is
/// given its own reference, bumped on the *new binding* right after the bind
/// (there is no source name to retain before it — `retain_sites`'s
/// `RetainTarget::Binding`).
///
/// Under the tape the theft is masked one-for-one by the tape's saved-operand
/// retains until `backward()` auto-clears the tape, at which point the
/// container's element is freed out from under it — the M32-addendum
/// "loss.data reads a freed buffer" corruption. Untaped, it is a direct
/// over-release (deterministic panic).
///
/// Handles the three statement shapes that bind or move an element read:
///   - `let w = base[i]`            → Retain/RetainAgg{w} inserted after
///   - `x = base[i]` (Ident target) → Retain/RetainAgg{x} inserted after
///   - `return base[i]` and `a[j] = base[i]` (slot targets) → the read is
///     hoisted to a fresh `__malus_tmp_N` Let (function-unique counter) with
///     its retain, and the original statement consumes the temp.
///
/// Transient inline reads (operands, call args) are untouched — they borrow
/// the container's reference for the duration of the enclosing statement,
/// exactly the `model.params[IDX]` inline pattern the capstone uses.
fn insert_container_read_retains(body: &mut Vec<TypedStmt>, tmp_counter: &mut u32) {
    use crate::retain_sites::{retain_sites, RetainKind, RetainTarget};

    fn binding_retain_kind(expr: &TypedExpr) -> Option<RetainKind> {
        retain_sites(expr)
            .into_iter()
            .find(|s| s.target == RetainTarget::Binding)
            .map(|s| s.kind)
    }

    fn retain_stmt(kind: RetainKind, name: String) -> TypedStmt {
        match kind {
            RetainKind::Tensor => TypedStmt::Retain { name },
            RetainKind::Agg => TypedStmt::RetainAgg { name },
        }
    }

    let mut i = 0;
    while i < body.len() {
        match &body[i] {
            TypedStmt::Let { name, expr } => {
                if let Some(kind) = binding_retain_kind(expr) {
                    let name = name.clone();
                    body.insert(i + 1, retain_stmt(kind, name));
                    i += 1;
                }
            }
            TypedStmt::Assign { target: crate::typed_ir::TypedAssignTarget::Ident(name), expr } => {
                if let Some(kind) = binding_retain_kind(expr) {
                    let name = name.clone();
                    body.insert(i + 1, retain_stmt(kind, name));
                    i += 1;
                }
            }
            TypedStmt::Assign { expr, .. } | TypedStmt::Return { expr } => {
                if let Some(kind) = binding_retain_kind(expr) {
                    let tmp_name = format!("__malus_tmp_{}", tmp_counter);
                    *tmp_counter += 1;
                    let (stmt, expr_owned) = match body.remove(i) {
                        TypedStmt::Assign { target, expr } => {
                            (Some(target), expr)
                        }
                        TypedStmt::Return { expr } => (None, expr),
                        _ => unreachable!(),
                    };
                    let ident = TypedExpr {
                        kind: TypedExprKind::Ident(tmp_name.clone()),
                        ty: expr_owned.ty.clone(),
                        placement: expr_owned.placement,
                        span: expr_owned.span,
                        grad_tracked: expr_owned.grad_tracked,
                    };
                    body.insert(i, TypedStmt::Let { name: tmp_name.clone(), expr: expr_owned });
                    body.insert(i + 1, retain_stmt(kind, tmp_name));
                    let consumer = match stmt {
                        Some(target) => TypedStmt::Assign { target, expr: ident },
                        None => TypedStmt::Return { expr: ident },
                    };
                    body.insert(i + 2, consumer);
                    i += 2;
                }
            }
            _ => {}
        }
        i += 1;
    }
}

fn collect_idents_in_expr(expr: &crate::typed_ir::TypedExpr, out: &mut HashSet<String>) {
    match &expr.kind {
        TypedExprKind::Ident(name) => {
            out.insert(name.clone());
        }
        TypedExprKind::BinOp { lhs, rhs, .. } => {
            collect_idents_in_expr(lhs, out);
            collect_idents_in_expr(rhs, out);
        }
        TypedExprKind::Unary { operand, .. } => collect_idents_in_expr(operand, out),
        TypedExprKind::Call { args, .. } => {
            for a in args {
                collect_idents_in_expr(a, out);
            }
        }
        TypedExprKind::KernelCall { args, .. } => {
            for a in args {
                collect_idents_in_expr(a, out);
            }
        }
        TypedExprKind::TensorLiteral { elements, .. } => {
            for e in elements {
                collect_idents_in_expr(e, out);
            }
        }
        TypedExprKind::Index { base, indices } => {
            collect_idents_in_expr(base, out);
            for i in indices {
                collect_idents_in_expr(i, out);
            }
        }
        TypedExprKind::FieldAccess { base, .. } => collect_idents_in_expr(base, out),
        TypedExprKind::StructInit { fields, .. } => {
            for f in fields {
                collect_idents_in_expr(f, out);
            }
        }
        TypedExprKind::EnumInit { payload, .. } => {
            for p in payload {
                collect_idents_in_expr(p, out);
            }
        }
        TypedExprKind::ArrayLiteral { elements } => {
            for e in elements {
                collect_idents_in_expr(e, out);
            }
        }
        TypedExprKind::TupleInit { elements } => {
            for e in elements {
                collect_idents_in_expr(e, out);
            }
        }
        TypedExprKind::TupleIndex { base, .. } => collect_idents_in_expr(base, out),
        TypedExprKind::KernelLaunch {
            grid,
            tg,
            out_shape,
            tensor_args,
            scalar_args,
            ..
        } => {
            collect_idents_in_expr(grid, out);
            collect_idents_in_expr(tg, out);
            if let Some(s) = out_shape {
                collect_idents_in_expr(s, out);
            }
            for a in tensor_args {
                collect_idents_in_expr(a, out);
            }
            for a in scalar_args {
                collect_idents_in_expr(a, out);
            }
        }
        TypedExprKind::Lit(_) => {}
    }
}
