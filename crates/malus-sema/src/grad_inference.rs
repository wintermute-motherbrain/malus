// M27 grad-inference: whole-program, field-sensitive, interprocedural static
// analysis that computes which tensor-producing expressions may be saved onto
// the autograd tape ("grad-tracked"). Runs after `check`, before `ctmm`.
//
// Replaces the old distinct Variable type-directed signal (ADR-0016) now
// that there is a single `Tensor` type (ADR-0030). See the M27 plan for the
// full design; summary of the fixpoint:
//   - `variable(t)` is the only grad-tracked seed (leaf marker).
//   - BinOp/Call results are grad-tracked iff any operand is.
//   - `.data`/`.grad` are detach points: always non-grad-tracked, regardless
//     of the receiver.
//   - Everything lexically inside `with no_grad:` is forced non-grad-tracked.
//   - Struct fields are tracked per `(struct_name, field_name)`, written by
//     construction/field-assign, read by field access — this is what carries
//     grad-tracking across `nanogpt.ml`'s `Block`/`GPT` structs.
//   - Function params/returns are tracked per-fn, unioned over all call sites
//     (context-insensitive).
// The whole thing is a monotone fixpoint (flags only flip false → true) over
// the typed IR, annotating `TypedExpr.grad_tracked` in place.

use std::collections::HashMap;
use crate::ty::ResolvedTy;
use crate::typed_ir::{TypedAssignTarget, TypedExpr, TypedExprKind, TypedProgram, TypedStmt};

pub fn infer(program: &mut TypedProgram) {
    for f in &program.fns {
        program.fn_param_grad.entry(f.name.clone()).or_insert_with(|| vec![false; f.params.len()]);
        program.fn_ret_grad.entry(f.name.clone()).or_insert(false);
    }

    loop {
        let mut changed = false;
        let mut fns = std::mem::take(&mut program.fns);
        for f in fns.iter_mut() {
            let param_grad = program.fn_param_grad.get(&f.name).cloned().unwrap_or_default();
            let mut locals: HashMap<String, bool> = HashMap::new();
            let mut local_types: HashMap<String, ResolvedTy> = HashMap::new();
            for (p, g) in f.params.iter().zip(param_grad.iter()) {
                locals.insert(p.name.clone(), *g);
                local_types.insert(p.name.clone(), p.ty.clone());
            }
            let mut pass = Pass { program, changed: false, no_grad_depth: 0, ret_grad: false };
            pass.infer_body(&mut f.body, &mut locals, &mut local_types);
            if pass.changed {
                changed = true;
            }
            let ret_grad = pass.ret_grad;
            let cur = program.fn_ret_grad.entry(f.name.clone()).or_insert(false);
            if ret_grad && !*cur {
                *cur = true;
                changed = true;
            }
        }
        program.fns = fns;
        if !changed {
            break;
        }
    }
}

struct Pass<'p> {
    program: &'p mut TypedProgram,
    changed: bool,
    no_grad_depth: usize,
    ret_grad: bool,
}

impl<'p> Pass<'p> {
    fn infer_body(
        &mut self,
        body: &mut [TypedStmt],
        locals: &mut HashMap<String, bool>,
        local_types: &mut HashMap<String, ResolvedTy>,
    ) {
        for stmt in body.iter_mut() {
            self.infer_stmt(stmt, locals, local_types);
        }
    }

    /// Process `body` in a cloned child scope (so bindings introduced inside
    /// don't leak out), plus any `extra` bindings scoped only to `body` (loop
    /// vars, match-arm payloads). After processing, folds any grad-flag flips
    /// for names that already existed in the parent back into `locals`
    /// (monotone union) — this is how a `let mut` reassignment inside a nested
    /// `if`/loop propagates out to the enclosing scope.
    fn infer_scoped(
        &mut self,
        body: &mut [TypedStmt],
        locals: &mut HashMap<String, bool>,
        local_types: &mut HashMap<String, ResolvedTy>,
        extra: &[(String, ResolvedTy)],
    ) {
        let mut child_locals = locals.clone();
        let mut child_types = local_types.clone();
        for (n, t) in extra {
            child_locals.insert(n.clone(), false);
            child_types.insert(n.clone(), t.clone());
        }
        self.infer_body(body, &mut child_locals, &mut child_types);
        for (k, v) in child_locals {
            if v {
                if let Some(slot) = locals.get_mut(&k) {
                    if !*slot {
                        *slot = true;
                    }
                }
            }
        }
    }

    fn infer_stmt(
        &mut self,
        stmt: &mut TypedStmt,
        locals: &mut HashMap<String, bool>,
        local_types: &mut HashMap<String, ResolvedTy>,
    ) {
        match stmt {
            TypedStmt::Let { name, expr } => {
                let g = self.infer_expr(expr, locals, local_types);
                local_types.insert(name.clone(), expr.ty.clone());
                let slot = locals.entry(name.clone()).or_insert(false);
                if g && !*slot {
                    *slot = true;
                }
            }
            TypedStmt::LetTuple { names, expr } => {
                let g = self.infer_expr(expr, locals, local_types);
                for (n, ty) in names.iter() {
                    local_types.insert(n.clone(), ty.clone());
                    let slot = locals.entry(n.clone()).or_insert(false);
                    if g && !*slot {
                        *slot = true;
                    }
                }
            }
            TypedStmt::Assign { target, expr } => {
                let g = self.infer_expr(expr, locals, local_types);
                match target {
                    TypedAssignTarget::Ident(name) => {
                        let slot = locals.entry(name.clone()).or_insert(false);
                        if g && !*slot {
                            *slot = true;
                        }
                    }
                    TypedAssignTarget::Field { base, slot_idx, .. } => {
                        if g {
                            if let Some(ResolvedTy::Struct { name: sname, fields }) =
                                local_types.get(base.as_str())
                            {
                                if let Some((fname, _)) = fields.get(*slot_idx) {
                                    let key = (sname.clone(), fname.clone());
                                    if self.program.struct_field_grad.insert(key) {
                                        self.changed = true;
                                    }
                                }
                            }
                        }
                    }
                    TypedAssignTarget::Index { index, .. } => {
                        self.infer_expr(index, locals, local_types);
                    }
                    TypedAssignTarget::BufferIndex { index, .. } => {
                        self.infer_expr(index, locals, local_types);
                    }
                }
            }
            TypedStmt::Return { expr } => {
                let g = self.infer_expr(expr, locals, local_types);
                if g {
                    self.ret_grad = true;
                }
            }
            TypedStmt::Expr(expr) => {
                self.infer_expr(expr, locals, local_types);
            }
            TypedStmt::If { condition, then_body, else_body } => {
                self.infer_expr(condition, locals, local_types);
                self.infer_scoped(then_body, locals, local_types, &[]);
                if let Some(eb) = else_body {
                    self.infer_scoped(eb, locals, local_types, &[]);
                }
            }
            TypedStmt::For { var, start, end, body } => {
                self.infer_expr(start, locals, local_types);
                self.infer_expr(end, locals, local_types);
                let vty = start.ty.clone();
                self.infer_scoped(body, locals, local_types, &[(var.clone(), vty)]);
            }
            TypedStmt::While { condition, body } => {
                self.infer_expr(condition, locals, local_types);
                self.infer_scoped(body, locals, local_types, &[]);
            }
            TypedStmt::ForIn { var, iter, body } => {
                self.infer_expr(iter, locals, local_types);
                let ety = if let ResolvedTy::Array { elem, .. } = &iter.ty {
                    (**elem).clone()
                } else {
                    ResolvedTy::Unit
                };
                self.infer_scoped(body, locals, local_types, &[(var.clone(), ety)]);
            }
            TypedStmt::Match { scrutinee, arms } => {
                self.infer_expr(scrutinee, locals, local_types);
                for arm in arms.iter_mut() {
                    let extra = arm.bindings.clone();
                    self.infer_scoped(&mut arm.body, locals, local_types, &extra);
                }
            }
            TypedStmt::NoGrad { body } => {
                self.no_grad_depth += 1;
                self.infer_scoped(body, locals, local_types, &[]);
                self.no_grad_depth -= 1;
            }
            // CTMM hasn't run yet at this point in the pipeline (grad_inference
            // runs between `check` and `ctmm`), so none of the Drop/Retain/
            // Release/GpuBarrier nodes exist. Break/Continue/LetShared carry no
            // expressions to annotate.
            TypedStmt::Break
            | TypedStmt::Continue
            | TypedStmt::LetShared { .. }
            | TypedStmt::Drop { .. }
            | TypedStmt::GpuBarrier
            | TypedStmt::Retain { .. }
            | TypedStmt::Release { .. }
            | TypedStmt::RetainAgg { .. }
            | TypedStmt::ReleaseAgg { .. }
            | TypedStmt::DropStruct { .. }
            | TypedStmt::DropEnum { .. }
            | TypedStmt::DropArray { .. }
            | TypedStmt::DropTuple { .. }
            | TypedStmt::DropBuffer { .. } => {}
        }
    }

    /// Computes the grad-tracked flag for `expr`, recursing into subexpressions,
    /// updating interprocedural/field-sensitive maps as a side effect, and
    /// writing the (monotone) result into `expr.grad_tracked`. Returns the flag.
    fn infer_expr(
        &mut self,
        expr: &mut TypedExpr,
        locals: &HashMap<String, bool>,
        local_types: &HashMap<String, ResolvedTy>,
    ) -> bool {
        let outer_ty = expr.ty.clone();
        let natural = match &mut expr.kind {
            TypedExprKind::Lit(_) => false,
            TypedExprKind::Ident(name) => locals.get(name.as_str()).copied().unwrap_or(false),
            TypedExprKind::BinOp { lhs, rhs, .. } => {
                let l = self.infer_expr(lhs, locals, local_types);
                let r = self.infer_expr(rhs, locals, local_types);
                l || r
            }
            TypedExprKind::Unary { operand, .. } => self.infer_expr(operand, locals, local_types),
            TypedExprKind::Call { callee, args } => {
                let mut any_arg_grad = false;
                for a in args.iter_mut() {
                    let g = self.infer_expr(a, locals, local_types);
                    any_arg_grad = any_arg_grad || g;
                }
                if callee == "variable" {
                    // Leaf marker: always grad-tracked regardless of the arg.
                    true
                } else if self.program.fn_ret_grad.contains_key(callee.as_str()) {
                    if let Some(slots) = self.program.fn_param_grad.get_mut(callee.as_str()) {
                        for (i, a) in args.iter().enumerate() {
                            if a.grad_tracked && i < slots.len() && !slots[i] {
                                slots[i] = true;
                                self.changed = true;
                            }
                        }
                    }
                    *self.program.fn_ret_grad.get(callee.as_str()).unwrap_or(&false)
                } else {
                    // Builtin: known-differentiable builtins (relu, matmul-as-BinOp,
                    // softmax, cross_entropy, embedding, reductions, shape ops, ...)
                    // are grad-tracked iff any operand is; non-differentiable builtins
                    // (zeros, randn, read_file, print, ...) never have a grad-tracked
                    // tensor arg in the first place, so this is safe to apply uniformly.
                    any_arg_grad
                }
            }
            TypedExprKind::KernelCall { args, .. } => {
                for a in args.iter_mut() {
                    self.infer_expr(a, locals, local_types);
                }
                // User-defined `kernel` dispatches are raw GPU work, never tape-recorded.
                false
            }
            TypedExprKind::Index { base, indices } => {
                let b = self.infer_expr(base, locals, local_types);
                for i in indices.iter_mut() {
                    self.infer_expr(i, locals, local_types);
                }
                b
            }
            TypedExprKind::TensorLiteral { elements, .. } => {
                for e in elements.iter_mut() {
                    self.infer_expr(e, locals, local_types);
                }
                false
            }
            TypedExprKind::FieldAccess { base, field } => {
                let b = self.infer_expr(base, locals, local_types);
                if field == "data" || field == "grad" {
                    // Detach point (decision #3): always non-grad-tracked.
                    false
                } else if field == "len" || field == "ndim" || field == "shape" || field == "strides" {
                    false
                } else if let ResolvedTy::Struct { name: sname, .. } = &base.ty {
                    self.program.struct_field_grad.contains(&(sname.clone(), field.clone()))
                } else {
                    b
                }
            }
            TypedExprKind::ArrayLiteral { elements } => {
                let mut any = false;
                for e in elements.iter_mut() {
                    let g = self.infer_expr(e, locals, local_types);
                    any = any || g;
                }
                any
            }
            TypedExprKind::StructInit { name, fields } => {
                let field_names: Vec<String> = if let ResolvedTy::Struct { fields: tfields, .. } = &outer_ty {
                    tfields.iter().map(|(n, _)| n.clone()).collect()
                } else {
                    vec![]
                };
                for (i, f) in fields.iter_mut().enumerate() {
                    let g = self.infer_expr(f, locals, local_types);
                    if g {
                        if let Some(fname) = field_names.get(i) {
                            let key = (name.clone(), fname.clone());
                            if self.program.struct_field_grad.insert(key) {
                                self.changed = true;
                            }
                        }
                    }
                }
                false
            }
            TypedExprKind::EnumInit { payload, .. } => {
                for p in payload.iter_mut() {
                    self.infer_expr(p, locals, local_types);
                }
                false
            }
            TypedExprKind::TupleInit { elements } => {
                let mut any = false;
                for e in elements.iter_mut() {
                    let g = self.infer_expr(e, locals, local_types);
                    any = any || g;
                }
                any
            }
            TypedExprKind::TupleIndex { base, .. } => self.infer_expr(base, locals, local_types),
            TypedExprKind::KernelLaunch { grid, tg, out_shape, tensor_args, scalar_args, .. } => {
                self.infer_expr(grid, locals, local_types);
                self.infer_expr(tg, locals, local_types);
                if let Some(s) = out_shape {
                    self.infer_expr(s, locals, local_types);
                }
                for a in tensor_args.iter_mut() {
                    self.infer_expr(a, locals, local_types);
                }
                for a in scalar_args.iter_mut() {
                    self.infer_expr(a, locals, local_types);
                }
                // Explicit kernel launches are raw GPU work, never tape-recorded.
                false
            }
        };
        // Bindings lexically inside `with no_grad:` are forced non-grad-tracked,
        // overriding whatever the normal propagation rules above computed.
        let g = if self.no_grad_depth > 0 { false } else { natural };
        if g && !expr.grad_tracked {
            expr.grad_tracked = true;
            self.changed = true;
        }
        g
    }
}
