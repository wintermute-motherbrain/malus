// M28 no-unroll lint (docs/milestones/m28-module-trait.md §6, ADR-0034 D6).
//
// A pure AST-level check — no type information needed. Confines the
// hand-unrolled-optimizer smell to exactly one place: the single generic fn
// bounded by `Module`. Retargeted from the original spec's `_m`/`_v`
// struct-field heuristic (dead under the List-backed write-back design,
// ADR-0034) to a stronger, more general property: `.grad` arithmetic may
// appear ONLY inside that one fn. Reintroducing a hand-unrolled update
// anywhere else necessarily reads `.grad`, so it trips this gate — unlike the
// original spec's narrower "no `.grad` arith in `main`'s loop" check, which
// would miss a hand-unrolled helper fn (exactly V3's `adamw_block`/
// `adamw_gpt_params` shape).
//
// This is a CI-gate check (ADR-0031: hard asserts against the running demo,
// not "an implementation exists"), exercised by `crates/malus-cli/src/
// tests.rs`, not the interactive CLI path — the lint's applicability is
// specific to the Module/generic-optimizer capstone shape (a program with no
// such fn, e.g. `add_tensors.ml`, would trivially fail check 1), so it isn't
// run unconditionally on every `malus <file>.ml` invocation.
#![allow(dead_code)]

use malus_syntax::ast::{CallArg, Expr, ExprKind, ItemKind, MatchArm, Program, Stmt, StmtKind};

#[derive(Debug, Clone, PartialEq)]
pub struct LintViolation(pub String);

impl std::fmt::Display for LintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Runs the three no-unroll checks (D6) against every top-level `fn` and
/// `impl` method in `program`. Returns one `LintViolation` per failed check —
/// empty means the program passes.
pub fn check_no_unroll(program: &Program) -> Vec<LintViolation> {
    let mut violations = Vec::new();

    // ── Check 1: exactly one fn generic over `Module` ─────────────────────
    let module_bounded: Vec<&str> = program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn { name, type_params, .. }
                if type_params.iter().any(|tp| tp.bound.as_deref() == Some("Module")) =>
            {
                Some(name.as_str())
            }
            _ => None,
        })
        .collect();
    if module_bounded.len() != 1 {
        violations.push(LintViolation(format!(
            "expected exactly one `fn` generic over `Module`, found {}: {:?}",
            module_bounded.len(),
            module_bounded
        )));
    }
    let optimizer_fn = module_bounded.first().copied();

    // ── Check 2: `.grad` read ONLY inside the optimizer fn ────────────────
    // Reintroducing a hand-unrolled update anywhere else — a helper fn, an
    // impl method, or main's own loop — necessarily reads `.grad` to compute
    // the update, so it always trips this check regardless of shape.
    for item in &program.items {
        match &item.kind {
            ItemKind::Fn { name, body, .. } => {
                let grad_reads = count_grad_reads_body(body);
                if grad_reads > 0 && Some(name.as_str()) != optimizer_fn {
                    violations.push(LintViolation(format!(
                        "fn '{name}' reads `.grad` {grad_reads} time(s) — hand-unrolled \
                         optimizer logic may only live in the fn generic over `Module`"
                    )));
                }
            }
            ItemKind::Impl { for_type, methods, .. } => {
                for m in methods {
                    if let ItemKind::Fn { name, body, .. } = &m.kind {
                        let grad_reads = count_grad_reads_body(body);
                        if grad_reads > 0 {
                            violations.push(LintViolation(format!(
                                "impl method '{for_type}::{name}' reads `.grad` {grad_reads} \
                                 time(s) — hand-unrolled optimizer logic may only live in the \
                                 fn generic over `Module`"
                            )));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // ── Check 3: the optimizer fn calls `.parameters()` exactly once ──────
    if let Some(opt_name) = optimizer_fn {
        if let Some(body) = find_fn_body(program, opt_name) {
            let count = count_parameters_calls_body(body);
            if count != 1 {
                violations.push(LintViolation(format!(
                    "expected exactly one `.parameters()` call inside '{opt_name}', found {count}"
                )));
            }
        }
    }

    violations
}

fn find_fn_body<'a>(program: &'a Program, name: &str) -> Option<&'a [Stmt]> {
    program.items.iter().find_map(|item| match &item.kind {
        ItemKind::Fn { name: n, body, .. } if n == name => Some(body.as_slice()),
        _ => None,
    })
}

fn count_grad_reads_body(body: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in body {
        walk_stmt(stmt, &mut |e| {
            if let ExprKind::FieldAccess { field, .. } = &e.kind {
                if field == "grad" {
                    count += 1;
                }
            }
        });
    }
    count
}

fn count_parameters_calls_body(body: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in body {
        walk_stmt(stmt, &mut |e| {
            if let ExprKind::Call { callee, .. } = &e.kind {
                if let ExprKind::FieldAccess { field, .. } = &callee.kind {
                    if field == "parameters" {
                        count += 1;
                    }
                }
            }
        });
    }
    count
}

/// Recurse into every statement in `stmt`, calling `f` on every expression
/// node (pre-order, including nested sub-expressions via `walk_expr`).
fn walk_stmt(stmt: &Stmt, f: &mut impl FnMut(&Expr)) {
    match &stmt.kind {
        StmtKind::Let { expr, .. } | StmtKind::LetMut { expr, .. } => walk_expr(expr, f),
        StmtKind::Assign { target, expr } => {
            walk_expr(target, f);
            walk_expr(expr, f);
        }
        StmtKind::Return { expr } => walk_expr(expr, f),
        StmtKind::Expr(expr) => walk_expr(expr, f),
        StmtKind::If { condition, then_body, else_body } => {
            walk_expr(condition, f);
            for s in then_body {
                walk_stmt(s, f);
            }
            if let Some(eb) = else_body {
                for s in eb {
                    walk_stmt(s, f);
                }
            }
        }
        StmtKind::For { start, end, body, .. } => {
            walk_expr(start, f);
            walk_expr(end, f);
            for s in body {
                walk_stmt(s, f);
            }
        }
        StmtKind::ForIn { iter, body, .. } => {
            walk_expr(iter, f);
            for s in body {
                walk_stmt(s, f);
            }
        }
        StmtKind::While { condition, body } => {
            walk_expr(condition, f);
            for s in body {
                walk_stmt(s, f);
            }
        }
        StmtKind::Match { scrutinee, arms } => {
            walk_expr(scrutinee, f);
            for MatchArm { body, .. } in arms {
                for s in body {
                    walk_stmt(s, f);
                }
            }
        }
        StmtKind::LetTuple { expr, .. } => walk_expr(expr, f),
        StmtKind::NoGrad { body } => {
            for s in body {
                walk_stmt(s, f);
            }
        }
        StmtKind::Break | StmtKind::Continue | StmtKind::LetShared { .. } => {}
    }
}

fn walk_expr(expr: &Expr, f: &mut impl FnMut(&Expr)) {
    f(expr);
    match &expr.kind {
        ExprKind::Lit(_) | ExprKind::Ident(_) => {}
        ExprKind::BinOp { lhs, rhs, .. } => {
            walk_expr(lhs, f);
            walk_expr(rhs, f);
        }
        ExprKind::Unary { operand, .. } => walk_expr(operand, f),
        ExprKind::Call { callee, args } => {
            walk_expr(callee, f);
            for CallArg { value, .. } in args {
                walk_expr(value, f);
            }
        }
        ExprKind::Index { base, indices } => {
            walk_expr(base, f);
            for i in indices {
                walk_expr(i, f);
            }
        }
        ExprKind::TensorLiteral { elements, .. } | ExprKind::ArrayLiteral { elements } => {
            for e in elements {
                walk_expr(e, f);
            }
        }
        ExprKind::FieldAccess { base, .. } => walk_expr(base, f),
        ExprKind::Tuple(elements) => {
            for e in elements {
                walk_expr(e, f);
            }
        }
        ExprKind::TupleIndex { base, .. } => walk_expr(base, f),
        ExprKind::KernelLaunch { config, args, .. } => {
            for (_, e) in config {
                walk_expr(e, f);
            }
            for CallArg { value, .. } in args {
                walk_expr(value, f);
            }
        }
    }
}
