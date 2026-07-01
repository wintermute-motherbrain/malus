// M29 — Lobster-style borrow-inference (ADR-0026, D3). Identifies, among the
// tensor-alias Retain+Drop/Release pairs CTMM's existing (pre-M29,
// conservative) machinery already emits, which ones are *provably*
// redundant — safe to remove as a matched pair without changing the
// runtime's refcount trajectory anywhere.
//
// Design (see the M29 plan / ADR-0026 "Resolved decisions" D2-D4): rather
// than a separate whole-program pass computing owner/borrow sets ahead of
// CTMM, this runs as a post-process cleanup over each function's *already
// fully CTMM-annotated* body. That body is provably correct on its own
// (every Retain has a matching Drop/Release); this pass only removes pairs
// it can show are a net no-op:
//
//   - `let b = a` (or `let t = v.data`) where `a` is a tensor PARAM (or a
//     transitively-classified borrow of one): the caller never retained a
//     before the call (uniform borrow ABI, D2), so `a` never independently
//     owns a reference in this function — any Retain{b}/Drop{b} pair CTMM
//     emitted for this alias is pure overhead, safe to remove unconditionally.
//   - `let b = a` where `a` is a local, never-reassigned owner: the pair is
//     safe to remove IFF `a`'s own Drop/Release occurs no earlier than `b`'s
//     in the final statement sequence — i.e. `a` stays alive at least as long
//     as `b` did, so `a`'s existing drop already covers everything `b`'s
//     drop would have (ADR-0026's conservative "last use" criterion, verified
//     directly against CTMM's own computed drop positions rather than
//     recomputing last-use order independently — avoids a second, possibly
//     inconsistent, source of truth for the same fact).
//
// Deliberately conservative and intraprocedural (ADR-0026 "V4 use a
// conservative criterion... if uncertain, treat as owner"): only the two
// alias shapes CTMM already special-cases (`Ident`, `.data`-`FieldAccess`)
// are considered; only flat top-level statements of one function body are
// scanned (an alias crossing an if/for/while boundary is left as-is, still
// correct, just not optimized — matching the scope CTMM's own passes use
// for this class of aliasing).

use std::collections::{HashMap, HashSet};
use crate::typed_ir::{TypedAssignTarget, TypedExpr, TypedExprKind, TypedStmt};

#[derive(Debug, Clone, PartialEq, Eq)]
enum BorrowRoot {
    /// Traces back to a function parameter (or a chain of aliases of one).
    /// Unconditionally safe to demote — no Drop event exists for a param.
    Param,
    /// Traces back to local binding `name`, a never-reassigned owner.
    /// Safe to demote only if `name`'s own drop occurs at or after the
    /// alias's drop in the final body.
    Owner(String),
}

/// Remove provably-redundant `Retain{b}` + `Drop{b}`/`Release{b}` pairs from
/// `body` (this function's fully CTMM-annotated top-level statements),
/// where `b` is a same-scope tensor alias of `param_names` (this function's
/// tensor params) or of a never-reassigned local owner. Mutates `body` in
/// place; returns the count of pairs removed (informational).
pub fn demote_safe_borrows(body: &mut Vec<TypedStmt>, param_names: &HashSet<String>) -> usize {
    let roots = classify_borrows(body, param_names);
    let struct_sources = struct_init_field_sources(body);
    if roots.is_empty() && struct_sources.is_empty() {
        return 0;
    }

    // Locate each name's own Retain and Drop/Release statement index in the
    // flat top-level body (each name has at most one of each, by
    // construction: `insert_variable_arc_retains`/`insert_drops` emit exactly
    // one Retain and one Drop-or-Release per binding).
    let mut retain_idx: HashMap<String, usize> = HashMap::new();
    let mut drop_idx: HashMap<String, usize> = HashMap::new();
    for (i, s) in body.iter().enumerate() {
        match s {
            TypedStmt::Retain { name } => {
                retain_idx.insert(name.clone(), i);
            }
            TypedStmt::Drop { name } | TypedStmt::Release { name } => {
                drop_idx.insert(name.clone(), i);
            }
            _ => {}
        }
    }

    let mut to_remove: HashSet<usize> = HashSet::new();
    for (b, root) in &roots {
        let (Some(&r_idx), Some(&d_idx)) = (retain_idx.get(b), drop_idx.get(b)) else {
            // No Retain/Drop pair was actually emitted for this alias (e.g. it
            // escapes, or wasn't grad-tracked to begin with) — nothing to do.
            continue;
        };
        let safe = match root {
            BorrowRoot::Param => true,
            BorrowRoot::Owner(a) => match drop_idx.get(a.as_str()) {
                // `a`'s own drop must be at or after `b`'s — `a` stays live
                // at least as long as `b` did.
                Some(&a_drop_idx) => a_drop_idx >= d_idx,
                // `a` has no drop of its own in this scope (e.g. it escapes,
                // or is itself a param) — conservative: don't demote.
                None => false,
            },
        };
        if safe {
            to_remove.insert(r_idx);
            to_remove.insert(d_idx);
        }
    }

    // StructInit/ArrayLiteral field/element aliasing: `let blk = Block(w=w, ...)`
    // retains each grad-tracked Ident field so the struct's copy survives
    // independently of the source binding's own drop (`variable_arc_retains_for_expr`'s
    // StructInit/ArrayLiteral cases, ctmm.rs — unconditional, unlike the
    // plain-alias cases above, since a struct/array field is a genuinely
    // different, longer-lived owner than a same-scope `let`).
    //
    // But when the source's OWN last use *is* this exact construction
    // statement (the overwhelmingly common shape: `let w = variable(...);
    // ...; let blk = Block(w=w, ...)` with `w` never referenced again) — its
    // Drop lands somewhere in the contiguous run of Drop/Release statements
    // `insert_drops` places right after the construction (every binding
    // whose last use is that same statement is dropped as one block, sorted
    // by name — not necessarily each at exactly `construction_idx + 1`). In
    // that case the retain+drop pair is pure overhead: ownership can
    // transfer to the field directly (the source binding is gone anyway;
    // ADR-0034's DropStruct field-release becomes the field's sole, correct
    // release point) instead of doubling the refcount only to immediately
    // halve it back.
    for (source, construction_idx) in struct_sources {
        let (Some(&r_idx), Some(&d_idx)) =
            (retain_idx.get(&source), drop_idx.get(&source))
        else {
            continue;
        };
        if d_idx > construction_idx
            && (construction_idx + 1..d_idx)
                .all(|k| matches!(body[k], TypedStmt::Drop { .. } | TypedStmt::Release { .. }))
        {
            to_remove.insert(r_idx);
            to_remove.insert(d_idx);
        }
    }

    if to_remove.is_empty() {
        return 0;
    }
    let removed_pairs = to_remove.len() / 2;
    let mut i = 0;
    body.retain(|_| {
        let keep = !to_remove.contains(&i);
        i += 1;
        keep
    });
    removed_pairs
}

/// For every top-level tensor-typed `let`/`let mut` whose RHS is a borrow-shaped
/// alias (`let b = a` or `let t = v.data`) of an in-scope source, resolve its
/// `BorrowRoot`. Mirrors the two shapes `variable_arc_retains_for_expr`
/// (ctmm.rs) already special-cases.
fn classify_borrows(
    body: &[TypedStmt],
    param_names: &HashSet<String>,
) -> HashMap<String, BorrowRoot> {
    let reassigned = reassigned_names(body);
    let mut roots: HashMap<String, BorrowRoot> = HashMap::new();

    for stmt in body {
        if let TypedStmt::Let { name, expr } = stmt {
            let Some(source) = alias_source(expr) else { continue };
            let root = if param_names.contains(&source) {
                Some(BorrowRoot::Param)
            } else if let Some(existing) = roots.get(&source) {
                // Transitive alias of an already-classified borrow: inherits
                // the same root, whatever class it is.
                Some(existing.clone())
            } else if !reassigned.contains(&source) {
                Some(BorrowRoot::Owner(source))
            } else {
                // Source is reassigned in this scope — its handle isn't
                // stable for the rest of the scope's lifetime. Conservative:
                // not a borrow (ADR-0026 "if uncertain, treat as owner").
                None
            };
            if let Some(r) = root {
                roots.insert(name.clone(), r);
            }
        }
    }
    roots
}

/// Extract `a` from `let b = a` (bare Ident) or `let t = v.data` (`.data`
/// detach-bind), restricted to `Tensor`-typed expressions — the two alias
/// shapes that reuse an existing handle rather than producing a fresh
/// allocation (see `ctmm.rs`'s `variable_arc_retains_for_expr`).
fn alias_source(expr: &TypedExpr) -> Option<String> {
    if !expr.ty.is_tensor() {
        return None;
    }
    match &expr.kind {
        TypedExprKind::Ident(name) => Some(name.clone()),
        TypedExprKind::FieldAccess { base, field } if field == "data" => match &base.kind {
            TypedExprKind::Ident(name) => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// For every top-level `let name = StructInit{...}` or `let name =
/// ArrayLiteral{...}`, map each bare-`Ident`, tensor-typed, grad-tracked
/// field/element source to the index of that construction statement.
/// Mirrors `variable_arc_retains_for_expr`'s StructInit/ArrayLiteral arms
/// (ctmm.rs), which is what actually emitted the Retain this function is
/// deciding whether to remove.
fn struct_init_field_sources(body: &[TypedStmt]) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for (i, stmt) in body.iter().enumerate() {
        let TypedStmt::Let { expr, .. } = stmt else { continue };
        let sources: Vec<&TypedExpr> = match &expr.kind {
            TypedExprKind::StructInit { fields, .. } => fields.iter().collect(),
            TypedExprKind::ArrayLiteral { elements } => elements.iter().collect(),
            _ => continue,
        };
        for f in sources {
            if f.grad_tracked && f.ty.is_tensor() {
                if let TypedExprKind::Ident(name) = &f.kind {
                    out.insert(name.clone(), i);
                }
            }
        }
    }
    out
}

/// Names reassigned anywhere in `body` at the top level (`name = expr`).
/// Matches `classify_borrows`'s flat-scope restriction.
fn reassigned_names(body: &[TypedStmt]) -> HashSet<String> {
    let mut out = HashSet::new();
    for stmt in body {
        if let TypedStmt::Assign { target: TypedAssignTarget::Ident(name), .. } = stmt {
            out.insert(name.clone());
        }
    }
    out
}
