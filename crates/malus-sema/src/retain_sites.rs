// M29 architecture deepening: the single source of truth for "which alias
// shapes get a Retain/RetainAgg, and what is their source?"
//
// Before this module, that question was answered by THREE independent
// recognizers kept in sync only by prose comments:
//   - ctmm.rs's `variable_arc_retains_for_expr` (tensor `Retain` sites)
//   - ctmm.rs's `list_retains_for_expr` (`List` `RetainAgg` sites)
//   - borrow_inference.rs's `alias_source` + `struct_init_field_sources`
//     (hand-mirroring the first, to decide which retains are safe to demote)
//
// `list_retains_for_expr`'s shapes were never mirrored by borrow_inference,
// which is exactly why a `List` read out via `Index` (e.g. `model.params[0]`)
// goes un-demoted and is a documented open hazard (see
// `test_list_indexed_tensor_alias_gets_drop_capstone_design_constraint` in
// tests.rs). Both CTMM's emission pass (`ctmm.rs`) and the borrow-demotion
// pass (`borrow_inference.rs`) now consult `retain_sites` instead of
// re-deriving alias shapes independently.
//
// `retain_sites` is exhaustive over `TypedExprKind`: every shape is named,
// including an explicit no-retain arm for `Index` — so a future retain shape
// is a single-point edit, and the open `Index` hazard is visible in one
// place rather than absent from three.

use crate::typed_ir::{TypedExpr, TypedExprKind};

/// Which runtime refcount primitive a retain site needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainKind {
    /// `tensor_retain` — a grad-tracked scalar `Tensor` handle.
    Tensor,
    /// `aggregate_retain` — a `List<T>` box (structural, not grad-tracked;
    /// ADR-0034).
    ListAgg,
}

/// The syntactic shape of the alias that produced a retain site.
/// `borrow_inference` applies a different demotion-safety rule per shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasShape {
    /// `let b = a` — bare identifier alias.
    Ident,
    /// `let t = v.data` — `.data` detach-bind.
    DataField,
    /// An `Ident` field inside a `StructInit { .. }` construction.
    StructField,
    /// An `Ident` element inside an `ArrayLiteral [ .. ]` construction.
    ArrayElem,
    /// `base[index]` — e.g. `model.params[0]`. Never a retain site today
    /// (M29's intraprocedural borrow-inference doesn't classify it either
    /// way — see ADR-0026 "Out of Scope"); named explicitly so the gap is
    /// visible here instead of a silent absence. Never constructed by
    /// `retain_sites` (the `Index` arm below produces no site at all) — kept
    /// as a documented, reachable variant for when interprocedural analysis
    /// lifts this restriction.
    #[allow(dead_code)]
    Index,
}

/// One alias that CTMM retains (or, for `AliasShape::Index`, deliberately
/// does not) when `expr` is bound or stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainSite {
    /// The binding whose refcount is bumped — the name that appears in the
    /// emitted `Retain`/`RetainAgg` statement.
    pub source: String,
    pub shape: AliasShape,
    pub kind: RetainKind,
}

/// Exhaustive over `TypedExprKind`. Returns every alias this expression
/// introduces a retain-worthy reference to, given `expr` is the RHS of a
/// `Let`/`LetTuple`/`Assign`/`Return` statement.
///
/// Tensor sites additionally require `grad_tracked` (a non-grad-tracked
/// tensor alias needs no RC: CTMM's static-drop-only path covers it, ADR-0026
/// D6). `List` sites are structural — required whenever the expression is
/// `List`-typed, regardless of `grad_tracked` (ADR-0034).
pub fn retain_sites(expr: &TypedExpr) -> Vec<RetainSite> {
    match &expr.kind {
        TypedExprKind::Ident(name) => {
            let mut out = Vec::new();
            if expr.grad_tracked && expr.ty.is_tensor() {
                out.push(RetainSite {
                    source: name.clone(),
                    shape: AliasShape::Ident,
                    kind: RetainKind::Tensor,
                });
            }
            if expr.ty.is_list() {
                out.push(RetainSite {
                    source: name.clone(),
                    shape: AliasShape::Ident,
                    kind: RetainKind::ListAgg,
                });
            }
            out
        }
        TypedExprKind::FieldAccess { base, field } if field == "data" => {
            if base.grad_tracked && base.ty.is_tensor() {
                if let TypedExprKind::Ident(n) = &base.kind {
                    return vec![RetainSite {
                        source: n.clone(),
                        shape: AliasShape::DataField,
                        kind: RetainKind::Tensor,
                    }];
                }
            }
            vec![]
        }
        TypedExprKind::FieldAccess { .. } => vec![],
        TypedExprKind::ArrayLiteral { elements } => elements
            .iter()
            .filter(|e| e.grad_tracked && e.ty.is_tensor())
            .filter_map(|e| match &e.kind {
                TypedExprKind::Ident(n) => Some(RetainSite {
                    source: n.clone(),
                    shape: AliasShape::ArrayElem,
                    kind: RetainKind::Tensor,
                }),
                _ => None,
            })
            .collect(),
        TypedExprKind::StructInit { fields, .. } => fields
            .iter()
            .filter_map(|f| match &f.kind {
                TypedExprKind::Ident(n) => {
                    if f.grad_tracked && f.ty.is_tensor() {
                        Some(RetainSite {
                            source: n.clone(),
                            shape: AliasShape::StructField,
                            kind: RetainKind::Tensor,
                        })
                    } else if f.ty.is_list() {
                        Some(RetainSite {
                            source: n.clone(),
                            shape: AliasShape::StructField,
                            kind: RetainKind::ListAgg,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect(),
        // `base[index]` — e.g. `model.params[0]`. Not a recognized retain
        // shape (see `AliasShape::Index` doc comment): named explicitly,
        // never produces a site.
        TypedExprKind::Index { .. } => vec![],
        // No other expression shape reuses an existing handle rather than
        // producing a fresh allocation/value.
        TypedExprKind::Lit(_)
        | TypedExprKind::BinOp { .. }
        | TypedExprKind::Unary { .. }
        | TypedExprKind::Call { .. }
        | TypedExprKind::KernelCall { .. }
        | TypedExprKind::TensorLiteral { .. }
        | TypedExprKind::EnumInit { .. }
        | TypedExprKind::TupleInit { .. }
        | TypedExprKind::TupleIndex { .. }
        | TypedExprKind::KernelLaunch { .. } => vec![],
    }
}
