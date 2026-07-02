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
// went un-demoted and was a documented open hazard. M34 closed that hazard:
// the `Index` arm now produces a `RetainTarget::Binding` site (the container
// owns the element; a *bound* element read needs its own reference, bumped on
// the new binding right after the bind — see ctmm.rs's
// `insert_container_read_retains`). Both CTMM's emission passes (`ctmm.rs`)
// and the borrow-demotion pass (`borrow_inference.rs`) consult `retain_sites`
// instead of re-deriving alias shapes independently.
//
// `retain_sites` is exhaustive over `TypedExprKind`: every shape is named, so
// a future retain shape is a single-point edit.

use crate::ty::ResolvedTy;
use crate::typed_ir::{TypedExpr, TypedExprKind};

/// Which runtime refcount primitive a retain site needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainKind {
    /// `tensor_retain` — a scalar `Tensor` handle.
    Tensor,
    /// `aggregate_retain` — an ARC-headed aggregate box: `List<T>` (ADR-0034),
    /// struct, enum, or tuple. (Fixed `Array`s are headerless and never take
    /// this kind.)
    Agg,
}

/// Where the emitted retain lands, relative to the statement whose RHS
/// produced the site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetainTarget {
    /// Retain the named in-scope source binding, inserted BEFORE the
    /// statement (the classic alias shapes: the source's own drop and the new
    /// reference's drop must both be covered).
    Source(String),
    /// Retain the statement's own new binding, inserted AFTER the statement.
    /// Used for container-element reads (`base[i]`): the container owns the
    /// element and there is no source *name* to retain — the new binding
    /// needs its own reference so its eventual Drop doesn't steal the
    /// container's (M34 done-when #0, bug (b)).
    Binding,
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
    /// `base[index]` — e.g. `model.params[0]`. A `RetainTarget::Binding`
    /// site: never demoted by borrow_inference (the container's ownership of
    /// the element is real and independent of any intraprocedural liveness
    /// argument).
    Index,
}

/// One alias that CTMM retains when `expr` is bound or stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainSite {
    /// Whose refcount is bumped, and on which side of the statement.
    pub target: RetainTarget,
    pub shape: AliasShape,
    pub kind: RetainKind,
}

fn source_site(name: &str, shape: AliasShape, kind: RetainKind) -> RetainSite {
    RetainSite { target: RetainTarget::Source(name.to_string()), shape, kind }
}

/// Exhaustive over `TypedExprKind`. Returns every alias this expression
/// introduces a retain-worthy reference to, given `expr` is the RHS of a
/// `Let`/`LetTuple`/`Assign`/`Return` statement.
///
/// Tensor `Source` sites additionally require `grad_tracked` (a
/// non-grad-tracked tensor alias needs no RC: CTMM's static-drop-only path
/// covers it, ADR-0026 D6). `Agg` sites and `Index` element sites are
/// structural — required regardless of `grad_tracked` (ADR-0034; for `Index`,
/// the container's element release at its own drop is unconditional, so the
/// binding's reference must be real even for non-grad-tracked tensors).
pub fn retain_sites(expr: &TypedExpr) -> Vec<RetainSite> {
    match &expr.kind {
        TypedExprKind::Ident(name) => {
            let mut out = Vec::new();
            if expr.grad_tracked && expr.ty.is_tensor() {
                out.push(source_site(name, AliasShape::Ident, RetainKind::Tensor));
            }
            if expr.ty.is_list() {
                out.push(source_site(name, AliasShape::Ident, RetainKind::Agg));
            }
            out
        }
        TypedExprKind::FieldAccess { base, field } if field == "data" => {
            if base.grad_tracked && base.ty.is_tensor() {
                if let TypedExprKind::Ident(n) = &base.kind {
                    return vec![source_site(n, AliasShape::DataField, RetainKind::Tensor)];
                }
            }
            vec![]
        }
        TypedExprKind::FieldAccess { .. } => vec![],
        TypedExprKind::ArrayLiteral { elements } => elements
            .iter()
            .filter(|e| e.grad_tracked && e.ty.is_tensor())
            .filter_map(|e| match &e.kind {
                TypedExprKind::Ident(n) => {
                    Some(source_site(n, AliasShape::ArrayElem, RetainKind::Tensor))
                }
                _ => None,
            })
            .collect(),
        TypedExprKind::StructInit { fields, .. } => fields
            .iter()
            .filter_map(|f| match &f.kind {
                TypedExprKind::Ident(n) => {
                    if f.grad_tracked && f.ty.is_tensor() {
                        Some(source_site(n, AliasShape::StructField, RetainKind::Tensor))
                    } else if f.ty.is_list() {
                        Some(source_site(n, AliasShape::StructField, RetainKind::Agg))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect(),
        // `base[index]` — e.g. `model.params[0]` (M34). The container owns
        // the element; binding it steals that reference unless the binding is
        // given its own. `.shape[i]`/`.strides[i]` reads fuse to `tensor_dim`
        // in codegen and are scalar-typed, so they fall through the type
        // filter naturally. Buffer/scalar elements need no RC.
        TypedExprKind::Index { base, .. } => {
            let container = matches!(base.ty, ResolvedTy::Array { .. } | ResolvedTy::List { .. });
            if !container {
                return vec![];
            }
            if expr.ty.is_tensor() {
                vec![RetainSite {
                    target: RetainTarget::Binding,
                    shape: AliasShape::Index,
                    kind: RetainKind::Tensor,
                }]
            } else if expr.ty.is_list() || expr.ty.is_struct() || expr.ty.is_enum() {
                vec![RetainSite {
                    target: RetainTarget::Binding,
                    shape: AliasShape::Index,
                    kind: RetainKind::Agg,
                }]
            } else {
                vec![]
            }
        }
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
