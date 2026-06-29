use std::collections::HashMap;
use malus_syntax::ast::{Placement, ScalarTy};
use crate::ty::ResolvedTy;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum BuiltinKind {
    /// Fixed arity with typed params.
    Fixed(Vec<ResolvedTy>),
    /// Accepts any number of tensors or string literals (e.g., print).
    Variadic,
    /// Accepts any number of i64 scalars as shape args (e.g., zeros, ones).
    ShapeArgs,
    /// Accepts any number of args, each constrained to a single type (e.g., zero_grad).
    VariadicTyped(ResolvedTy),
    /// Axis reduction: one tensor + optional named args `axis=i32`, `keepdim=bool`.
    /// Normalized to positional [tensor, axis, keepdim] in check_call.
    /// For `sum` the axis arg is optional (no axis = whole-tensor sum, unchanged).
    Reduction,
    /// Tensor-then-shape-args: first arg is a tensor/variable, remaining args are
    /// i64 dim/axis values.  Used by reshape(t, d0..dn), transpose(t[, i, j]),
    /// permute(t, p0..pn).  Normalized to positional [tensor, d0..dn] in check_call.
    /// Variable propagation: if arg0 is Variable the return type is Variable.
    TensorThenShapeArgs,
}

#[derive(Debug, Clone)]
pub struct BuiltinSig {
    pub kind: BuiltinKind,
    pub return_ty: ResolvedTy,
    pub return_placement: Option<Placement>,
}

pub fn register_builtins() -> HashMap<String, BuiltinSig> {
    let mut m = HashMap::new();

    // print(a, ...) — variadic, returns Unit; no trailing newline
    m.insert("print".to_string(), BuiltinSig {
        kind: BuiltinKind::Variadic,
        return_ty: ResolvedTy::Unit,
        return_placement: None,
    });

    // println(a, ...) — like print but appends a newline
    m.insert("println".to_string(), BuiltinSig {
        kind: BuiltinKind::Variadic,
        return_ty: ResolvedTy::Unit,
        return_placement: None,
    });

    // zeros(d0, d1, ...) -> Tensor<f32> on GPU
    m.insert("zeros".to_string(), BuiltinSig {
        kind: BuiltinKind::ShapeArgs,
        return_ty: ResolvedTy::Tensor { dtype: ScalarTy::F32 },
        return_placement: Some(Placement::Gpu),
    });

    // ones(d0, d1, ...) -> Tensor<f32> on GPU
    m.insert("ones".to_string(), BuiltinSig {
        kind: BuiltinKind::ShapeArgs,
        return_ty: ResolvedTy::Tensor { dtype: ScalarTy::F32 },
        return_placement: Some(Placement::Gpu),
    });

    // Unary math builtins — dispatched as built-in GPU kernels (one tensor in, same-shape tensor out).
    // return_placement: Some(Gpu) is load-bearing: CTMM marks results pending so barriers are
    // inserted before any CPU read (e.g. tensor_print after exp(x)).
    let tensor_f32 = ResolvedTy::Tensor { dtype: ScalarTy::F32 };
    for name in &["relu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs"] {
        m.insert(name.to_string(), BuiltinSig {
            kind: BuiltinKind::Fixed(vec![tensor_f32.clone()]),
            return_ty: tensor_f32.clone(),
            return_placement: Some(Placement::Gpu),
        });
    }

    // transpose(t[, i, j]) — swap two axes (or reverse a 2-D tensor with no args).
    // permute(t, p0..p_rank) — reorder all axes.
    // reshape(t, d0..dn)    — zero-copy contiguous reshape.
    // All: TensorThenShapeArgs; Variable input propagates to Variable output.
    for name in &["transpose", "permute", "reshape"] {
        m.insert(name.to_string(), BuiltinSig {
            kind: BuiltinKind::TensorThenShapeArgs,
            return_ty: tensor_f32.clone(),
            return_placement: Some(Placement::Gpu),
        });
    }

    // sum(t) — whole-tensor sum (returns [1] tensor) OR axis reduction (see Reduction kind).
    // mean/max/var require axis=N.
    m.insert("sum".to_string(), BuiltinSig {
        kind: BuiltinKind::Reduction,
        return_ty: tensor_f32.clone(),
        return_placement: Some(Placement::Gpu),
    });
    for name in &["mean", "max", "var"] {
        m.insert(name.to_string(), BuiltinSig {
            kind: BuiltinKind::Reduction,
            return_ty: tensor_f32.clone(),
            return_placement: Some(Placement::Gpu),
        });
    }

    // variable(t) -> Variable<f32> — wrap a Tensor in an RC Variable.
    m.insert("variable".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![tensor_f32.clone()]),
        return_ty: ResolvedTy::Variable { dtype: ScalarTy::F32 },
        return_placement: Some(Placement::Gpu),
    });

    // tensor_print(t) — print a tensor directly (alias for print with single tensor arg).
    m.insert("tensor_print".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![tensor_f32]),
        return_ty: ResolvedTy::Unit,
        return_placement: None,
    });

    // backward(loss: Variable<f32>) — walk the tape in reverse, accumulate grads, clear tape.
    m.insert("backward".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Variable { dtype: ScalarTy::F32 }]),
        return_ty: ResolvedTy::Unit,
        return_placement: None,
    });

    // zero_grad(v1, v2, ...) — clear accumulated grads for the given Variables.
    m.insert("zero_grad".to_string(), BuiltinSig {
        kind: BuiltinKind::VariadicTyped(ResolvedTy::Variable { dtype: ScalarTy::F32 }),
        return_ty: ResolvedTy::Unit,
        return_placement: None,
    });

    m
}
