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

    // transpose(t) -> Tensor<f32> — eager CPU op, 2-D only in V1
    m.insert("transpose".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![tensor_f32.clone()]),
        return_ty: tensor_f32.clone(),
        return_placement: Some(Placement::Gpu),
    });

    // sum(t) -> Tensor<f32> — eager CPU op, returns a 1-element [1] tensor
    m.insert("sum".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![tensor_f32.clone()]),
        return_ty: tensor_f32.clone(),
        return_placement: Some(Placement::Gpu),
    });

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

    m
}
