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

    m
}
