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
    /// Axis-only: one tensor/variable positional arg + required named `axis=i32`.
    /// No keepdim.  Normalized to positional [tensor, axis] in check_call.
    /// Variable propagation: if arg0 is Variable the return type is Variable.
    AxisOnly,
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

    // randn(d0, d1, ...) -> Tensor<f32> on GPU; Philox4x32-10 + Box-Muller; non-differentiable.
    m.insert("randn".to_string(), BuiltinSig {
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

    // M18 transformer stdlib
    // softmax(t, axis=N) — normalized exponentials over axis; same shape as input.
    // layernorm(t, axis=N) — (x−μ)/σ over axis; no affine; same shape as input.
    for name in &["softmax", "layernorm"] {
        m.insert(name.to_string(), BuiltinSig {
            kind: BuiltinKind::AxisOnly,
            return_ty: tensor_f32.clone(),
            return_placement: Some(Placement::Gpu),
        });
    }

    // gelu(t) — elementwise tanh-approx GELU; like relu/sigmoid, Variable propagates.
    m.insert("gelu".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![tensor_f32.clone()]),
        return_ty: tensor_f32.clone(),
        return_placement: Some(Placement::Gpu),
    });

    // cross_entropy(logits: Variable<f32>, targets: Tensor<i32|i64>) -> Variable<f32>
    // targets accept both i32 and i64; the hint guides to i32 (M19 done-when dtype).
    // Always returns Variable (always recorded on tape).
    let var_f32   = ResolvedTy::Variable { dtype: ScalarTy::F32 };
    let tensor_i32 = ResolvedTy::Tensor { dtype: ScalarTy::I32 };
    m.insert("cross_entropy".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![var_f32.clone(), tensor_i32.clone()]),
        return_ty: var_f32.clone(),
        return_placement: Some(Placement::Gpu),
    });

    // embedding(weight: Variable<f32>, indices: Tensor<i32|i64>) -> Variable<f32>
    // weight is [V, D]; indices is [T]; output is [T, D].
    // Both i32 and i64 are valid index dtypes; hint guides to i32.
    m.insert("embedding".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![var_f32.clone(), tensor_i32]),
        return_ty: var_f32,
        return_placement: Some(Placement::Gpu),
    });

    // causal_mask(T: i64) -> Tensor<f32>  — [T, T] mask; non-differentiable.
    m.insert("causal_mask".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Scalar(ScalarTy::I64)]),
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

    // M22 string I/O.
    // read_file(path: str) -> str
    m.insert("read_file".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Str]),
        return_ty: ResolvedTy::Str,
        return_placement: None,
    });
    // str_len(s: str) -> i64  (byte length; matches malus integer default I64)
    m.insert("str_len".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Str]),
        return_ty: ResolvedTy::Scalar(ScalarTy::I64),
        return_placement: None,
    });
    // str_char_at(s: str, i: i64) -> i64  (Unicode codepoint at position i)
    // i64 so that loop counters (let mut i = 0) work without explicit casts.
    m.insert("str_char_at".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Str, ResolvedTy::Scalar(ScalarTy::I64)]),
        return_ty: ResolvedTy::Scalar(ScalarTy::I64),
        return_placement: None,
    });
    // str_from_char(c: i64) -> str  (encode a Unicode codepoint as a str)
    m.insert("str_from_char".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Scalar(ScalarTy::I64)]),
        return_ty: ResolvedTy::Str,
        return_placement: None,
    });

    // M22 rand_uniform() -> f32  — Philox4x32-10; non-differentiable.
    m.insert("rand_uniform".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![]),
        return_ty: ResolvedTy::Scalar(ScalarTy::F32),
        return_placement: None,
    });
    // M22 rand_int(n: i64) -> i64  — uniform random integer in [0, n).
    m.insert("rand_int".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Scalar(ScalarTy::I64)]),
        return_ty: ResolvedTy::Scalar(ScalarTy::I64),
        return_placement: None,
    });

    // M22 Buffer<i32> — mutable CPU-side staging buffer for tokenization.
    // buffer_i32(n: i64) -> Buffer<i32>
    m.insert("buffer_i32".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Scalar(ScalarTy::I64)]),
        return_ty: ResolvedTy::Buffer { dtype: ScalarTy::I32 },
        return_placement: None,
    });
    // freeze(buf: Buffer<i32>) -> Tensor<i32>
    m.insert("freeze".to_string(), BuiltinSig {
        kind: BuiltinKind::Fixed(vec![ResolvedTy::Buffer { dtype: ScalarTy::I32 }]),
        return_ty: ResolvedTy::Tensor { dtype: ScalarTy::I32 },
        return_placement: Some(Placement::Gpu),
    });

    m
}
