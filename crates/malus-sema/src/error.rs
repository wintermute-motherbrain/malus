use malus_syntax::Span;
use crate::ty::ResolvedTy;
use std::fmt;

#[derive(Debug, Clone)]
pub enum SemaError {
    TypeMismatch { expected: ResolvedTy, found: ResolvedTy, span: Span },
    DtypeMismatch { lhs: String, rhs: String, span: Span },
    PlacementMismatch { lhs: String, rhs: String, span: Span },
    UnknownIdent { name: String, span: Span },
    ArgCountMismatch { callee: String, expected: usize, found: usize, span: Span },
    ReturnTypeMismatch { expected: ResolvedTy, found: ResolvedTy, span: Span },
    DuplicateDefinition { name: String, first: Span, second: Span },
    NotAFunction { name: String, span: Span },
    MainNotFound,
    LossyCoercion { from: String, to: String, span: Span },
    KernelCalledFromKernel { name: String, span: Span },
    UnknownType { name: String, span: Span },
}

impl fmt::Display for SemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemaError::TypeMismatch { expected, found, .. } =>
                write!(f, "type mismatch: expected {}, found {}", expected, found),
            SemaError::DtypeMismatch { lhs, rhs, .. } =>
                write!(f, "dtype mismatch: {} vs {}", lhs, rhs),
            SemaError::PlacementMismatch { lhs, rhs, .. } =>
                write!(f, "placement mismatch: {} tensor vs {} tensor", lhs, rhs),
            SemaError::UnknownIdent { name, .. } =>
                write!(f, "unknown identifier '{}'", name),
            SemaError::ArgCountMismatch { callee, expected, found, .. } =>
                write!(f, "'{}' expects {} argument(s), got {}", callee, expected, found),
            SemaError::ReturnTypeMismatch { expected, found, .. } =>
                write!(f, "return type mismatch: declared {}, got {}", expected, found),
            SemaError::DuplicateDefinition { name, .. } =>
                write!(f, "duplicate definition of '{}'", name),
            SemaError::NotAFunction { name, .. } =>
                write!(f, "'{}' is not a function or kernel", name),
            SemaError::MainNotFound =>
                write!(f, "no 'fn main()' entry point found"),
            SemaError::LossyCoercion { from, to, .. } =>
                write!(f, "lossy coercion from {} to {} is not allowed", from, to),
            SemaError::KernelCalledFromKernel { name, .. } =>
                write!(f, "kernel '{}' cannot be called from inside another kernel", name),
            SemaError::UnknownType { name, .. } =>
                write!(f, "unknown type '{}'", name),
        }
    }
}
