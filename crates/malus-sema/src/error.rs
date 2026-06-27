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
    FormatArgCountMismatch { callee: String, placeholders: usize, args: usize, span: Span },
    StringLiteralOutsidePrint { span: Span },
    AssignToImmutable { name: String, span: Span },
    // ── M10: aggregate types ─────────────────────────────────────────────────
    UnknownField { struct_name: String, field: String, span: Span },
    UnknownVariant { enum_name: String, variant: String, span: Span },
    NonExhaustiveMatch { enum_name: String, missing: Vec<String>, span: Span },
    DuplicateMatchArm { variant: String, span: Span },
    MatchWildcard { span: Span },
    MatchArmArityMismatch { variant: String, expected: usize, found: usize, span: Span },
    MissingField { struct_name: String, field: String, span: Span },
    UnknownConstructorField { struct_name: String, field: String, span: Span },
    DuplicateTypeDefinition { name: String, first: Span, second: Span },
    MatchScrutineeNotEnum { found: String, span: Span },
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
            SemaError::FormatArgCountMismatch { callee, placeholders, args, .. } =>
                write!(f, "'{}' format string has {} placeholder(s) but {} value arg(s) provided", callee, placeholders, args),
            SemaError::StringLiteralOutsidePrint { .. } =>
                write!(f, "string literals are only valid as the first argument of print/println"),
            SemaError::AssignToImmutable { name, .. } =>
                write!(f, "cannot assign to '{}' because it is not declared `let mut`", name),
            SemaError::UnknownField { struct_name, field, .. } =>
                write!(f, "struct '{}' has no field '{}'", struct_name, field),
            SemaError::UnknownVariant { enum_name, variant, .. } =>
                write!(f, "enum '{}' has no variant '{}'", enum_name, variant),
            SemaError::NonExhaustiveMatch { enum_name, missing, .. } =>
                write!(f, "non-exhaustive match on '{}': missing variants {:?}", enum_name, missing),
            SemaError::DuplicateMatchArm { variant, .. } =>
                write!(f, "duplicate match arm for variant '{}'", variant),
            SemaError::MatchWildcard { .. } =>
                write!(f, "wildcard '_' arms are not supported in V1 match — list every variant explicitly"),
            SemaError::MatchArmArityMismatch { variant, expected, found, .. } =>
                write!(f, "variant '{}' has {} field(s) but arm binds {}", variant, expected, found),
            SemaError::MissingField { struct_name, field, .. } =>
                write!(f, "struct '{}' requires field '{}' but it was not provided", struct_name, field),
            SemaError::UnknownConstructorField { struct_name, field, .. } =>
                write!(f, "struct '{}' has no field '{}'", struct_name, field),
            SemaError::DuplicateTypeDefinition { name, .. } =>
                write!(f, "duplicate type definition '{}'", name),
            SemaError::MatchScrutineeNotEnum { found, .. } =>
                write!(f, "match scrutinee must be an enum type, got {}", found),
        }
    }
}
