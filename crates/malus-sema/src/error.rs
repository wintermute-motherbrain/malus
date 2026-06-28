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
    // ── M11: diagnostics ─────────────────────────────────────────────────────
    TensorShapeMismatch { expected: usize, found: usize, span: Span },
    // ── M12: hardening ───────────────────────────────────────────────────────
    BreakOutsideLoop { span: Span },
    ContinueOutsideLoop { span: Span },
    // ── M13.5: tuples ────────────────────────────────────────────────────────
    /// Tuple element type is itself a tuple (flat-only rule).
    NestedTuple { span: Span },
    /// Tuple type used as a struct field type (not allowed).
    TupleInStructField { struct_name: String, field: String, span: Span },
    /// Tuple type used as an array element type (not allowed).
    TupleInArrayElement { span: Span },
    /// `let (a, b, ...) = expr` arity mismatch.
    TupleDestructureArity { expected: usize, found: usize, span: Span },
    /// RHS of `let (a, b) = ...` is not a tuple.
    TupleDestructureNotTuple { found: String, span: Span },
    /// `x.N` where N is out of range for the tuple's arity.
    TupleIndexOutOfRange { len: usize, index: usize, span: Span },
    /// `x.N` where `x` is not a tuple.
    TupleIndexNotTuple { found: String, span: Span },
    /// Tuples with fewer than 2 elements are not allowed.
    TupleTooShort { span: Span },
    // ── M14: tape control ────────────────────────────────────────────────────
    /// `return`/`break`/`continue` inside a `with no_grad:` body would skip
    /// `tape_resume()`. Rejected in M14 (D6).
    EarlyExitInNoGrad { span: Span },
}

impl SemaError {
    pub fn primary_span(&self) -> Option<Span> {
        use SemaError::*;
        match self {
            TypeMismatch { span, .. }
            | DtypeMismatch { span, .. }
            | PlacementMismatch { span, .. }
            | UnknownIdent { span, .. }
            | ArgCountMismatch { span, .. }
            | ReturnTypeMismatch { span, .. }
            | NotAFunction { span, .. }
            | LossyCoercion { span, .. }
            | KernelCalledFromKernel { span, .. }
            | UnknownType { span, .. }
            | FormatArgCountMismatch { span, .. }
            | StringLiteralOutsidePrint { span }
            | AssignToImmutable { span, .. }
            | UnknownField { span, .. }
            | UnknownVariant { span, .. }
            | NonExhaustiveMatch { span, .. }
            | DuplicateMatchArm { span, .. }
            | MatchWildcard { span }
            | MatchArmArityMismatch { span, .. }
            | MissingField { span, .. }
            | UnknownConstructorField { span, .. }
            | MatchScrutineeNotEnum { span, .. }
            | TensorShapeMismatch { span, .. }
            | BreakOutsideLoop { span }
            | ContinueOutsideLoop { span }
            | NestedTuple { span }
            | TupleInStructField { span, .. }
            | TupleInArrayElement { span }
            | TupleDestructureArity { span, .. }
            | TupleDestructureNotTuple { span, .. }
            | TupleIndexOutOfRange { span, .. }
            | TupleIndexNotTuple { span, .. }
            | TupleTooShort { span }
            | EarlyExitInNoGrad { span } => Some(*span),
            DuplicateDefinition { second, .. } | DuplicateTypeDefinition { second, .. } => Some(*second),
            MainNotFound => None,
        }
    }

    pub fn secondary_span(&self) -> Option<Span> {
        use SemaError::*;
        match self {
            DuplicateDefinition { first, .. } | DuplicateTypeDefinition { first, .. } => Some(*first),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        use SemaError::*;
        match self {
            TypeMismatch { .. } => "type mismatch here",
            DtypeMismatch { .. } => "dtype mismatch in this expression",
            PlacementMismatch { .. } => "placement mismatch here",
            UnknownIdent { .. } => "not found in scope",
            ArgCountMismatch { .. } => "wrong number of arguments",
            ReturnTypeMismatch { .. } => "this expression has the wrong type",
            NotAFunction { .. } => "not a function or kernel",
            LossyCoercion { .. } => "lossy conversion not allowed",
            KernelCalledFromKernel { .. } => "kernel called from kernel",
            UnknownType { .. } => "unknown type",
            FormatArgCountMismatch { .. } => "wrong number of {} placeholders",
            StringLiteralOutsidePrint { .. } => "string literal here",
            AssignToImmutable { .. } => "cannot assign to immutable binding",
            UnknownField { .. } => "no such field",
            UnknownVariant { .. } => "no such variant",
            NonExhaustiveMatch { .. } => "match is not exhaustive",
            DuplicateMatchArm { .. } => "duplicate arm",
            MatchWildcard { .. } => "wildcard not allowed",
            MatchArmArityMismatch { .. } => "wrong number of bindings in this arm",
            MissingField { .. } => "this field is required",
            UnknownConstructorField { .. } => "no such field in this struct",
            DuplicateDefinition { .. } => "defined again here",
            DuplicateTypeDefinition { .. } => "defined again here",
            MatchScrutineeNotEnum { .. } => "not an enum type",
            TensorShapeMismatch { .. } => "tensor shape mismatch",
            BreakOutsideLoop { .. } => "break outside loop",
            ContinueOutsideLoop { .. } => "continue outside loop",
            NestedTuple { .. } => "nested tuple not allowed (flat-only)",
            TupleInStructField { .. } => "tuple type not allowed as struct field",
            TupleInArrayElement { .. } => "tuple type not allowed as array element",
            TupleDestructureArity { .. } => "wrong number of bindings in let destructuring",
            TupleDestructureNotTuple { .. } => "right-hand side is not a tuple",
            TupleIndexOutOfRange { .. } => "tuple index out of range",
            TupleIndexNotTuple { .. } => "not a tuple type",
            TupleTooShort { .. } => "tuple must have at least 2 elements",
            EarlyExitInNoGrad { .. } => "early exit inside no_grad block",
            MainNotFound => "",
        }
    }

    pub fn note(&self) -> Option<&'static str> {
        use SemaError::*;
        match self {
            AssignToImmutable { .. } => Some("change `let` to `let mut` to allow reassignment"),
            LossyCoercion { .. } => Some("widen the narrower type explicitly — implicit narrowing is not allowed"),
            DtypeMismatch { .. } => Some("ensure both sides of the operation use the same dtype"),
            PlacementMismatch { .. } => Some("ensure both tensors are on the same device"),
            KernelCalledFromKernel { .. } => Some("kernels can only be called from fn bodies"),
            FormatArgCountMismatch { .. } => Some("add or remove {} placeholders to match the number of value arguments"),
            StringLiteralOutsidePrint { .. } => Some("use println(\"{}\", value) to print a non-string value"),
            NonExhaustiveMatch { .. } => Some("list all variants explicitly — wildcard _ arms are not supported in V1"),
            EarlyExitInNoGrad { .. } => Some("move the early exit outside the `with no_grad:` block"),
            MatchWildcard { .. } => Some("list each variant explicitly instead of using _"),
            MissingField { .. } => Some("all fields must be provided in struct literals"),
            _ => None,
        }
    }
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
            SemaError::TensorShapeMismatch { expected, found, .. } =>
                write!(f, "tensor literal declares shape with {} element(s) but {} value(s) provided", expected, found),
            SemaError::BreakOutsideLoop { .. } =>
                write!(f, "`break` is only valid inside a loop body"),
            SemaError::ContinueOutsideLoop { .. } =>
                write!(f, "`continue` is only valid inside a loop body"),
            SemaError::NestedTuple { .. } =>
                write!(f, "tuple element types may not themselves be tuples (flat-only rule)"),
            SemaError::TupleInStructField { struct_name, field, .. } =>
                write!(f, "struct '{}' field '{}' may not have a tuple type", struct_name, field),
            SemaError::TupleInArrayElement { .. } =>
                write!(f, "array element type may not be a tuple"),
            SemaError::TupleDestructureArity { expected, found, .. } =>
                write!(f, "tuple has {} element(s) but {} binding(s) provided", expected, found),
            SemaError::TupleDestructureNotTuple { found, .. } =>
                write!(f, "cannot destructure: expected a tuple, got {}", found),
            SemaError::TupleIndexOutOfRange { len, index, .. } =>
                write!(f, "tuple index {} is out of range for a {}-element tuple", index, len),
            SemaError::TupleIndexNotTuple { found, .. } =>
                write!(f, "cannot index with `.N`: expected a tuple, got {}", found),
            SemaError::TupleTooShort { .. } =>
                write!(f, "tuples must have at least 2 elements"),
            SemaError::EarlyExitInNoGrad { .. } =>
                write!(f, "`return`/`break`/`continue` may not cross a `with no_grad:` boundary"),
        }
    }
}
