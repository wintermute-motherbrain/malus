use malus_syntax::ast::{BinOp, Lit, Placement, ScalarTy, UnaryOp};
use malus_syntax::Span;
use crate::ty::ResolvedTy;

// ── Program ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypedProgram {
    pub fns: Vec<TypedFn>,
    pub kernels: Vec<TypedKernel>,
}

// ── Functions and Kernels ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypedParam {
    pub name: String,
    pub ty: ResolvedTy,
}

#[derive(Debug, Clone)]
pub struct TypedKernelParam {
    pub inout: bool,
    pub name: String,
    pub ty: ResolvedTy,
}

#[derive(Debug, Clone)]
pub struct TypedFn {
    pub name: String,
    pub params: Vec<TypedParam>,
    pub return_ty: ResolvedTy,
    pub body: Vec<TypedStmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TypedKernel {
    pub name: String,
    pub params: Vec<TypedKernelParam>,
    pub return_ty: ResolvedTy,
    pub body: Vec<TypedStmt>,
    pub span: Span,
}

// ── Statements ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum TypedStmt {
    Let { name: String, expr: TypedExpr },
    Return { expr: TypedExpr },
    Expr(TypedExpr),
    /// CTMM: free this binding's tensor allocation.
    Drop { name: String },
    /// CTMM: CPU barrier — wait for in-flight GPU work before freeing.
    GpuBarrier,
}

// ── Expressions ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypedExpr {
    pub kind: TypedExprKind,
    pub ty: ResolvedTy,
    /// Non-None only for tensor-typed expressions.
    pub placement: Option<Placement>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypedExprKind {
    Lit(Lit),
    Ident(String),
    BinOp {
        op: BinOp,
        lhs: Box<TypedExpr>,
        rhs: Box<TypedExpr>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<TypedExpr>,
    },
    /// Call to a user-defined `fn` or a builtin.
    Call {
        callee: String,
        args: Vec<TypedExpr>,
    },
    /// Call to a user-defined `kernel` — dispatched via Metal.
    KernelCall {
        callee: String,
        args: Vec<TypedExpr>,
        /// Binding names of tensor arguments that are now in-flight on the GPU.
        in_flight: Vec<String>,
    },
    Index {
        base: Box<TypedExpr>,
        indices: Vec<TypedExpr>,
    },
    TensorLiteral {
        placement: Placement,
        dtype: ScalarTy,
        elements: Vec<TypedExpr>,
    },
    FieldAccess {
        base: Box<TypedExpr>,
        field: String,
    },
}
