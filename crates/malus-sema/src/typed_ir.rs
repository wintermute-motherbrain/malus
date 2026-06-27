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
    /// Reassignment of a `let mut` binding. The old value is dropped by CTMM
    /// before this stmt executes; this stmt performs a pure rebind.
    Assign { name: String, expr: TypedExpr },
    Return { expr: TypedExpr },
    Expr(TypedExpr),
    /// CTMM: free this binding's tensor allocation.
    Drop { name: String },
    /// CTMM: CPU barrier — wait for in-flight GPU work before freeing.
    GpuBarrier,
    // ── Control flow (M9) ─────────────────────────────────────────────────────
    /// `if condition: then_body [else: else_body]`
    ///
    /// Bindings introduced inside either branch do not escape into the outer
    /// scope.  CTMM treats the whole `If` node as an opaque use site for outer
    /// bindings (see ADR-0014).
    If {
        condition: TypedExpr,
        then_body: Vec<TypedStmt>,
        else_body: Option<Vec<TypedStmt>>,
    },
    /// `for var in range(start, end): body`
    ///
    /// The loop variable is `Scalar(I64)`, scoped to `body`.  CTMM annotates
    /// `body` independently; loop-local tensors get their `Drop` nodes inside
    /// the body (fired on every iteration at runtime).
    For {
        var: String,
        start: TypedExpr,
        end: TypedExpr,
        body: Vec<TypedStmt>,
    },
    /// `while condition: body`
    While { condition: TypedExpr, body: Vec<TypedStmt> },
    // ── M10 readiness: reference-counted tensor nodes ─────────────────────────
    //
    // M9's CTMM emits **zero** Retain/Release nodes — the hierarchical Drop
    // placement is sufficient for if/for/while as statements (see ADR-0014).
    // The nodes exist now so M10 can generate them for struct-field tensors
    // without touching the runtime ABI or typed IR again.
    /// Increment the tensor's reference count (`tensor_retain`). Not emitted by M9.
    Retain { name: String },
    /// Decrement the tensor's reference count; frees when it reaches zero
    /// (`tensor_release`). Not emitted by M9.
    Release { name: String },
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
