use crate::span::Span;

// ── Scalar types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarTy {
    F32, F16, Bf16,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
}

// ── Tensor placement ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    Cpu,
    Gpu,
}

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Tensor { dtype: ScalarTy },
    Scalar(ScalarTy),
    Bool,
    Tuple(Vec<Ty>),
    Named(String),
}

// ── Operators ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Matmul,
    Eq, NotEq, Lt, LtEq, Gt, GtEq,
    And, Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

// ── Literals ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

// ── Expressions ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Lit(Lit),
    Ident(String),
    BinOp { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Unary { op: UnaryOp, operand: Box<Expr> },
    Call { callee: Box<Expr>, args: Vec<Expr> },
    Index { base: Box<Expr>, indices: Vec<Expr> },
    TensorLiteral { placement: Placement, dtype: ScalarTy, elements: Vec<Expr> },
    FieldAccess { base: Box<Expr>, field: String },
}

// ── Statements ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Let { name: String, expr: Expr },
    Return { expr: Expr },
    Expr(Expr),
}

// ── Parameters ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelParam {
    pub inout: bool,
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

// ── Module paths ──────────────────────────────────────────────────────────────

/// A dot-separated module path, e.g. `models.transformer`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModulePath {
    pub segments: Vec<String>,
    pub span: Span,
}

impl ModulePath {
    /// The module's short name — the last segment.
    pub fn name(&self) -> &str {
        self.segments.last().map(String::as_str).unwrap_or("")
    }
}

// ── Top-level items ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    Fn {
        name: String,
        params: Vec<Param>,
        return_ty: Option<Ty>,
        body: Vec<Stmt>,
    },
    Kernel {
        name: String,
        params: Vec<KernelParam>,
        return_ty: Ty,
        body: Vec<Stmt>,
    },
    /// `import models.transformer`
    Import {
        path: ModulePath,
    },
    /// `from ops import add, mul`
    FromImport {
        path: ModulePath,
        names: Vec<(String, Span)>,
    },
}

// ── Program ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}
