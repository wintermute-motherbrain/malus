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
    Variable { dtype: ScalarTy },
    Scalar(ScalarTy),
    Bool,
    Tuple(Vec<Ty>),
    Named(String),
    /// `Array<T, N>` — fixed-length homogeneous array.
    Array { elem: Box<Ty>, len: usize },
}

// ── Operators ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

pub fn elementwise_builtin_name(op: &BinOp) -> Option<&'static str> {
    match op {
        BinOp::Add => Some("malus_add"),
        BinOp::Sub => Some("malus_sub"),
        BinOp::Mul => Some("malus_mul"),
        BinOp::Div => Some("malus_div"),
        BinOp::Matmul | BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq
        | BinOp::Gt | BinOp::GtEq | BinOp::And | BinOp::Or => None,
    }
}

/// Builtin kernel name for scalar-broadcast ops: `Tensor op Scalar` or `Scalar op Tensor`.
/// `scalar_on_right` = true means `tensor op scalar` (e.g. `a * 0.5`).
/// For commutative ops (Add, Mul), `scalar_on_right=false` canonicalises to the right form.
pub fn scalar_broadcast_builtin_name(op: &BinOp, scalar_on_right: bool) -> Option<&'static str> {
    match op {
        BinOp::Add => Some("malus_add_scalar"),
        BinOp::Sub => if scalar_on_right { Some("malus_sub_scalar") } else { Some("malus_rsub_scalar") },
        BinOp::Mul => Some("malus_mul_scalar"),
        BinOp::Div => if scalar_on_right { Some("malus_div_scalar") } else { Some("malus_rdiv_scalar") },
        _ => None,
    }
}

// ── Literals ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

// ── Aggregate type definitions ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariantDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
    pub span: Span,
}

// ── Call argument (supports keyword args for struct/enum constructors) ─────────

#[derive(Debug, Clone, PartialEq)]
pub struct CallArg {
    /// `Some(name)` for named args like `weights=w`; `None` for positional.
    pub name: Option<String>,
    pub value: Expr,
}

// ── Match arm ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub variant: String,
    /// Positional bindings: bind variant's fields by declaration order.
    pub bindings: Vec<String>,
    pub body: Vec<Stmt>,
    pub span: Span,
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
    /// Call or struct/enum constructor — args may be named (struct/enum) or
    /// positional (regular fn call).
    Call { callee: Box<Expr>, args: Vec<CallArg> },
    Index { base: Box<Expr>, indices: Vec<Expr> },
    TensorLiteral { placement: Placement, dtype: ScalarTy, elements: Vec<Expr>, shape: Vec<usize> },
    /// `[e1, e2, e3]` — fixed-length array literal.
    ArrayLiteral { elements: Vec<Expr> },
    FieldAccess { base: Box<Expr>, field: String },
    /// `(e1, e2, ...)` — tuple construction, minimum 2 elements.
    Tuple(Vec<Expr>),
    /// `expr.0`, `expr.1` — positional field access on a tuple.
    TupleIndex { base: Box<Expr>, index: usize },
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
    LetMut { name: String, expr: Expr },
    Assign { target: String, expr: Expr },
    Return { expr: Expr },
    Expr(Expr),
    /// `if condition: body [else: body]`
    ///
    /// `else if` is expressed as `else_body = Some(vec![If { .. }])` — the user
    /// writes an `if` inside the `else:` block, which produces the same tree.
    If {
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    /// `for var in range(end):` or `for var in range(start, end):`
    ///
    /// `range` is syntactic sugar recognised only in this position — it is NOT
    /// a runtime function. The parser desugars `range(n)` to `start = 0, end = n`.
    For {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
    /// `for var in <array_expr>: body`
    ///
    /// Only reached when the iterator is NOT `range(...)`. The `iter` must
    /// resolve to an `Array<T, N>` binding; `var` is bound to `T` inside body.
    ForIn { var: String, iter: Box<Expr>, body: Vec<Stmt> },
    /// `while condition: body`
    While { condition: Expr, body: Vec<Stmt> },
    /// `match scrutinee: arms`
    ///
    /// Exhaustive — every variant must appear exactly once. Arms may bind payload
    /// fields positionally. `return` is valid as an arm terminator.
    Match { scrutinee: Expr, arms: Vec<MatchArm> },
    /// `let [mut] (a, b, ...) = expr` — tuple destructuring.
    LetTuple { names: Vec<String>, mutable: bool, expr: Expr },
    /// `break` — exit the innermost loop immediately.
    Break,
    /// `continue` — jump to the next iteration of the innermost loop.
    Continue,
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
    /// `struct Name: field: Type ...`
    Struct {
        name: String,
        fields: Vec<FieldDef>,
    },
    /// `enum Name: Variant / Variant(fields) ...`
    Enum {
        name: String,
        variants: Vec<VariantDef>,
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
