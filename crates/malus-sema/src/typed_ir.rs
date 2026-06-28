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
    /// Increment the aggregate box's reference count (`aggregate_retain`). M13+.
    RetainAgg { name: String },
    /// Decrement the aggregate box's reference count; frees when it reaches zero
    /// (`aggregate_release`). M13+.
    ReleaseAgg { name: String },
    // ── M11: fixed arrays ─────────────────────────────────────────────────────
    /// `for var in <array_expr>: body`  (from AST `StmtKind::ForIn`).
    ///
    /// `var` is bound inside `body` to each element in declaration order.
    /// Elements are borrowed — no ownership transfer; the array binding retains
    /// ownership and is dropped in the outer scope after the ForIn exits.
    ForIn { var: String, iter: TypedExpr, body: Vec<TypedStmt> },
    // ── M10/M11: aggregate types ──────────────────────────────────────────────
    /// CTMM: free a struct's heap box and recursively drop its owned fields.
    /// `droppable_fields` is `(slot_index, field_ty)` for every field that owns
    /// heap resources (Tensor, Struct, Enum, Array).  Codegen recurses based on
    /// the type so nested aggregates are fully released before the box is freed.
    DropStruct {
        name: String,
        droppable_fields: Vec<(usize, ResolvedTy)>,
        /// Slot indices of nested Variable/struct/enum fields that need ARC release.
        /// Populated in M13+; empty in M12 and earlier.
        retained_agg_slots: Vec<usize>,
    },
    /// CTMM (M11): free an enum's heap box, releasing the active variant's
    /// owned fields.  `variants` is `(tag_value, droppable_fields, retained_agg_slots)`.
    /// Codegen emits a tag-switch + per-arm release + shared free.
    DropEnum { name: String, variants: Vec<(u32, Vec<(usize, ResolvedTy)>, Vec<usize>)> },
    /// CTMM (M11): release each element of a fixed array (Phase 4 implementation).
    DropArray { name: String, elem_ty: ResolvedTy, len: usize },
    /// Exhaustive `match` on an enum binding.
    Match { scrutinee: TypedExpr, arms: Vec<TypedMatchArm> },
    // ── M12: loop control ─────────────────────────────────────────────────────
    /// `break` — exit the innermost loop.  CTMM injects Drop/DropStruct/DropEnum
    /// for all loop-body locals live at this point before the jump.
    Break,
    /// `continue` — jump to the next iteration of the innermost loop.  Same
    /// CTMM unwind as `Break`.
    Continue,
    // ── M13.5: tuples ─────────────────────────────────────────────────────────
    /// `let [mut] (a, b, ...) = expr` — tuple destructuring.
    /// `names` is `(binding_name, element_type)`.
    /// The tuple box is freed immediately after extracting fields in codegen.
    LetTuple { names: Vec<(String, ResolvedTy)>, expr: TypedExpr },
    /// CTMM: free a tuple's heap box and release owned fields.
    /// `droppable_fields` is `(slot_index, field_ty)` for Tensor/Variable fields.
    DropTuple { name: String, droppable_fields: Vec<(usize, ResolvedTy)> },
}

/// One arm of a `match` statement.
#[derive(Debug, Clone)]
pub struct TypedMatchArm {
    pub variant: String,
    pub variant_index: u32,
    /// `(local_name, field_type)` in field-declaration order.
    pub bindings: Vec<(String, ResolvedTy)>,
    pub body: Vec<TypedStmt>,
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
        /// Row-major shape inferred from the literal syntax.  1-D literals have
        /// `shape = [N]`; 2-D literals have `shape = [rows, cols]`.
        shape: Vec<usize>,
    },
    FieldAccess {
        base: Box<TypedExpr>,
        field: String,
    },
    // ── M11: fixed arrays ────────────────────────────────────────────────────
    /// `[e1, e2, e3]` — typed array literal. All elements have the same type.
    ArrayLiteral {
        elements: Vec<TypedExpr>,
    },
    // ── M10: aggregate constructors ──────────────────────────────────────────
    /// Struct construction: fields reordered to declaration order.
    StructInit {
        name: String,
        fields: Vec<TypedExpr>,
    },
    /// Enum variant construction.
    /// `max_payload_slots` = max field count across all variants (for allocation).
    EnumInit {
        enum_name: String,
        variant: String,
        variant_index: u32,
        payload: Vec<TypedExpr>,
        max_payload_slots: usize,
    },
    // ── M13.5: tuples ────────────────────────────────────────────────────────
    /// `(e1, e2, ...)` — typed tuple construction, minimum 2 elements.
    TupleInit {
        elements: Vec<TypedExpr>,
    },
    /// `expr.0`, `expr.1` — positional field access on a tuple.
    TupleIndex {
        base: Box<TypedExpr>,
        index: usize,
    },
}
