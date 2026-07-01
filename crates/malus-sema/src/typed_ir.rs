use std::collections::{HashMap, HashSet};
use malus_syntax::ast::{BinOp, Lit, Placement, ScalarTy, UnaryOp};
use malus_syntax::Span;
use crate::ty::ResolvedTy;

// ── Program ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct TypedProgram {
    pub fns: Vec<TypedFn>,
    pub kernels: Vec<TypedKernel>,
    /// M27 grad-inference (`grad_inference.rs`): `(struct_name, field_name)` pairs
    /// where at least one construction/field-assign site stores a grad-tracked value.
    pub struct_field_grad: HashSet<(String, String)>,
    /// M27 grad-inference: per-fn, per-parameter-position grad-tracked flag.
    /// True if any call site passes a grad-tracked argument in that position.
    pub fn_param_grad: HashMap<String, Vec<bool>>,
    /// M27 grad-inference: per-fn return-value grad-tracked flag.
    pub fn_ret_grad: HashMap<String, bool>,
}

// ── Functions and Kernels ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypedParam {
    pub name: String,
    pub ty: ResolvedTy,
    /// True for `mut` parameters — interior mutation allowed, bare rebind rejected.
    pub is_mut: bool,
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
    /// True when the kernel body is the legacy implicit-map form: only `let`
    /// bindings and a single final `return scalar_expr` with no thread
    /// intrinsics, indexing, shared memory, or control flow.  Codegen-gpu
    /// uses this to choose the old `out[tid]=expr` lowering vs the new
    /// explicit lowering.
    pub is_implicit_map: bool,
}

// ── Assign targets ───────────────────────────────────────────────────────────

/// The left-hand side of an lvalue assignment (`a[i]=e` or `s.f=e` or `x=e`).
/// Single-level only; nested lvalues (`a.b[i]`) are rejected in sema (M20 scope).
#[derive(Debug, Clone)]
pub enum TypedAssignTarget {
    /// `name = e` — bare variable rebind. Requires `let mut` local.
    Ident(String),
    /// `base[index] = e` — indexed array element assignment.
    /// Requires the base binding to be mutable (`let mut` or `mut` param).
    Index {
        base: String,
        index: Box<TypedExpr>,
        elem_ty: ResolvedTy,
    },
    /// `base.field = e` — struct field assignment.
    /// Requires the base binding to be mutable (`let mut` or `mut` param).
    Field {
        base: String,
        slot_idx: usize,
        field_ty: ResolvedTy,
    },
    /// `base[index] = e` — Buffer element assignment. Narrows i64 → i32 in runtime.
    BufferIndex {
        base: String,
        index: Box<TypedExpr>,
        dtype: malus_syntax::ast::ScalarTy,
    },
    /// `base[index] = e` — `List<T>` element assignment (M28). Distinct from `Index`
    /// because `List`'s runtime layout has a length word after the ARC header
    /// (`[refcount | len | elem0 | elem1 | ...]`), offsetting elements by 16 bytes
    /// instead of `Array`'s 0. Codegen releases the old element (if tensor-typed)
    /// before storing. See ADR-0034.
    ListIndex {
        base: String,
        index: Box<TypedExpr>,
        elem_ty: ResolvedTy,
    },
}

// ── Statements ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum TypedStmt {
    Let { name: String, expr: TypedExpr },
    /// Lvalue assignment. CTMM drops the old element/field value before this
    /// stmt executes. Codegen stores the new value to the computed address.
    Assign { target: TypedAssignTarget, expr: TypedExpr },
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
        /// Slot indices of nested struct/enum fields that need ARC release.
        /// Populated in M13+; empty in M12 and earlier.
        retained_agg_slots: Vec<usize>,
    },
    /// CTMM (M11): free an enum's heap box, releasing the active variant's
    /// owned fields.  `variants` is `(tag_value, droppable_fields, retained_agg_slots)`.
    /// Codegen emits a tag-switch + per-arm release + shared free.
    DropEnum { name: String, variants: Vec<(u32, Vec<(usize, ResolvedTy)>, Vec<usize>)> },
    /// CTMM (M11): release each element of a fixed array (Phase 4 implementation).
    DropArray { name: String, elem_ty: ResolvedTy, len: usize },
    /// CTMM (M28): release a `List<T>` binding. Unlike `DropArray`, this is ALWAYS
    /// reference-counted, never a static free — `List` values may alias across a
    /// call boundary (e.g. `Module::parameters(self) -> List<Tensor<f32>>` returning
    /// a model's own field by identity) that neither M28's nor M29's (intraprocedural
    /// -only) static analysis can prove safe to free unconditionally. Codegen: read
    /// the length word, decrement the refcount, and only if that was the last
    /// reference release each element (tensor-typed elements only in V4 scope) then
    /// free the box. See ADR-0034.
    DropList { name: String, elem_ty: ResolvedTy },
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
    /// `droppable_fields` is `(slot_index, field_ty)` for Tensor fields.
    DropTuple { name: String, droppable_fields: Vec<(usize, ResolvedTy)> },
    // ── M14: tape control ─────────────────────────────────────────────────────
    /// `with no_grad: body` — emit tape_pause() before body, tape_resume() after.
    /// M27: grad-inference also forces every expression lexically inside `body`
    /// to be non-grad-tracked, so CTMM statically drops (not RC-releases) locals
    /// bound here (ADR-0030/0032). Early-exit (return/break/continue) across this
    /// boundary is rejected by sema.
    NoGrad { body: Vec<TypedStmt> },
    // ── M22: Buffer<i32> ─────────────────────────────────────────────────────
    /// CTMM: free a CPU-side staging buffer.
    DropBuffer { name: String },
    // ── M24: kernel shared memory ─────────────────────────────────────────────
    /// `let shared name: Array<T, N>` inside an explicit kernel body.
    /// Codegen-gpu emits `threadgroup T name[N]`.
    LetShared { name: String, elem_ty: malus_syntax::ast::ScalarTy, size: usize },
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
    /// M27 grad-inference (`grad_inference.rs`): true if this expression's result
    /// may be saved onto the autograd tape. Set by the pass; `false` as produced
    /// by `check.rs`. Drives tape recording (codegen-cpu), `.grad` legality, and
    /// the CTMM Release-vs-Drop choice (escape_set == grad_tracked, ADR-0030).
    pub grad_tracked: bool,
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
    /// M25 kernel launch: `kernel[grid=[..], tg=[..], out=[..]](tensor_args, scalar_args)`.
    /// `grid`/`tg` are i64[3] arrays; `out_shape` is optional (None = first tensor input's shape).
    /// Lowers to `kernel_dispatch_v2` in codegen-cpu.
    KernelLaunch {
        kernel: String,
        grid: Box<TypedExpr>,
        tg: Box<TypedExpr>,
        out_shape: Option<Box<TypedExpr>>,
        /// Tensor input args in declaration order (become handles).
        tensor_args: Vec<TypedExpr>,
        /// Scalar uniform args in declaration order (packed into Uniforms blob).
        scalar_args: Vec<TypedExpr>,
    },
}
