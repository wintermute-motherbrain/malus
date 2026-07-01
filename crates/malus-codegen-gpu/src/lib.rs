use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use malus_sema::{ResolvedTy, TypedExpr, TypedExprKind, TypedKernel, TypedProgram, TypedStmt};
use malus_syntax::ast::{
    elementwise_builtin_name, scalar_broadcast_builtin_name, BinOp, Lit, ScalarTy, UnaryOp,
};

// Process-global, never reset: guarantees kernel ids stay unique across every
// compile_kernels() call in a process, not just within one. See ADR-0033.
static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod tests;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CodegenError {
    UnsupportedKernelBody(String),
    NonTensorReturnType(String),
    NonTensorParamType(String),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::UnsupportedKernelBody(s) => {
                write!(f, "unsupported kernel body: {s}")
            }
            CodegenError::NonTensorReturnType(s) => {
                write!(f, "kernel must return a tensor, got: {s}")
            }
            CodegenError::NonTensorParamType(s) => {
                write!(f, "kernel param must be a tensor, got: {s}")
            }
        }
    }
}

impl std::error::Error for CodegenError {}

// ── Kernel registry ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KernelRegistry {
    kernels: HashMap<u64, String>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: u64, msl_source: String) {
        self.kernels.insert(id, msl_source);
    }

    pub fn msl_for(&self, id: u64) -> Option<&str> {
        self.kernels.get(&id).map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&u64, &String)> {
        self.kernels.iter()
    }

    pub fn into_hashmap(self) -> HashMap<u64, String> {
        self.kernels
    }

    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl From<HashMap<u64, String>> for KernelRegistry {
    fn from(kernels: HashMap<u64, String>) -> Self {
        Self { kernels }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

// Unary math builtin names (sorted alphabetically for BTreeSet determinism).
const UNARY_BUILTIN_NAMES: &[&str] = &["abs", "exp", "log", "relu", "sigmoid", "sqrt", "tanh"];

pub fn compile_kernels(
    program: &TypedProgram,
) -> Result<(KernelRegistry, HashMap<String, u64>), CodegenError> {
    let mut registry = KernelRegistry::new();
    let mut name_to_id = HashMap::new();

    for kernel in &program.kernels {
        let kernel_id = NEXT_KERNEL_ID.fetch_add(1, Ordering::SeqCst);
        let msl = lower_kernel(kernel, kernel_id)?;
        registry.insert(kernel_id, msl);
        name_to_id.insert(kernel.name.clone(), kernel_id);
    }

    // Tensor-tensor element-wise builtins (ADR-0010: appended after user kernels).
    let mut tensor_ops: BTreeSet<BinOp> = BTreeSet::new();
    // Scalar-broadcast builtins: (op, scalar_on_right).
    let mut scalar_ops: BTreeSet<(BinOp, bool)> = BTreeSet::new();
    // Unary math builtins used in fn bodies.
    let mut unary_ops: BTreeSet<String> = BTreeSet::new();

    for f in &program.fns {
        for stmt in &f.body {
            collect_binops_in_stmt(stmt, &mut tensor_ops, &mut scalar_ops);
            collect_unary_builtins_in_stmt(stmt, &mut unary_ops);
        }
    }

    for op in &tensor_ops {
        let name = elementwise_builtin_name(op)
            .expect("collected op must have a builtin name");
        let kernel_id = NEXT_KERNEL_ID.fetch_add(1, Ordering::SeqCst);
        let msl = synthesize_elementwise_builtin(*op, kernel_id)?;
        registry.insert(kernel_id, msl);
        name_to_id.insert(name.to_string(), kernel_id);
    }

    for (op, scalar_on_right) in &scalar_ops {
        let name = scalar_broadcast_builtin_name(op, *scalar_on_right)
            .expect("collected scalar op must have a builtin name");
        if name_to_id.contains_key(name) {
            continue; // commutative: both orderings share the same kernel
        }
        let kernel_id = NEXT_KERNEL_ID.fetch_add(1, Ordering::SeqCst);
        let msl = synthesize_scalar_builtin(*op, *scalar_on_right, kernel_id)?;
        registry.insert(kernel_id, msl);
        name_to_id.insert(name.to_string(), kernel_id);
    }

    // Unary math builtins (sorted, appended after tensor/scalar builtins per ADR-0010).
    for name in &unary_ops {
        if name_to_id.contains_key(name.as_str()) {
            continue;
        }
        let kernel_id = NEXT_KERNEL_ID.fetch_add(1, Ordering::SeqCst);
        let msl = synthesize_unary_builtin(name, kernel_id)?;
        registry.insert(kernel_id, msl);
        name_to_id.insert(name.to_string(), kernel_id);
    }

    Ok((registry, name_to_id))
}

fn collect_binops_in_stmt(
    stmt: &TypedStmt,
    tensor_ops: &mut BTreeSet<BinOp>,
    scalar_ops: &mut BTreeSet<(BinOp, bool)>,
) {
    match stmt {
        TypedStmt::Let { expr, .. } => collect_binops_in_expr(expr, tensor_ops, scalar_ops),
        TypedStmt::Assign { expr, .. } => collect_binops_in_expr(expr, tensor_ops, scalar_ops),
        TypedStmt::Return { expr } => collect_binops_in_expr(expr, tensor_ops, scalar_ops),
        TypedStmt::Expr(expr) => collect_binops_in_expr(expr, tensor_ops, scalar_ops),
        TypedStmt::Drop { .. } | TypedStmt::DropStruct { .. } | TypedStmt::DropEnum { .. }
        | TypedStmt::DropArray { .. } | TypedStmt::DropTuple { .. } | TypedStmt::DropBuffer { .. }
        | TypedStmt::GpuBarrier
        | TypedStmt::Retain { .. } | TypedStmt::Release { .. }
        | TypedStmt::RetainAgg { .. } | TypedStmt::ReleaseAgg { .. } => {}
        TypedStmt::LetTuple { expr, .. } => collect_binops_in_expr(expr, tensor_ops, scalar_ops),
        TypedStmt::If { condition, then_body, else_body } => {
            collect_binops_in_expr(condition, tensor_ops, scalar_ops);
            for s in then_body { collect_binops_in_stmt(s, tensor_ops, scalar_ops); }
            if let Some(eb) = else_body { for s in eb { collect_binops_in_stmt(s, tensor_ops, scalar_ops); } }
        }
        TypedStmt::For { body, .. } => {
            for s in body { collect_binops_in_stmt(s, tensor_ops, scalar_ops); }
        }
        TypedStmt::While { condition, body } => {
            collect_binops_in_expr(condition, tensor_ops, scalar_ops);
            for s in body { collect_binops_in_stmt(s, tensor_ops, scalar_ops); }
        }
        TypedStmt::ForIn { body, .. } => {
            for s in body { collect_binops_in_stmt(s, tensor_ops, scalar_ops); }
        }
        TypedStmt::Match { scrutinee, arms } => {
            collect_binops_in_expr(scrutinee, tensor_ops, scalar_ops);
            for arm in arms {
                for s in &arm.body {
                    collect_binops_in_stmt(s, tensor_ops, scalar_ops);
                }
            }
        }
        TypedStmt::Break | TypedStmt::Continue => {}
        TypedStmt::NoGrad { body } => {
            for s in body { collect_binops_in_stmt(s, tensor_ops, scalar_ops); }
        }
        TypedStmt::LetShared { .. } => {}
    }
}

fn collect_binops_in_expr(
    expr: &TypedExpr,
    tensor_ops: &mut BTreeSet<BinOp>,
    scalar_ops: &mut BTreeSet<(BinOp, bool)>,
) {
    match &expr.kind {
        TypedExprKind::BinOp { op, lhs, rhs } => {
            let lhs_agg = lhs.ty.is_tensor() || lhs.ty.is_variable();
            let rhs_agg = rhs.ty.is_tensor() || rhs.ty.is_variable();
            if lhs_agg && rhs_agg {
                if elementwise_builtin_name(op).is_some() {
                    tensor_ops.insert(*op);
                }
            } else if lhs.ty.is_tensor() && matches!(rhs.ty, ResolvedTy::Scalar(_)) {
                if scalar_broadcast_builtin_name(op, true).is_some() {
                    scalar_ops.insert((*op, true));
                }
            } else if matches!(lhs.ty, ResolvedTy::Scalar(_)) && rhs.ty.is_tensor() {
                if scalar_broadcast_builtin_name(op, false).is_some() {
                    scalar_ops.insert((*op, false));
                }
            }
            collect_binops_in_expr(lhs, tensor_ops, scalar_ops);
            collect_binops_in_expr(rhs, tensor_ops, scalar_ops);
        }
        TypedExprKind::Unary { operand, .. } => {
            collect_binops_in_expr(operand, tensor_ops, scalar_ops);
        }
        TypedExprKind::Call { args, .. } => {
            for a in args { collect_binops_in_expr(a, tensor_ops, scalar_ops); }
        }
        TypedExprKind::KernelCall { args, .. } => {
            for a in args { collect_binops_in_expr(a, tensor_ops, scalar_ops); }
        }
        TypedExprKind::TensorLiteral { elements, .. } => {
            for e in elements { collect_binops_in_expr(e, tensor_ops, scalar_ops); }
        }
        TypedExprKind::Index { base, indices } => {
            collect_binops_in_expr(base, tensor_ops, scalar_ops);
            for i in indices { collect_binops_in_expr(i, tensor_ops, scalar_ops); }
        }
        TypedExprKind::FieldAccess { base, .. } => {
            collect_binops_in_expr(base, tensor_ops, scalar_ops);
        }
        TypedExprKind::StructInit { fields, .. } => {
            for f in fields { collect_binops_in_expr(f, tensor_ops, scalar_ops); }
        }
        TypedExprKind::EnumInit { payload, .. } => {
            for p in payload { collect_binops_in_expr(p, tensor_ops, scalar_ops); }
        }
        TypedExprKind::ArrayLiteral { elements } => {
            for e in elements { collect_binops_in_expr(e, tensor_ops, scalar_ops); }
        }
        TypedExprKind::TupleInit { elements } => {
            for e in elements { collect_binops_in_expr(e, tensor_ops, scalar_ops); }
        }
        TypedExprKind::TupleIndex { base, .. } => collect_binops_in_expr(base, tensor_ops, scalar_ops),
        TypedExprKind::KernelLaunch { tensor_args, scalar_args, grid, tg, out_shape, .. } => {
            for a in tensor_args { collect_binops_in_expr(a, tensor_ops, scalar_ops); }
            for a in scalar_args { collect_binops_in_expr(a, tensor_ops, scalar_ops); }
            collect_binops_in_expr(grid, tensor_ops, scalar_ops);
            collect_binops_in_expr(tg, tensor_ops, scalar_ops);
            if let Some(os) = out_shape { collect_binops_in_expr(os, tensor_ops, scalar_ops); }
        }
        TypedExprKind::Lit(_) | TypedExprKind::Ident(_) => {}
    }
}

fn collect_unary_builtins_in_stmt(stmt: &TypedStmt, out: &mut BTreeSet<String>) {
    match stmt {
        TypedStmt::Let { expr, .. } | TypedStmt::Assign { expr, .. } | TypedStmt::Return { expr } => {
            collect_unary_builtins_in_expr(expr, out);
        }
        TypedStmt::Expr(expr) => collect_unary_builtins_in_expr(expr, out),
        TypedStmt::Drop { .. } | TypedStmt::DropStruct { .. } | TypedStmt::DropEnum { .. }
        | TypedStmt::DropArray { .. } | TypedStmt::DropTuple { .. } | TypedStmt::DropBuffer { .. }
        | TypedStmt::GpuBarrier
        | TypedStmt::Retain { .. } | TypedStmt::Release { .. }
        | TypedStmt::RetainAgg { .. } | TypedStmt::ReleaseAgg { .. } => {}
        TypedStmt::LetTuple { expr, .. } => collect_unary_builtins_in_expr(expr, out),
        TypedStmt::If { condition, then_body, else_body } => {
            collect_unary_builtins_in_expr(condition, out);
            for s in then_body { collect_unary_builtins_in_stmt(s, out); }
            if let Some(eb) = else_body { for s in eb { collect_unary_builtins_in_stmt(s, out); } }
        }
        TypedStmt::For { body, .. } => {
            for s in body { collect_unary_builtins_in_stmt(s, out); }
        }
        TypedStmt::While { condition, body } => {
            collect_unary_builtins_in_expr(condition, out);
            for s in body { collect_unary_builtins_in_stmt(s, out); }
        }
        TypedStmt::ForIn { body, .. } => {
            for s in body { collect_unary_builtins_in_stmt(s, out); }
        }
        TypedStmt::Match { scrutinee, arms } => {
            collect_unary_builtins_in_expr(scrutinee, out);
            for arm in arms {
                for s in &arm.body {
                    collect_unary_builtins_in_stmt(s, out);
                }
            }
        }
        TypedStmt::Break | TypedStmt::Continue => {}
        TypedStmt::NoGrad { body } => {
            for s in body { collect_unary_builtins_in_stmt(s, out); }
        }
        TypedStmt::LetShared { .. } => {}
    }
}

fn collect_unary_builtins_in_expr(expr: &TypedExpr, out: &mut BTreeSet<String>) {
    match &expr.kind {
        TypedExprKind::Call { callee, args } => {
            if UNARY_BUILTIN_NAMES.contains(&callee.as_str()) {
                out.insert(callee.clone());
            }
            for a in args {
                collect_unary_builtins_in_expr(a, out);
            }
        }
        TypedExprKind::BinOp { lhs, rhs, .. } => {
            collect_unary_builtins_in_expr(lhs, out);
            collect_unary_builtins_in_expr(rhs, out);
        }
        TypedExprKind::Unary { operand, .. } => collect_unary_builtins_in_expr(operand, out),
        TypedExprKind::KernelCall { args, .. } => {
            for a in args {
                collect_unary_builtins_in_expr(a, out);
            }
        }
        TypedExprKind::TensorLiteral { elements, .. } => {
            for e in elements {
                collect_unary_builtins_in_expr(e, out);
            }
        }
        TypedExprKind::Index { base, indices } => {
            collect_unary_builtins_in_expr(base, out);
            for i in indices {
                collect_unary_builtins_in_expr(i, out);
            }
        }
        TypedExprKind::FieldAccess { base, .. } => collect_unary_builtins_in_expr(base, out),
        TypedExprKind::StructInit { fields, .. } => {
            for f in fields { collect_unary_builtins_in_expr(f, out); }
        }
        TypedExprKind::EnumInit { payload, .. } => {
            for p in payload { collect_unary_builtins_in_expr(p, out); }
        }
        TypedExprKind::ArrayLiteral { elements } => {
            for e in elements { collect_unary_builtins_in_expr(e, out); }
        }
        TypedExprKind::TupleInit { elements } => {
            for e in elements { collect_unary_builtins_in_expr(e, out); }
        }
        TypedExprKind::TupleIndex { base, .. } => collect_unary_builtins_in_expr(base, out),
        TypedExprKind::KernelLaunch { tensor_args, scalar_args, grid, tg, out_shape, .. } => {
            for a in tensor_args { collect_unary_builtins_in_expr(a, out); }
            for a in scalar_args { collect_unary_builtins_in_expr(a, out); }
            collect_unary_builtins_in_expr(grid, out);
            collect_unary_builtins_in_expr(tg, out);
            if let Some(os) = out_shape { collect_unary_builtins_in_expr(os, out); }
        }
        TypedExprKind::Lit(_) | TypedExprKind::Ident(_) => {}
    }
}

fn synthesize_unary_builtin(name: &str, kernel_id: u64) -> Result<String, CodegenError> {
    let expr = match name {
        "relu"    => "fmax(0.0f, a[tid])".to_string(),
        "sigmoid" => "1.0f / (1.0f + exp(-a[tid]))".to_string(),
        "tanh"    => "tanh(a[tid])".to_string(),
        "exp"     => "exp(a[tid])".to_string(),
        "log"     => "log(a[tid])".to_string(),
        "sqrt"    => "sqrt(a[tid])".to_string(),
        "abs"     => "fabs(a[tid])".to_string(),
        _ => return Err(CodegenError::UnsupportedKernelBody(
            format!("unknown unary builtin: {name}")
        )),
    };
    Ok(format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void malus_kernel_{}(\n    device float* a [[buffer(0)]],\n    device float* out [[buffer(1)]],\n    uint tid [[thread_position_in_grid]]\n) {{\n    out[tid] = {};\n}}\n",
        kernel_id, expr,
    ))
}

fn synthesize_elementwise_builtin(op: BinOp, kernel_id: u64) -> Result<String, CodegenError> {
    let msl_op = binop_to_msl(&op)?;
    Ok(format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void malus_kernel_{}(\n    device float* a [[buffer(0)]],\n    device float* b [[buffer(1)]],\n    device float* out [[buffer(2)]],\n    uint tid [[thread_position_in_grid]]\n) {{\n    out[tid] = (a[tid] {} b[tid]);\n}}\n",
        kernel_id, msl_op,
    ))
}

/// Synthesize a scalar-broadcast builtin. Layout: a@0, scalar_val@1, out@2.
/// For `scalar_on_right=true`: `out = a op scalar_val[0]`.
/// For `scalar_on_right=false` (reversed): `out = scalar_val[0] op a`.
fn synthesize_scalar_builtin(
    op: BinOp,
    scalar_on_right: bool,
    kernel_id: u64,
) -> Result<String, CodegenError> {
    let msl_op = binop_to_msl(&op)?;
    let expr = if scalar_on_right {
        format!("(a[tid] {} scalar_val[0])", msl_op)
    } else {
        format!("(scalar_val[0] {} a[tid])", msl_op)
    };
    Ok(format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void malus_kernel_{}(\n    device float* a [[buffer(0)]],\n    device float* scalar_val [[buffer(1)]],\n    device float* out [[buffer(2)]],\n    uint tid [[thread_position_in_grid]]\n) {{\n    out[tid] = {};\n}}\n",
        kernel_id, expr,
    ))
}

// ── MSL lowering ──────────────────────────────────────────────────────────────

fn lower_kernel(kernel: &TypedKernel, kernel_id: u64) -> Result<String, CodegenError> {
    if kernel.is_implicit_map {
        lower_kernel_implicit(kernel, kernel_id)
    } else {
        lower_kernel_explicit(kernel, kernel_id)
    }
}

// ── Implicit-map kernel (legacy element-space form) ───────────────────────────

fn lower_kernel_implicit(kernel: &TypedKernel, kernel_id: u64) -> Result<String, CodegenError> {
    let func_name = format!("malus_kernel_{}", kernel_id);

    let return_dtype = kernel.return_ty.tensor_dtype().ok_or_else(|| {
        CodegenError::NonTensorReturnType(kernel.return_ty.to_string())
    })?;
    let return_msl_type = dtype_to_msl(return_dtype);

    let mut params = Vec::new();
    let mut param_names = HashSet::new();

    for (i, param) in kernel.params.iter().enumerate() {
        let param_dtype = param.ty.tensor_dtype().ok_or_else(|| {
            CodegenError::NonTensorParamType(param.ty.to_string())
        })?;
        let param_msl_type = dtype_to_msl(param_dtype);
        params.push(format!(
            "device {}* {} [[buffer({})]]",
            param_msl_type, param.name, i
        ));
        param_names.insert(param.name.clone());
    }

    let out_index = kernel.params.len();
    params.push(format!(
        "device {}* out [[buffer({})]]",
        return_msl_type, out_index
    ));
    params.push("uint tid [[thread_position_in_grid]]".to_string());

    let body_msl = lower_kernel_body_implicit(&kernel.body, &param_names)?;

    let msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void {}(\n    {}\n) {{\n    {}\n}}\n",
        func_name,
        params.join(",\n    "),
        body_msl,
    );

    Ok(msl)
}

fn lower_kernel_body_implicit(
    body: &[TypedStmt],
    param_names: &HashSet<String>,
) -> Result<String, CodegenError> {
    if body.is_empty() {
        return Err(CodegenError::UnsupportedKernelBody(
            "kernel body must not be empty".into(),
        ));
    }

    let mut local_names: HashSet<String> = HashSet::new();
    let mut lines: Vec<String> = Vec::new();

    for (i, stmt) in body.iter().enumerate() {
        let is_last = i == body.len() - 1;
        match stmt {
            TypedStmt::Let { name, expr } => {
                if is_last {
                    return Err(CodegenError::UnsupportedKernelBody(
                        "kernel body must end with a return statement".into(),
                    ));
                }
                let msl_ty = resolved_ty_to_msl(&expr.ty)?;
                let expr_msl = lower_expr_implicit(expr, param_names, &local_names)?;
                lines.push(format!("{} {} = {};", msl_ty, name, expr_msl));
                local_names.insert(name.clone());
            }
            TypedStmt::Return { expr } => {
                if !is_last {
                    return Err(CodegenError::UnsupportedKernelBody(
                        "return must be the last statement in kernel body".into(),
                    ));
                }
                let expr_msl = lower_expr_implicit(expr, param_names, &local_names)?;
                lines.push(format!("out[tid] = {};", expr_msl));
            }
            _ => {
                return Err(CodegenError::UnsupportedKernelBody(
                    "only let bindings and a final return are allowed in implicit-map kernel bodies".into(),
                ));
            }
        }
    }

    Ok(lines.join("\n    "))
}

fn lower_expr_implicit(
    expr: &TypedExpr,
    param_names: &HashSet<String>,
    local_names: &HashSet<String>,
) -> Result<String, CodegenError> {
    match &expr.kind {
        TypedExprKind::Ident(name) => {
            if param_names.contains(name) {
                Ok(format!("{}[tid]", name))
            } else if local_names.contains(name) {
                Ok(name.clone())
            } else {
                Err(CodegenError::UnsupportedKernelBody(format!(
                    "unknown identifier in kernel: {}", name
                )))
            }
        }

        TypedExprKind::Lit(lit) => lower_lit(lit),

        TypedExprKind::BinOp { op, lhs, rhs } => {
            let l = lower_expr_implicit(lhs, param_names, local_names)?;
            let r = lower_expr_implicit(rhs, param_names, local_names)?;
            let msl_op = binop_to_msl_infix(op)?;
            Ok(format!("({} {} {})", l, msl_op, r))
        }

        TypedExprKind::Unary { op, operand } => {
            let val = lower_expr_implicit(operand, param_names, local_names)?;
            match op {
                UnaryOp::Neg => Ok(format!("(-{})", val)),
                UnaryOp::Not => Err(CodegenError::UnsupportedKernelBody(
                    "logical not not supported in kernel bodies".into(),
                )),
            }
        }

        _ => Err(CodegenError::UnsupportedKernelBody(
            "unsupported expression kind in implicit-map kernel body".into(),
        )),
    }
}

// ── Explicit kernel (M24 full GPU programming model) ──────────────────────────

/// Context threaded through explicit kernel body lowering.
#[derive(Clone)]
struct KernelCtx {
    /// Names of tensor pointer parameters (device const T* name).
    tensor_param_names: HashSet<String>,
    /// Names of scalar uniform parameters (accessed as u.name).
    scalar_param_names: HashSet<String>,
    /// All names bound as locals in the current or any enclosing scope,
    /// including shared-memory arrays.  Let-bound loop variables are added
    /// as the For stmt is processed.
    local_names: HashSet<String>,
}

/// Thread intrinsic names and their MSL [[attribute]] equivalents.
const INTRINSIC_ATTR: &[(&str, &str, &str)] = &[
    ("thread_id",              "_tid",     "thread_position_in_grid"),
    ("threadgroup_id",         "_tgid",    "threadgroup_position_in_grid"),
    ("thread_in_threadgroup",  "_lid",     "thread_position_in_threadgroup"),
    ("threads_per_threadgroup","_tgsize",  "threads_per_threadgroup"),
    ("threads_per_grid",       "_gridsize","threads_per_grid"),
];

fn intrinsic_var(name: &str) -> Option<&'static str> {
    INTRINSIC_ATTR.iter().find(|(n, _, _)| *n == name).map(|(_, v, _)| *v)
}

fn collect_used_intrinsics(body: &[TypedStmt], out: &mut HashSet<String>) {
    for stmt in body {
        collect_used_intrinsics_stmt(stmt, out);
    }
}

fn collect_used_intrinsics_stmt(stmt: &TypedStmt, out: &mut HashSet<String>) {
    match stmt {
        TypedStmt::Let { expr, .. } | TypedStmt::Return { expr } | TypedStmt::Assign { expr, .. } => {
            collect_used_intrinsics_expr(expr, out);
        }
        TypedStmt::Expr(expr) => collect_used_intrinsics_expr(expr, out),
        TypedStmt::If { condition, then_body, else_body } => {
            collect_used_intrinsics_expr(condition, out);
            collect_used_intrinsics(then_body, out);
            if let Some(eb) = else_body { collect_used_intrinsics(eb, out); }
        }
        TypedStmt::For { start, end, body, .. } => {
            collect_used_intrinsics_expr(start, out);
            collect_used_intrinsics_expr(end, out);
            collect_used_intrinsics(body, out);
        }
        TypedStmt::While { condition, body } => {
            collect_used_intrinsics_expr(condition, out);
            collect_used_intrinsics(body, out);
        }
        _ => {}
    }
}

fn collect_used_intrinsics_expr(expr: &TypedExpr, out: &mut HashSet<String>) {
    match &expr.kind {
        TypedExprKind::Call { callee, args } => {
            if INTRINSIC_ATTR.iter().any(|(n, _, _)| n == callee) {
                out.insert(callee.clone());
            }
            for a in args { collect_used_intrinsics_expr(a, out); }
        }
        TypedExprKind::BinOp { lhs, rhs, .. } => {
            collect_used_intrinsics_expr(lhs, out);
            collect_used_intrinsics_expr(rhs, out);
        }
        TypedExprKind::Unary { operand, .. } => collect_used_intrinsics_expr(operand, out),
        TypedExprKind::Index { base, indices } => {
            collect_used_intrinsics_expr(base, out);
            for i in indices { collect_used_intrinsics_expr(i, out); }
        }
        _ => {}
    }
}

fn lower_kernel_explicit(kernel: &TypedKernel, kernel_id: u64) -> Result<String, CodegenError> {
    let func_name = format!("malus_kernel_{}", kernel_id);

    let return_dtype = kernel.return_ty.tensor_dtype().ok_or_else(|| {
        CodegenError::NonTensorReturnType(kernel.return_ty.to_string())
    })?;
    let return_msl_type = dtype_to_msl(return_dtype);

    // Partition params into tensor buffers and scalar uniforms.
    let mut tensor_params: Vec<_> = Vec::new();
    let mut scalar_params: Vec<_> = Vec::new();
    for p in &kernel.params {
        if p.ty.is_tensor() || p.ty.is_variable() {
            tensor_params.push(p);
        } else {
            scalar_params.push(p);
        }
    }

    // Collect which thread intrinsics the body uses.
    let mut used_intrinsics: HashSet<String> = HashSet::new();
    collect_used_intrinsics(&kernel.body, &mut used_intrinsics);

    // Build MSL parameter list.
    let mut msl_params: Vec<String> = Vec::new();

    // Tensor params → device const buffers 0..N-1.
    for (i, p) in tensor_params.iter().enumerate() {
        let dtype = p.ty.tensor_dtype().unwrap();
        let msl_ty = dtype_to_msl(dtype);
        msl_params.push(format!("device const {}* {} [[buffer({})]]", msl_ty, p.name, i));
    }

    // out → device buffer N.
    let out_idx = tensor_params.len();
    msl_params.push(format!("device {}* out [[buffer({})]]", return_msl_type, out_idx));

    // Uniforms → constant buffer N+1.
    let has_uniforms = !scalar_params.is_empty();
    if has_uniforms {
        msl_params.push(format!("constant Uniforms_{}& u [[buffer({})]]", kernel_id, out_idx + 1));
    }

    // TensorMeta buffers: inputs at hc+2.., out at hc+2+input_count.
    // Convention mirrors the runtime's D5 binding (always present; codegen emits for all tensors).
    for (i, p) in tensor_params.iter().enumerate() {
        msl_params.push(format!("constant TensorMeta& {}_meta [[buffer({})]]", p.name, out_idx + 2 + i));
    }
    msl_params.push(format!("constant TensorMeta& out_meta [[buffer({})]]", out_idx + 2 + tensor_params.len()));

    // Thread-position attributes (injected only for intrinsics the body uses).
    for (name, var, attr) in INTRINSIC_ATTR {
        if used_intrinsics.contains(*name) {
            msl_params.push(format!("uint {} [[{}]]", var, attr));
        }
    }

    // Build uniforms struct (if needed).
    let uniforms_struct = if has_uniforms {
        let fields: Vec<String> = scalar_params.iter().map(|p| {
            let msl_ty = match &p.ty {
                ResolvedTy::Scalar(s) => dtype_to_msl(s),
                other => panic!("scalar param expected, got {}", other),
            };
            format!("    {} {};", msl_ty, p.name)
        }).collect();
        format!("struct Uniforms_{} {{\n{}}};\n\n", kernel_id, fields.join("\n"))
    } else {
        String::new()
    };

    let tensor_param_names: HashSet<String> = tensor_params.iter().map(|p| p.name.clone()).collect();
    let scalar_param_names: HashSet<String> = scalar_params.iter().map(|p| p.name.clone()).collect();

    let ctx = KernelCtx {
        tensor_param_names,
        scalar_param_names,
        local_names: HashSet::new(),
    };

    let mut body_lines: Vec<String> = Vec::new();
    lower_kernel_body_explicit(&kernel.body, &ctx, 1, &mut body_lines)?;

    let indent = "    ";
    let body_str = body_lines.iter()
        .map(|l| format!("{}{}", indent, l))
        .collect::<Vec<_>>()
        .join("\n");

    // TensorMeta struct shared across all explicit kernels (D5 ABI).
    let tensor_meta_struct = "\
struct TensorMeta {\n    \
    uint ndim;\n    \
    uint shape[8];\n    \
    uint strides[8];\n\
};\n\n";

    let msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\n{}{}kernel void {}(\n    {}\n) {{\n{}\n}}\n",
        tensor_meta_struct,
        uniforms_struct,
        func_name,
        msl_params.join(",\n    "),
        body_str,
    );

    Ok(msl)
}

/// Lower a sequence of explicit-kernel statements, appending MSL lines to `out`.
/// `depth` controls indentation (1 = inside the kernel function, 2 = inside an if/for body).
fn lower_kernel_body_explicit(
    body: &[TypedStmt],
    ctx: &KernelCtx,
    depth: usize,
    out: &mut Vec<String>,
) -> Result<(), CodegenError> {
    let indent = "    ".repeat(depth.saturating_sub(1));
    let mut ctx = ctx.clone();

    for stmt in body {
        match stmt {
            TypedStmt::Let { name, expr } => {
                let msl_ty = resolved_ty_to_msl(&expr.ty)?;
                let expr_s = lower_expr_kernel(expr, &ctx)?;
                out.push(format!("{}{} {} = {};", indent, msl_ty, name, expr_s));
                ctx.local_names.insert(name.clone());
            }

            TypedStmt::LetShared { name, elem_ty, size } => {
                let msl_ty = dtype_to_msl(elem_ty);
                out.push(format!("{}threadgroup {} {}[{}];", indent, msl_ty, name, size));
                ctx.local_names.insert(name.clone());
            }

            TypedStmt::Assign { target, expr } => {
                use malus_sema::TypedAssignTarget;
                let expr_s = lower_expr_kernel(expr, &ctx)?;
                match target {
                    TypedAssignTarget::Ident(name) => {
                        out.push(format!("{}{} = {};", indent, name, expr_s));
                    }
                    TypedAssignTarget::Index { base, index, .. } => {
                        let idx_s = lower_expr_kernel(index, &ctx)?;
                        out.push(format!("{}{}[{}] = {};", indent, base, idx_s, expr_s));
                    }
                    other => {
                        return Err(CodegenError::UnsupportedKernelBody(format!(
                            "unsupported lvalue target in kernel: {:?}", other
                        )));
                    }
                }
            }

            TypedStmt::Expr(expr) => {
                if let TypedExprKind::Call { callee, .. } = &expr.kind {
                    if callee == "barrier" {
                        out.push(format!("{}threadgroup_barrier(mem_flags::mem_threadgroup);", indent));
                        continue;
                    }
                }
                let expr_s = lower_expr_kernel(expr, &ctx)?;
                out.push(format!("{}{};", indent, expr_s));
            }

            TypedStmt::If { condition, then_body, else_body } => {
                let cond_s = lower_expr_kernel(condition, &ctx)?;
                out.push(format!("{}if ({}) {{", indent, cond_s));
                lower_kernel_body_explicit(then_body, &ctx, depth + 1, out)?;
                if let Some(eb) = else_body {
                    out.push(format!("{}}} else {{", indent));
                    lower_kernel_body_explicit(eb, &ctx, depth + 1, out)?;
                }
                out.push(format!("{}}}", indent));
            }

            TypedStmt::For { var, start, end, body } => {
                let start_s = lower_expr_kernel(start, &ctx)?;
                let end_s = lower_expr_kernel(end, &ctx)?;
                let var_ty = resolved_ty_to_msl(&start.ty)?;
                out.push(format!("{}for({} {} = {}; {} < {}; {}++) {{",
                    indent, var_ty, var, start_s, var, end_s, var));
                let mut sub_ctx = ctx.clone();
                sub_ctx.local_names.insert(var.clone());
                lower_kernel_body_explicit(body, &sub_ctx, depth + 1, out)?;
                out.push(format!("{}}}", indent));
            }

            TypedStmt::While { condition, body } => {
                let cond_s = lower_expr_kernel(condition, &ctx)?;
                out.push(format!("{}while ({}) {{", indent, cond_s));
                lower_kernel_body_explicit(body, &ctx, depth + 1, out)?;
                out.push(format!("{}}}", indent));
            }

            TypedStmt::Return { .. } => {
                // Explicit kernels write to `out`; a `return` here is an early exit.
                out.push(format!("{}return;", indent));
            }

            // CTMM nodes don't appear in kernel bodies — skip silently.
            TypedStmt::Drop { .. }
            | TypedStmt::GpuBarrier
            | TypedStmt::Retain { .. }
            | TypedStmt::Release { .. }
            | TypedStmt::RetainAgg { .. }
            | TypedStmt::ReleaseAgg { .. }
            | TypedStmt::DropStruct { .. }
            | TypedStmt::DropEnum { .. }
            | TypedStmt::DropArray { .. }
            | TypedStmt::DropTuple { .. }
            | TypedStmt::DropBuffer { .. }
            | TypedStmt::NoGrad { .. }
            | TypedStmt::Break
            | TypedStmt::Continue
            | TypedStmt::LetTuple { .. }
            | TypedStmt::ForIn { .. }
            | TypedStmt::Match { .. } => {
                return Err(CodegenError::UnsupportedKernelBody(format!(
                    "statement not valid in explicit kernel body: {:?}",
                    std::mem::discriminant(stmt)
                )));
            }
        }
    }
    Ok(())
}

/// Extract the tensor name from a simple Ident expression, for metadata access.
fn extract_tensor_name(expr: &TypedExpr) -> Result<String, CodegenError> {
    match &expr.kind {
        TypedExprKind::Ident(name) => Ok(name.clone()),
        other => Err(CodegenError::UnsupportedKernelBody(format!(
            "shape/stride/ndim access requires a simple tensor parameter name, got {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

/// Lower an expression in the context of an explicit kernel body.
fn lower_expr_kernel(expr: &TypedExpr, ctx: &KernelCtx) -> Result<String, CodegenError> {
    match &expr.kind {
        TypedExprKind::Ident(name) => {
            if ctx.tensor_param_names.contains(name) || name == "out" {
                // Tensor parameter or output buffer: raw pointer name (caller indexes explicitly).
                Ok(name.clone())
            } else if ctx.scalar_param_names.contains(name) {
                // Scalar uniform: accessed via the constant uniforms struct.
                Ok(format!("u.{}", name))
            } else {
                // Local variable (let-bound scalar, shared array, loop var).
                Ok(name.clone())
            }
        }

        TypedExprKind::Lit(lit) => lower_lit(lit),

        TypedExprKind::BinOp { op, lhs, rhs } => {
            if *op == BinOp::Pow {
                let l = lower_expr_kernel(lhs, ctx)?;
                let r = lower_expr_kernel(rhs, ctx)?;
                return Ok(format!("pow({}, {})", l, r));
            }
            let l = lower_expr_kernel(lhs, ctx)?;
            let r = lower_expr_kernel(rhs, ctx)?;
            let msl_op = binop_to_msl_infix(op)?;
            Ok(format!("({} {} {})", l, msl_op, r))
        }

        TypedExprKind::Unary { op, operand } => {
            let val = lower_expr_kernel(operand, ctx)?;
            match op {
                UnaryOp::Neg => Ok(format!("(-{})", val)),
                UnaryOp::Not => Ok(format!("(!{})", val)),
            }
        }

        TypedExprKind::Index { base, indices } => {
            // Detect indexed metadata access: `t.shape[k]`, `t.strides[k]`.
            if let TypedExprKind::FieldAccess { base: field_base, field } = &base.kind {
                if field == "shape" || field == "strides" {
                    if indices.len() != 1 {
                        return Err(CodegenError::UnsupportedKernelBody(
                            "shape/strides index must be single-dimensional".into(),
                        ));
                    }
                    let tensor_name = extract_tensor_name(field_base)?;
                    let meta_name = format!("{}_meta", tensor_name);
                    let idx_s = lower_expr_kernel(&indices[0], ctx)?;
                    return Ok(format!("(int){}.{}[{}]", meta_name, field, idx_s));
                }
            }
            // Multi-dim tensor indexing: a[i,j,...] → a[i*strides[0]+j*strides[1]+...]
            if indices.len() > 1 {
                let base_name = extract_tensor_name(base)?;
                let meta_name = format!("{}_meta", base_name);
                let parts: Result<Vec<_>, _> = indices.iter().enumerate().map(|(dim, idx)| {
                    let idx_s = lower_expr_kernel(idx, ctx)?;
                    Ok(format!("({}) * (int){}.strides[{}]", idx_s, meta_name, dim))
                }).collect();
                let flat_idx = parts?.join(" + ");
                return Ok(format!("{}[{}]", base_name, flat_idx));
            }
            let base_s = lower_expr_kernel(base, ctx)?;
            let idx_s = lower_expr_kernel(&indices[0], ctx)?;
            Ok(format!("{}[{}]", base_s, idx_s))
        }

        TypedExprKind::FieldAccess { base, field } => {
            // `t.ndim` → `t_meta.ndim`.
            if field == "ndim" {
                let tensor_name = extract_tensor_name(base)?;
                return Ok(format!("(int){}_meta.ndim", tensor_name));
            }
            // `.shape` and `.strides` without indexing are not directly emittable;
            // they are only valid as the base of an Index expression.
            Err(CodegenError::UnsupportedKernelBody(format!(
                "field '{}' on tensor must be indexed (e.g. t.shape[i])", field
            )))
        }

        TypedExprKind::Call { callee, args } => {
            // Thread-hierarchy intrinsics → cast MSL attribute variable to int.
            if let Some(var) = intrinsic_var(callee) {
                return Ok(format!("(int){}", var));
            }
            // barrier() is handled at statement level; if it appears as an expr, emit the call.
            if callee == "barrier" {
                return Ok("threadgroup_barrier(mem_flags::mem_threadgroup)".to_string());
            }
            // Scalar math builtins → MSL function calls.
            let msl_callee = match callee.as_str() {
                "exp"   => "exp",
                "log"   => "log",
                "sqrt"  => "sqrt",
                "rsqrt" => "rsqrt",
                "tanh"  => "tanh",
                "abs"   => "fabs",
                "relu"  => return {
                    let a = lower_expr_kernel(&args[0], ctx)?;
                    Ok(format!("fmax(0.0f, {})", a))
                },
                "sigmoid" => return {
                    let a = lower_expr_kernel(&args[0], ctx)?;
                    Ok(format!("(1.0f / (1.0f + exp(-{})))", a))
                },
                "fmax"  => "fmax",
                "fmin"  => "fmin",
                "pow"   => "pow",
                other => return Err(CodegenError::UnsupportedKernelBody(format!(
                    "unsupported call in explicit kernel body: {}", other
                ))),
            };
            let arg_strs: Result<Vec<_>, _> = args.iter().map(|a| lower_expr_kernel(a, ctx)).collect();
            Ok(format!("{}({})", msl_callee, arg_strs?.join(", ")))
        }

        other => Err(CodegenError::UnsupportedKernelBody(format!(
            "unsupported expression kind in explicit kernel: {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────────────

fn lower_lit(lit: &Lit) -> Result<String, CodegenError> {
    match lit {
        Lit::Float(f) => Ok(format!("{:?}f", f)),
        Lit::Int(n) => Ok(format!("{}", n)),
        Lit::Bool(b) => Ok(if *b { "true".to_string() } else { "false".to_string() }),
        Lit::Str(_) => Err(CodegenError::UnsupportedKernelBody(
            "string literals not supported in kernel bodies".into(),
        )),
    }
}

fn binop_to_msl_infix(op: &BinOp) -> Result<&'static str, CodegenError> {
    match op {
        BinOp::Add   => Ok("+"),
        BinOp::Sub   => Ok("-"),
        BinOp::Mul   => Ok("*"),
        BinOp::Div   => Ok("/"),
        BinOp::Eq    => Ok("=="),
        BinOp::NotEq => Ok("!="),
        BinOp::Lt    => Ok("<"),
        BinOp::LtEq  => Ok("<="),
        BinOp::Gt    => Ok(">"),
        BinOp::GtEq  => Ok(">="),
        BinOp::And   => Ok("&&"),
        BinOp::Or    => Ok("||"),
        BinOp::Matmul => Err(CodegenError::UnsupportedKernelBody(
            "matmul is not element-wise".into(),
        )),
        BinOp::Pow => Err(CodegenError::UnsupportedKernelBody(
            "Pow must be handled before binop_to_msl_infix".into(),
        )),
    }
}

// Keep the old name as an alias for the synthesize_* functions that still use it.
fn binop_to_msl(op: &BinOp) -> Result<&'static str, CodegenError> {
    binop_to_msl_infix(op)
}

/// Map a resolved type to its MSL type string (scalar types only; tensors not valid here).
fn resolved_ty_to_msl(ty: &ResolvedTy) -> Result<&'static str, CodegenError> {
    match ty {
        ResolvedTy::Scalar(s) => Ok(dtype_to_msl(s)),
        ResolvedTy::Unit => Ok("void"),
        _ => Err(CodegenError::UnsupportedKernelBody(format!(
            "expected scalar type for kernel local, got: {}", ty
        ))),
    }
}

fn dtype_to_msl(dtype: &ScalarTy) -> &'static str {
    match dtype {
        ScalarTy::F32 => "float",
        ScalarTy::F16 => "half",
        ScalarTy::Bf16 => "bfloat",
        ScalarTy::I8 => "char",
        ScalarTy::I16 => "short",
        ScalarTy::I32 => "int",
        ScalarTy::I64 => "long",
        ScalarTy::U8 => "uchar",
        ScalarTy::U16 => "ushort",
        ScalarTy::U32 => "uint",
        ScalarTy::U64 => "ulong",
    }
}
