use std::collections::{BTreeSet, HashMap, HashSet};

use malus_sema::{ResolvedTy, TypedExpr, TypedExprKind, TypedKernel, TypedProgram, TypedStmt};
use malus_syntax::ast::{
    elementwise_builtin_name, scalar_broadcast_builtin_name, BinOp, Lit, ScalarTy, UnaryOp,
};

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

    let mut next_id: u64 = 0;

    for kernel in &program.kernels {
        let kernel_id = next_id;
        next_id += 1;
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
        let kernel_id = next_id;
        next_id += 1;
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
        let kernel_id = next_id;
        next_id += 1;
        let msl = synthesize_scalar_builtin(*op, *scalar_on_right, kernel_id)?;
        registry.insert(kernel_id, msl);
        name_to_id.insert(name.to_string(), kernel_id);
    }

    // Unary math builtins (sorted, appended after tensor/scalar builtins per ADR-0010).
    for name in &unary_ops {
        if name_to_id.contains_key(name.as_str()) {
            continue;
        }
        let kernel_id = next_id;
        next_id += 1;
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
        | TypedStmt::DropArray { .. } | TypedStmt::GpuBarrier
        | TypedStmt::Retain { .. } | TypedStmt::Release { .. }
        | TypedStmt::RetainAgg { .. } | TypedStmt::ReleaseAgg { .. } => {}
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
    }
}

fn collect_binops_in_expr(
    expr: &TypedExpr,
    tensor_ops: &mut BTreeSet<BinOp>,
    scalar_ops: &mut BTreeSet<(BinOp, bool)>,
) {
    match &expr.kind {
        TypedExprKind::BinOp { op, lhs, rhs } => {
            if lhs.ty.is_tensor() && rhs.ty.is_tensor() {
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
        | TypedStmt::DropArray { .. } | TypedStmt::GpuBarrier
        | TypedStmt::Retain { .. } | TypedStmt::Release { .. }
        | TypedStmt::RetainAgg { .. } | TypedStmt::ReleaseAgg { .. } => {}
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

    let body_msl = lower_kernel_body(&kernel.body, &param_names)?;

    let msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void {}(\n    {}\n) {{\n    {}\n}}\n",
        func_name,
        params.join(",\n    "),
        body_msl,
    );

    Ok(msl)
}

fn lower_kernel_body(
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
                let msl_ty = resolved_scalar_to_msl(&expr.ty)?;
                let expr_msl = lower_expr(expr, param_names, &local_names)?;
                lines.push(format!("{} {} = {};", msl_ty, name, expr_msl));
                local_names.insert(name.clone());
            }
            TypedStmt::Return { expr } => {
                if !is_last {
                    return Err(CodegenError::UnsupportedKernelBody(
                        "return must be the last statement in kernel body".into(),
                    ));
                }
                let expr_msl = lower_expr(expr, param_names, &local_names)?;
                lines.push(format!("out[tid] = {};", expr_msl));
            }
            _ => {
                return Err(CodegenError::UnsupportedKernelBody(
                    "only let bindings and a final return are allowed in kernel bodies".into(),
                ));
            }
        }
    }

    Ok(lines.join("\n    "))
}

fn lower_expr(
    expr: &TypedExpr,
    param_names: &HashSet<String>,
    local_names: &HashSet<String>,
) -> Result<String, CodegenError> {
    match &expr.kind {
        TypedExprKind::Ident(name) => {
            if param_names.contains(name) {
                // Tensor parameters: index by thread id (element-space).
                Ok(format!("{}[tid]", name))
            } else if local_names.contains(name) {
                // Let-bound locals: scalar value, no indexing.
                Ok(name.clone())
            } else {
                Err(CodegenError::UnsupportedKernelBody(format!(
                    "unknown identifier in kernel: {}", name
                )))
            }
        }

        TypedExprKind::Lit(lit) => match lit {
            Lit::Float(f) => Ok(format!("{:?}f", f)),
            Lit::Int(n) => Ok(format!("{}", n)),
            Lit::Bool(_) => Err(CodegenError::UnsupportedKernelBody(
                "bool literals not supported in kernel bodies (comparisons yield float masks)".into(),
            )),
            Lit::Str(_) => Err(CodegenError::UnsupportedKernelBody(
                "string literals not supported in kernel bodies".into(),
            )),
        },

        TypedExprKind::BinOp { op, lhs, rhs } => {
            let l = lower_expr(lhs, param_names, local_names)?;
            let r = lower_expr(rhs, param_names, local_names)?;
            let msl_op = binop_to_msl(op)?;
            Ok(format!("({} {} {})", l, msl_op, r))
        }

        TypedExprKind::Unary { op, operand } => {
            let val = lower_expr(operand, param_names, local_names)?;
            match op {
                UnaryOp::Neg => Ok(format!("(-{})", val)),
                UnaryOp::Not => Err(CodegenError::UnsupportedKernelBody(
                    "bitwise not not supported in kernel bodies".into(),
                )),
            }
        }

        _ => Err(CodegenError::UnsupportedKernelBody(format!(
            "unsupported expression kind in kernel body"
        ))),
    }
}

fn binop_to_msl(op: &BinOp) -> Result<&'static str, CodegenError> {
    match op {
        BinOp::Add => Ok("+"),
        BinOp::Sub => Ok("-"),
        BinOp::Mul => Ok("*"),
        BinOp::Div => Ok("/"),
        BinOp::Eq    => Ok("=="),
        BinOp::NotEq => Ok("!="),
        BinOp::Lt    => Ok("<"),
        BinOp::LtEq  => Ok("<="),
        BinOp::Gt    => Ok(">"),
        BinOp::GtEq  => Ok(">="),
        BinOp::Matmul => Err(CodegenError::UnsupportedKernelBody(
            "matmul is not element-wise".into(),
        )),
        _ => Err(CodegenError::UnsupportedKernelBody(format!(
            "unsupported binop in kernel: {:?}", op
        ))),
    }
}

/// Map a resolved scalar type to its MSL type name.
fn resolved_scalar_to_msl(ty: &ResolvedTy) -> Result<&'static str, CodegenError> {
    match ty {
        ResolvedTy::Scalar(s) => Ok(dtype_to_msl(s)),
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
