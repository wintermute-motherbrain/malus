use std::collections::{HashMap, HashSet};

use malus_sema::{TypedExpr, TypedExprKind, TypedKernel, TypedProgram, TypedStmt};
use malus_syntax::ast::{BinOp, ScalarTy, UnaryOp};

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

pub fn compile_kernels(
    program: &TypedProgram,
) -> Result<(KernelRegistry, HashMap<String, u64>), CodegenError> {
    let mut registry = KernelRegistry::new();
    let mut name_to_id = HashMap::new();

    for (id, kernel) in program.kernels.iter().enumerate() {
        let kernel_id = id as u64;
        let msl = lower_kernel(kernel, kernel_id)?;
        registry.insert(kernel_id, msl);
        name_to_id.insert(kernel.name.clone(), kernel_id);
    }

    Ok((registry, name_to_id))
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
    match body {
        [TypedStmt::Return { expr }] => {
            let expr_msl = lower_expr(expr, param_names)?;
            Ok(format!("out[tid] = {};", expr_msl))
        }
        _ => Err(CodegenError::UnsupportedKernelBody(
            "kernel body must be a single return statement".into(),
        )),
    }
}

fn lower_expr(
    expr: &TypedExpr,
    param_names: &HashSet<String>,
) -> Result<String, CodegenError> {
    match &expr.kind {
        TypedExprKind::Ident(name) => {
            if param_names.contains(name) {
                Ok(format!("{}[tid]", name))
            } else {
                Err(CodegenError::UnsupportedKernelBody(format!(
                    "unknown identifier in kernel: {}", name
                )))
            }
        }

        TypedExprKind::BinOp { op, lhs, rhs } => {
            let l = lower_expr(lhs, param_names)?;
            let r = lower_expr(rhs, param_names)?;
            let msl_op = binop_to_msl(op)?;
            Ok(format!("({} {} {})", l, msl_op, r))
        }

        TypedExprKind::Unary { op, operand } => {
            let val = lower_expr(operand, param_names)?;
            match op {
                UnaryOp::Neg => Ok(format!("(-{})", val)),
                UnaryOp::Not => Err(CodegenError::UnsupportedKernelBody(
                    "bitwise not on tensors not supported".into(),
                )),
            }
        }

        _ => Err(CodegenError::UnsupportedKernelBody(format!(
            "unsupported expression in kernel body"
        ))),
    }
}

fn binop_to_msl(op: &BinOp) -> Result<&'static str, CodegenError> {
    match op {
        BinOp::Add => Ok("+"),
        BinOp::Sub => Ok("-"),
        BinOp::Mul => Ok("*"),
        BinOp::Div => Ok("/"),
        BinOp::Matmul => Err(CodegenError::UnsupportedKernelBody(
            "matmul is not element-wise".into(),
        )),
        _ => Err(CodegenError::UnsupportedKernelBody(format!(
            "unsupported binop in kernel: {:?}", op
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
