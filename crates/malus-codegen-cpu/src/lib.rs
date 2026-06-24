// Cranelift JIT backend for `fn` bodies (CPU host functions).
// Lowers malus's typed IR to Cranelift IR and JIT-compiles to native code.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Mutex;

use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Signature, UserFuncName};
use cranelift_codegen::ir::types::{F32, I8, I16, I32, I64};
use cranelift_codegen::settings::{self, Configurable};

use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};

use malus_sema::{ResolvedTy, TypedExprKind, TypedFn, TypedProgram, TypedStmt};
use malus_syntax::ast::{BinOp, Lit, ScalarTy, UnaryOp};

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CodegenError {
    NoMainFunction,
    UnsupportedExpr(String),
    UnsupportedType(String),
    JitError(String),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::NoMainFunction => write!(f, "no fn main() found"),
            CodegenError::UnsupportedExpr(s) => write!(f, "unsupported expression: {s}"),
            CodegenError::UnsupportedType(s) => write!(f, "unsupported type: {s}"),
            CodegenError::JitError(s) => write!(f, "JIT error: {s}"),
        }
    }
}

// ── Runtime stubs ─────────────────────────────────────────────────────────────

struct TensorStore {
    data: HashMap<i64, Vec<f32>>,
    next_id: i64,
}

impl TensorStore {
    fn new() -> Self {
        Self { data: HashMap::new(), next_id: 1 }
    }

    fn insert(&mut self, elements: Vec<f32>) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        self.data.insert(id, elements);
        id
    }
}

static TENSOR_STORE: Mutex<Option<TensorStore>> = Mutex::new(None);

fn with_store<R>(f: impl FnOnce(&mut TensorStore) -> R) -> R {
    let mut guard = TENSOR_STORE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.as_mut().expect("tensor store not initialized"))
}

extern "C" fn tensor_alloc_gpu(dtype: i32, len: i64, data: *const f32) -> i64 {
    let _ = dtype;
    let elements = if data.is_null() || len == 0 {
        vec![]
    } else {
        unsafe { std::slice::from_raw_parts(data, len as usize).to_vec() }
    };
    with_store(|s| s.insert(elements))
}

extern "C" fn tensor_print(handle: i64) {
    let elems = with_store(|s| s.data.get(&handle).cloned().unwrap_or_default());
    print!("[");
    for (i, v) in elems.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{v}");
    }
    println!("]");
}

extern "C" fn tensor_free(handle: i64) {
    with_store(|s| { s.data.remove(&handle); });
}

extern "C" fn kernel_dispatch(_name: *const u8, _handles: *const i64, _n: i32) -> i64 {
    with_store(|s| s.insert(vec![]))
}

extern "C" fn gpu_barrier() {}

// ── dtype_tag — ScalarTy enum discriminant order ──────────────────────────────

fn dtype_tag(s: &ScalarTy) -> i32 {
    match s {
        ScalarTy::F32  => 0,
        ScalarTy::F16  => 1,
        ScalarTy::Bf16 => 2,
        ScalarTy::I8   => 3,
        ScalarTy::I16  => 4,
        ScalarTy::I32  => 5,
        ScalarTy::I64  => 6,
        ScalarTy::U8   => 7,
        ScalarTy::U16  => 8,
        ScalarTy::U32  => 9,
        ScalarTy::U64  => 10,
    }
}

// ── Type mapping ──────────────────────────────────────────────────────────────

fn cranelift_type(ty: &ResolvedTy) -> Result<Option<cranelift_codegen::ir::Type>, CodegenError> {
    match ty {
        ResolvedTy::Tensor { .. } => Ok(Some(I64)),
        ResolvedTy::Scalar(s) => Ok(Some(scalar_cranelift_type(s))),
        ResolvedTy::Bool => Ok(Some(I8)),
        ResolvedTy::Unit => Ok(None),
        ResolvedTy::Tuple(_) => Err(CodegenError::UnsupportedType("tuple".into())),
    }
}

fn scalar_cranelift_type(s: &ScalarTy) -> cranelift_codegen::ir::Type {
    match s {
        ScalarTy::F32 | ScalarTy::F16 | ScalarTy::Bf16 => F32,
        ScalarTy::I8  | ScalarTy::U8  => I8,
        ScalarTy::I16 | ScalarTy::U16 => I16,
        ScalarTy::I32 | ScalarTy::U32 => I32,
        ScalarTy::I64 | ScalarTy::U64 => I64,
    }
}

fn is_float_scalar(s: &ScalarTy) -> bool {
    matches!(s, ScalarTy::F32 | ScalarTy::F16 | ScalarTy::Bf16)
}

// ── Codegen context ───────────────────────────────────────────────────────────

struct Codegen<'m> {
    module: &'m mut JITModule,
    func_ids: HashMap<String, FuncId>,
    rt_tensor_alloc_gpu: FuncId,
    rt_tensor_print: FuncId,
    rt_tensor_free: FuncId,
    rt_kernel_dispatch: FuncId,
    rt_gpu_barrier: FuncId,
}

impl<'m> Codegen<'m> {
    fn ptr_type(&self) -> cranelift_codegen::ir::Type {
        self.module.target_config().pointer_type()
    }

    fn compile_fn(&mut self, typed_fn: &TypedFn) -> Result<(), CodegenError> {
        let func_id = self.func_ids[&typed_fn.name];

        let mut sig = Signature::new(self.module.target_config().default_call_conv);
        for param in &typed_fn.params {
            if let Some(t) = cranelift_type(&param.ty)? {
                sig.params.push(AbiParam::new(t));
            }
        }
        if let Some(ret_t) = cranelift_type(&typed_fn.return_ty)? {
            sig.returns.push(AbiParam::new(ret_t));
        }

        let mut func = Function::with_name_signature(
            UserFuncName::user(0, func_id.as_u32()),
            sig,
        );

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let mut var_map: HashMap<String, Variable> = HashMap::new();
        let mut next_var = 0usize;

        let param_vals: Vec<_> = builder.block_params(entry).to_vec();
        for (i, param) in typed_fn.params.iter().enumerate() {
            if let Some(t) = cranelift_type(&param.ty)? {
                let var = Variable::from_u32(next_var as u32);
                next_var += 1;
                builder.declare_var(var, t);
                builder.def_var(var, param_vals[i]);
                var_map.insert(param.name.clone(), var);
            }
        }

        let mut translator = FnTranslator {
            builder,
            var_map,
            next_var,
            codegen: self,
        };

        let mut returned = false;
        for stmt in &typed_fn.body {
            if translator.lower_stmt(stmt)? {
                returned = true;
                break;
            }
        }

        if !returned {
            translator.builder.ins().return_(&[]);
        }

        translator.builder.finalize();

        let mut ctx = cranelift_codegen::Context::for_function(func);
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CodegenError::JitError(e.to_string()))?;

        Ok(())
    }
}

// ── Per-function translator ───────────────────────────────────────────────────

struct FnTranslator<'a, 'm> {
    builder: FunctionBuilder<'a>,
    var_map: HashMap<String, Variable>,
    next_var: usize,
    codegen: &'a mut Codegen<'m>,
}

impl<'a, 'm> FnTranslator<'a, 'm> {
    // Returns true when the statement terminates the block (Return).
    fn lower_stmt(&mut self, stmt: &TypedStmt) -> Result<bool, CodegenError> {
        match stmt {
            TypedStmt::Let { name, expr } => {
                let val = self.lower_expr(expr)?;
                let t = match cranelift_type(&expr.ty)? {
                    Some(t) => t,
                    None => return Ok(false),
                };
                let var = Variable::from_u32(self.next_var as u32);
                self.next_var += 1;
                self.builder.declare_var(var, t);
                self.builder.def_var(var, val);
                self.var_map.insert(name.clone(), var);
                Ok(false)
            }

            TypedStmt::Return { expr } => {
                let val = self.lower_expr(expr)?;
                self.builder.ins().return_(&[val]);
                Ok(true)
            }

            TypedStmt::Expr(expr) => {
                self.lower_expr(expr)?;
                Ok(false)
            }

            TypedStmt::Drop { name } => {
                let handle = self.use_var(name)?;
                self.call_runtime_free(handle);
                Ok(false)
            }

            TypedStmt::GpuBarrier => {
                self.call_runtime_barrier();
                Ok(false)
            }
        }
    }

    fn use_var(&mut self, name: &str) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let var = self.var_map.get(name)
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr(format!("unknown variable: {name}")))?;
        Ok(self.builder.use_var(var))
    }

    fn import_func(&mut self, func_id: FuncId) -> cranelift_codegen::ir::FuncRef {
        self.codegen.module.declare_func_in_func(func_id, self.builder.func)
    }

    fn lower_expr(&mut self, expr: &malus_sema::TypedExpr) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        match &expr.kind {
            TypedExprKind::Lit(lit) => self.lower_lit(lit, &expr.ty),

            TypedExprKind::Ident(name) => self.use_var(name),

            TypedExprKind::BinOp { op, lhs, rhs } => {
                match &lhs.ty {
                    ResolvedTy::Tensor { .. } => Err(CodegenError::UnsupportedExpr(
                        "tensor BinOp in host fns not yet supported (pending language design on MPS dispatch)".into()
                    )),
                    ResolvedTy::Scalar(s) => {
                        let l = self.lower_expr(lhs)?;
                        let r = self.lower_expr(rhs)?;
                        self.lower_scalar_binop(op, l, r, s)
                    }
                    ResolvedTy::Bool => {
                        let l = self.lower_expr(lhs)?;
                        let r = self.lower_expr(rhs)?;
                        self.lower_bool_binop(op, l, r)
                    }
                    _ => Err(CodegenError::UnsupportedExpr(format!("BinOp on {}", lhs.ty))),
                }
            }

            TypedExprKind::Unary { op, operand } => {
                let val = self.lower_expr(operand)?;
                match op {
                    UnaryOp::Neg => match &operand.ty {
                        ResolvedTy::Scalar(s) if is_float_scalar(s) => Ok(self.builder.ins().fneg(val)),
                        ResolvedTy::Scalar(_) => Ok(self.builder.ins().ineg(val)),
                        _ => Err(CodegenError::UnsupportedExpr("Neg on non-scalar".into())),
                    },
                    UnaryOp::Not => {
                        let one = self.builder.ins().iconst(I8, 1);
                        Ok(self.builder.ins().bxor(val, one))
                    }
                }
            }

            TypedExprKind::Call { callee, args } => {
                if callee == "print" {
                    for arg in args {
                        let handle = self.lower_expr(arg)?;
                        self.call_runtime_print(handle);
                    }
                    Ok(self.builder.ins().iconst(I64, 0))
                } else {
                    let func_id = self.codegen.func_ids.get(callee)
                        .copied()
                        .ok_or_else(|| CodegenError::UnsupportedExpr(format!("unknown callee: {callee}")))?;
                    let func_ref = self.import_func(func_id);
                    let mut arg_vals = Vec::new();
                    for arg in args {
                        arg_vals.push(self.lower_expr(arg)?);
                    }
                    let call = self.builder.ins().call(func_ref, &arg_vals);
                    let results = self.builder.inst_results(call).to_vec();
                    if results.is_empty() {
                        Ok(self.builder.ins().iconst(I64, 0))
                    } else {
                        Ok(results[0])
                    }
                }
            }

            TypedExprKind::KernelCall { callee, args, .. } => {
                let n = args.len() as u32;
                // Stack slot for the handles array (n * 8 bytes, 8-byte aligned).
                let slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        n * 8,
                        3,
                    )
                );
                for (i, arg) in args.iter().enumerate() {
                    let val = self.lower_expr(arg)?;
                    self.builder.ins().stack_store(val, slot, (i as i32) * 8);
                }

                let name_ptr = self.emit_static_cstr(callee);
                let handles_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
                let n_val = self.builder.ins().iconst(I32, n as i64);

                let dispatch_ref = self.import_func(self.codegen.rt_kernel_dispatch);
                let call = self.builder.ins().call(dispatch_ref, &[name_ptr, handles_ptr, n_val]);
                let results = self.builder.inst_results(call).to_vec();
                Ok(results[0])
            }

            TypedExprKind::TensorLiteral { dtype, elements, .. } => {
                let len = elements.len() as u32;
                // Stack slot for f32 elements (len * 4 bytes, 4-byte aligned).
                let slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        len * 4,
                        2,
                    )
                );
                for (i, elem) in elements.iter().enumerate() {
                    let val = self.lower_expr(elem)?;
                    // Widen integer literals to f32 for the data buffer.
                    let f32_val = match &elem.ty {
                        ResolvedTy::Scalar(s) if !is_float_scalar(s) => {
                            self.builder.ins().fcvt_from_sint(F32, val)
                        }
                        _ => val,
                    };
                    self.builder.ins().stack_store(f32_val, slot, (i as i32) * 4);
                }

                let data_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
                let dtype_val = self.builder.ins().iconst(I32, dtype_tag(dtype) as i64);
                let len_val = self.builder.ins().iconst(I64, len as i64);

                let alloc_ref = self.import_func(self.codegen.rt_tensor_alloc_gpu);
                let call = self.builder.ins().call(alloc_ref, &[dtype_val, len_val, data_ptr]);
                let results = self.builder.inst_results(call).to_vec();
                Ok(results[0])
            }

            TypedExprKind::Index { .. } => {
                Err(CodegenError::UnsupportedExpr("Index not yet supported".into()))
            }
            TypedExprKind::FieldAccess { .. } => {
                Err(CodegenError::UnsupportedExpr("FieldAccess not yet supported".into()))
            }
        }
    }

    fn lower_lit(&mut self, lit: &Lit, ty: &ResolvedTy) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        match lit {
            Lit::Int(n) => match ty {
                ResolvedTy::Scalar(s) if is_float_scalar(s) => {
                    Ok(self.builder.ins().f32const(*n as f32))
                }
                ResolvedTy::Scalar(s) => {
                    let t = scalar_cranelift_type(s);
                    Ok(self.builder.ins().iconst(t, *n))
                }
                _ => Ok(self.builder.ins().f32const(*n as f32)),
            },
            Lit::Float(f) => Ok(self.builder.ins().f32const(*f as f32)),
            Lit::Bool(b) => Ok(self.builder.ins().iconst(I8, *b as i64)),
            Lit::Str(_) => Err(CodegenError::UnsupportedExpr("string literal in codegen".into())),
        }
    }

    fn lower_scalar_binop(
        &mut self,
        op: &BinOp,
        l: cranelift_codegen::ir::Value,
        r: cranelift_codegen::ir::Value,
        scalar: &ScalarTy,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
        let float = is_float_scalar(scalar);
        match op {
            BinOp::Add => Ok(if float { self.builder.ins().fadd(l, r) } else { self.builder.ins().iadd(l, r) }),
            BinOp::Sub => Ok(if float { self.builder.ins().fsub(l, r) } else { self.builder.ins().isub(l, r) }),
            BinOp::Mul => Ok(if float { self.builder.ins().fmul(l, r) } else { self.builder.ins().imul(l, r) }),
            BinOp::Div => Ok(if float { self.builder.ins().fdiv(l, r) } else { self.builder.ins().sdiv(l, r) }),
            BinOp::Eq => {
                let cmp = if float { self.builder.ins().fcmp(FloatCC::Equal, l, r) } else { self.builder.ins().icmp(IntCC::Equal, l, r) };
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            BinOp::NotEq => {
                let cmp = if float { self.builder.ins().fcmp(FloatCC::NotEqual, l, r) } else { self.builder.ins().icmp(IntCC::NotEqual, l, r) };
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            BinOp::Lt => {
                let cmp = if float { self.builder.ins().fcmp(FloatCC::LessThan, l, r) } else { self.builder.ins().icmp(IntCC::SignedLessThan, l, r) };
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            BinOp::LtEq => {
                let cmp = if float { self.builder.ins().fcmp(FloatCC::LessThanOrEqual, l, r) } else { self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r) };
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            BinOp::Gt => {
                let cmp = if float { self.builder.ins().fcmp(FloatCC::GreaterThan, l, r) } else { self.builder.ins().icmp(IntCC::SignedGreaterThan, l, r) };
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            BinOp::GtEq => {
                let cmp = if float { self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r) } else { self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, l, r) };
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            BinOp::Matmul => Err(CodegenError::UnsupportedExpr("Matmul not supported in host fns".into())),
            BinOp::And | BinOp::Or => Err(CodegenError::UnsupportedExpr("And/Or on scalars not supported".into())),
        }
    }

    fn lower_bool_binop(
        &mut self,
        op: &BinOp,
        l: cranelift_codegen::ir::Value,
        r: cranelift_codegen::ir::Value,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        use cranelift_codegen::ir::condcodes::IntCC;
        match op {
            BinOp::And => Ok(self.builder.ins().band(l, r)),
            BinOp::Or  => Ok(self.builder.ins().bor(l, r)),
            BinOp::Eq  => {
                let cmp = self.builder.ins().icmp(IntCC::Equal, l, r);
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            BinOp::NotEq => {
                let cmp = self.builder.ins().icmp(IntCC::NotEqual, l, r);
                Ok(self.builder.ins().bmask(I8, cmp))
            }
            _ => Err(CodegenError::UnsupportedExpr(format!("BinOp {op:?} on bool not supported"))),
        }
    }

    fn emit_static_cstr(&mut self, s: &str) -> cranelift_codegen::ir::Value {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        let mut desc = DataDescription::new();
        desc.define(bytes.into_boxed_slice());
        let id = self.codegen.module
            .declare_anonymous_data(false, false)
            .expect("anonymous data declaration failed");
        self.codegen.module.define_data(id, &desc).expect("data definition failed");
        let global = self.codegen.module.declare_data_in_func(id, self.builder.func);
        self.builder.ins().global_value(self.codegen.ptr_type(), global)
    }

    fn call_runtime_print(&mut self, handle: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_tensor_print);
        self.builder.ins().call(func_ref, &[handle]);
    }

    fn call_runtime_free(&mut self, handle: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_tensor_free);
        self.builder.ins().call(func_ref, &[handle]);
    }

    fn call_runtime_barrier(&mut self) {
        let func_ref = self.import_func(self.codegen.rt_gpu_barrier);
        self.builder.ins().call(func_ref, &[]);
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn compile_and_run(program: &TypedProgram) -> Result<(), CodegenError> {
    if !program.fns.iter().any(|f| f.name == "main") {
        return Err(CodegenError::NoMainFunction);
    }

    *TENSOR_STORE.lock().unwrap() = Some(TensorStore::new());

    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let isa = cranelift_native::builder()
        .map_err(|e| CodegenError::JitError(e.to_string()))?
        .finish(flags)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    jit_builder.symbol("tensor_alloc_gpu", tensor_alloc_gpu as *const u8);
    jit_builder.symbol("tensor_print",     tensor_print     as *const u8);
    jit_builder.symbol("tensor_free",      tensor_free      as *const u8);
    jit_builder.symbol("kernel_dispatch",  kernel_dispatch  as *const u8);
    jit_builder.symbol("gpu_barrier",      gpu_barrier      as *const u8);

    let mut module = JITModule::new(jit_builder);
    let ptr = module.target_config().pointer_type();
    let call_conv = module.target_config().default_call_conv;

    // Declare runtime extern signatures.
    let sig_alloc = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I32));
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(ptr));
        s.returns.push(AbiParam::new(I64));
        s
    };
    let sig_print = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s
    };
    let sig_free = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s
    };
    let sig_dispatch = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(ptr));
        s.params.push(AbiParam::new(ptr));
        s.params.push(AbiParam::new(I32));
        s.returns.push(AbiParam::new(I64));
        s
    };
    let sig_barrier = Signature::new(call_conv);

    let rt_tensor_alloc_gpu = module.declare_function("tensor_alloc_gpu", Linkage::Import, &sig_alloc)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_print = module.declare_function("tensor_print", Linkage::Import, &sig_print)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_free = module.declare_function("tensor_free", Linkage::Import, &sig_free)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_kernel_dispatch = module.declare_function("kernel_dispatch", Linkage::Import, &sig_dispatch)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_gpu_barrier = module.declare_function("gpu_barrier", Linkage::Import, &sig_barrier)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // First pass: declare all user fn signatures.
    let mut func_ids: HashMap<String, FuncId> = HashMap::new();
    for typed_fn in &program.fns {
        let mut sig = Signature::new(call_conv);
        for param in &typed_fn.params {
            if let Some(t) = cranelift_type(&param.ty)? {
                sig.params.push(AbiParam::new(t));
            }
        }
        if let Some(ret_t) = cranelift_type(&typed_fn.return_ty)? {
            sig.returns.push(AbiParam::new(ret_t));
        }
        let id = module.declare_function(&typed_fn.name, Linkage::Local, &sig)
            .map_err(|e| CodegenError::JitError(e.to_string()))?;
        func_ids.insert(typed_fn.name.clone(), id);
    }

    let mut cg = Codegen {
        module: &mut module,
        func_ids,
        rt_tensor_alloc_gpu,
        rt_tensor_print,
        rt_tensor_free,
        rt_kernel_dispatch,
        rt_gpu_barrier,
    };

    // Second pass: compile each fn body.
    let fns: Vec<TypedFn> = program.fns.clone();
    for typed_fn in &fns {
        cg.compile_fn(typed_fn)?;
    }

    cg.module.finalize_definitions()
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    let main_id = cg.func_ids["main"];
    let main_ptr = cg.module.get_finalized_function(main_id);
    let main_fn = unsafe { std::mem::transmute::<_, fn()>(main_ptr) };
    main_fn();

    Ok(())
}
