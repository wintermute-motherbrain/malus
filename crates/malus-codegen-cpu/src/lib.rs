// Cranelift JIT backend for `fn` bodies (CPU host functions).
// Lowers malus's typed IR to Cranelift IR and JIT-compiles to native code.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Signature, UserFuncName};
use cranelift_codegen::ir::types::{F32, I8, I16, I32, I64};
use cranelift_codegen::settings::{self, Configurable};

use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};

use malus_sema::{ResolvedTy, TypedExpr, TypedExprKind, TypedFn, TypedProgram, TypedStmt};
use malus_syntax::ast::{elementwise_builtin_name, scalar_broadcast_builtin_name, BinOp, Lit, ScalarTy, UnaryOp};

// ── Runtime symbol injection ─────────────────────────────────────────────────

#[repr(C)]
pub struct RuntimeSymbols {
    pub tensor_alloc_gpu:       extern "C" fn(i32, *const usize, usize, *const f32) -> i64,
    pub tensor_free:            extern "C" fn(i64),
    pub tensor_print:           extern "C" fn(i64),
    pub kernel_dispatch:        extern "C" fn(u64, *const i64, usize) -> i64,
    pub gpu_barrier:            extern "C" fn(),
    pub tensor_alloc_zeros_gpu: extern "C" fn(*const usize, usize) -> i64,
    pub tensor_alloc_ones_gpu:  extern "C" fn(*const usize, usize) -> i64,
    pub tensor_matmul:          extern "C" fn(i64, i64) -> i64,
    pub tensor_transpose:       extern "C" fn(i64) -> i64,
    pub tensor_sum:             extern "C" fn(i64) -> i64,
    pub tensor_len:             extern "C" fn(i64) -> i64,
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CodegenError {
    NoMainFunction,
    UnsupportedExpr(String),
    UnsupportedType(String),
    UnknownKernel { name: String },
    JitError(String),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::NoMainFunction => write!(f, "no fn main() found"),
            CodegenError::UnsupportedExpr(s) => write!(f, "unsupported expression: {s}"),
            CodegenError::UnsupportedType(s) => write!(f, "unsupported type: {s}"),
            CodegenError::UnknownKernel { name } => write!(f, "unknown kernel: {name}"),
            CodegenError::JitError(s) => write!(f, "JIT error: {s}"),
        }
    }
}

// ── Host print helpers ───────────────────────────────────────────────────────

extern "C" fn print_cstr(ptr: *const u8) {
    let s = unsafe { std::ffi::CStr::from_ptr(ptr as *const i8) };
    print!("{}", s.to_str().unwrap_or("<invalid utf-8>"));
}

extern "C" fn print_f32(v: f32)  { print!("{v}"); }
extern "C" fn print_i64(v: i64)  { print!("{v}"); }
extern "C" fn print_bool(v: i8)  { print!("{}", if v != 0 { "true" } else { "false" }); }

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

// ── Format string helpers ─────────────────────────────────────────────────────

enum FormatSegment {
    Literal(String),
    Placeholder,
}

fn parse_format_string(s: &str) -> Vec<FormatSegment> {
    let mut segments = Vec::new();
    let mut rest = s;
    while let Some(idx) = rest.find("{}") {
        if idx > 0 {
            segments.push(FormatSegment::Literal(rest[..idx].to_string()));
        }
        segments.push(FormatSegment::Placeholder);
        rest = &rest[idx + 2..];
    }
    if !rest.is_empty() {
        segments.push(FormatSegment::Literal(rest.to_string()));
    }
    segments
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
    kernel_ids: &'m HashMap<String, u64>,
    rt_tensor_alloc_gpu: FuncId,
    rt_tensor_print: FuncId,
    rt_tensor_free: FuncId,
    rt_kernel_dispatch: FuncId,
    rt_gpu_barrier: FuncId,
    rt_tensor_alloc_zeros_gpu: FuncId,
    rt_tensor_alloc_ones_gpu: FuncId,
    rt_tensor_matmul: FuncId,
    rt_tensor_transpose: FuncId,
    rt_tensor_sum: FuncId,
    rt_tensor_len: FuncId,
    rt_print_cstr: FuncId,
    rt_print_f32: FuncId,
    rt_print_i64: FuncId,
    rt_print_bool: FuncId,
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

            TypedStmt::Assign { name, expr } => {
                let val = self.lower_expr(expr)?;
                let var = self.var_map.get(name)
                    .copied()
                    .ok_or_else(|| CodegenError::UnsupportedExpr(format!("assign to unknown variable: {name}")))?;
                self.builder.def_var(var, val);
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

    fn lower_kernel_dispatch(
        &mut self,
        callee: &str,
        args: &[&TypedExpr],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let kernel_id = *self.codegen.kernel_ids.get(callee)
            .ok_or_else(|| CodegenError::UnknownKernel { name: callee.to_string() })?;
        let n = args.len() as u32;
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

        let id_val = self.builder.ins().iconst(I64, kernel_id as i64);
        let handles_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
        let n_val = self.builder.ins().iconst(self.codegen.ptr_type(), n as i64);

        let dispatch_ref = self.import_func(self.codegen.rt_kernel_dispatch);
        let call = self.builder.ins().call(dispatch_ref, &[id_val, handles_ptr, n_val]);
        let results = self.builder.inst_results(call).to_vec();
        Ok(results[0])
    }

    fn lower_expr(&mut self, expr: &malus_sema::TypedExpr) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        match &expr.kind {
            TypedExprKind::Lit(lit) => self.lower_lit(lit, &expr.ty),

            TypedExprKind::Ident(name) => self.use_var(name),

            TypedExprKind::BinOp { op, lhs, rhs } => {
                match (&lhs.ty, &rhs.ty) {
                    // tensor ⊕ tensor — matmul or element-wise builtin kernel
                    (ResolvedTy::Tensor { dtype: ld }, ResolvedTy::Tensor { dtype: rd })
                        if ld == rd && *ld == ScalarTy::F32 =>
                    {
                        if *op == BinOp::Matmul {
                            let a = self.lower_expr(lhs)?;
                            let b = self.lower_expr(rhs)?;
                            let matmul_ref = self.import_func(self.codegen.rt_tensor_matmul);
                            let call = self.builder.ins().call(matmul_ref, &[a, b]);
                            Ok(self.builder.inst_results(call).to_vec()[0])
                        } else if let Some(name) = elementwise_builtin_name(op) {
                            self.lower_kernel_dispatch(name, &[lhs.as_ref(), rhs.as_ref()])
                        } else {
                            Err(CodegenError::UnsupportedExpr(format!("binop {:?} on tensors not supported", op)))
                        }
                    }
                    (ResolvedTy::Tensor { .. }, ResolvedTy::Tensor { .. }) => {
                        Err(CodegenError::UnsupportedExpr("non-f32 tensor BinOp not yet supported".into()))
                    }
                    // tensor ⊕ scalar — materialize scalar as 1-elem tensor, dispatch scalar builtin
                    (ResolvedTy::Tensor { dtype }, ResolvedTy::Scalar(sd)) if dtype == sd && *dtype == ScalarTy::F32 => {
                        if let Some(name) = scalar_broadcast_builtin_name(op, true) {
                            let tensor_handle = self.lower_expr(lhs)?;
                            let scalar_handle = self.materialize_scalar_tensor(rhs)?;
                            self.lower_kernel_dispatch_with_handles(name, &[tensor_handle, scalar_handle])
                        } else {
                            Err(CodegenError::UnsupportedExpr(format!("scalar broadcast {:?} not supported", op)))
                        }
                    }
                    // scalar ⊕ tensor — scalar-on-left: buffer layout is [tensor, scalar, out]
                    (ResolvedTy::Scalar(sd), ResolvedTy::Tensor { dtype }) if sd == dtype && *dtype == ScalarTy::F32 => {
                        if let Some(name) = scalar_broadcast_builtin_name(op, false) {
                            let tensor_handle = self.lower_expr(rhs)?;
                            let scalar_handle = self.materialize_scalar_tensor(lhs)?;
                            self.lower_kernel_dispatch_with_handles(name, &[tensor_handle, scalar_handle])
                        } else {
                            Err(CodegenError::UnsupportedExpr(format!("scalar broadcast {:?} (reversed) not supported", op)))
                        }
                    }
                    // scalar ⊕ scalar
                    (ResolvedTy::Scalar(s), _) => {
                        let l = self.lower_expr(lhs)?;
                        let r = self.lower_expr(rhs)?;
                        self.lower_scalar_binop(op, l, r, s)
                    }
                    (ResolvedTy::Bool, _) => {
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
                if callee == "print" || callee == "println" {
                    let is_println = callee == "println";

                    if args.is_empty() {
                        // println() with no args → just a newline
                        if is_println {
                            let ptr = self.emit_static_cstr("\n");
                            self.call_print_cstr(ptr);
                        }
                    } else if let TypedExprKind::Lit(Lit::Str(fmt)) = &args[0].kind {
                        // Format string mode: compile-time expand
                        let segments = parse_format_string(fmt);
                        let mut val_idx = 0usize;
                        for seg in &segments {
                            match seg {
                                FormatSegment::Literal(text) => {
                                    let ptr = self.emit_static_cstr(text);
                                    self.call_print_cstr(ptr);
                                }
                                FormatSegment::Placeholder => {
                                    let arg = &args[1 + val_idx];
                                    self.emit_print_value(arg)?;
                                    val_idx += 1;
                                }
                            }
                        }
                        if is_println {
                            let ptr = self.emit_static_cstr("\n");
                            self.call_print_cstr(ptr);
                        }
                    } else {
                        // Legacy: print each arg by type, no separator
                        for arg in args {
                            self.emit_print_value(arg)?;
                        }
                        if is_println {
                            let ptr = self.emit_static_cstr("\n");
                            self.call_print_cstr(ptr);
                        }
                    }

                    Ok(self.builder.ins().iconst(I64, 0))
                } else if self.codegen.kernel_ids.contains_key(callee.as_str()) {
                    // Unary math builtins (relu, sigmoid, tanh, exp, log, sqrt, abs)
                    // dispatched as built-in GPU kernels via kernel_dispatch.
                    let arg_refs: Vec<&TypedExpr> = args.iter().collect();
                    self.lower_kernel_dispatch(callee, &arg_refs)
                } else if callee == "zeros" || callee == "ones" {
                    self.lower_zeros_ones(callee, args)
                } else if callee == "transpose" || callee == "sum" {
                    self.lower_eager_cpu_op(callee, args)
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
                let arg_refs: Vec<&TypedExpr> = args.iter().collect();
                self.lower_kernel_dispatch(callee, &arg_refs)
            }

            TypedExprKind::TensorLiteral { dtype, elements, .. } => {
                let len = elements.len() as u32;
                // Stack slot for f32 elements (len * 4 bytes, 4-byte aligned).
                let data_slot = self.builder.create_sized_stack_slot(
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
                    self.builder.ins().stack_store(f32_val, data_slot, (i as i32) * 4);
                }

                // Shape slot: one usize (8 bytes, 8-byte aligned) holding the element count.
                let shape_slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        8,
                        3,
                    )
                );
                let len_as_usize = self.builder.ins().iconst(I64, len as i64);
                self.builder.ins().stack_store(len_as_usize, shape_slot, 0);

                let data_ptr  = self.builder.ins().stack_addr(self.codegen.ptr_type(), data_slot, 0);
                let shape_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), shape_slot, 0);
                let dtype_val = self.builder.ins().iconst(I32, dtype_tag(dtype) as i64);
                let ndims_val = self.builder.ins().iconst(self.codegen.ptr_type(), 1);

                let alloc_ref = self.import_func(self.codegen.rt_tensor_alloc_gpu);
                let call = self.builder.ins().call(alloc_ref, &[dtype_val, shape_ptr, ndims_val, data_ptr]);
                let results = self.builder.inst_results(call).to_vec();
                Ok(results[0])
            }

            TypedExprKind::Index { .. } => {
                Err(CodegenError::UnsupportedExpr("Index not yet supported".into()))
            }

            TypedExprKind::FieldAccess { base, field } => {
                if field == "len" && base.ty.is_tensor() {
                    let handle = self.lower_expr(base)?;
                    let len_ref = self.import_func(self.codegen.rt_tensor_len);
                    let call = self.builder.ins().call(len_ref, &[handle]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else {
                    Err(CodegenError::UnsupportedExpr(format!("FieldAccess .{field} not yet supported")))
                }
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
            BinOp::Matmul => Err(CodegenError::UnsupportedExpr("Matmul requires tensor operands".into())),
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

    fn call_print_cstr(&mut self, ptr: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_print_cstr);
        self.builder.ins().call(func_ref, &[ptr]);
    }

    fn emit_print_value(&mut self, arg: &malus_sema::TypedExpr) -> Result<(), CodegenError> {
        let val = self.lower_expr(arg)?;
        match &arg.ty {
            ResolvedTy::Tensor { .. } => self.call_runtime_print(val),
            ResolvedTy::Scalar(s) if is_float_scalar(s) => {
                let func_ref = self.import_func(self.codegen.rt_print_f32);
                self.builder.ins().call(func_ref, &[val]);
            }
            ResolvedTy::Scalar(s) => {
                let wide = match scalar_cranelift_type(s) {
                    I64 => val,
                    _ => self.builder.ins().sextend(I64, val),
                };
                let func_ref = self.import_func(self.codegen.rt_print_i64);
                self.builder.ins().call(func_ref, &[wide]);
            }
            ResolvedTy::Bool => {
                let func_ref = self.import_func(self.codegen.rt_print_bool);
                self.builder.ins().call(func_ref, &[val]);
            }
            _ => return Err(CodegenError::UnsupportedExpr(
                format!("print of {} not supported", arg.ty)
            )),
        }
        Ok(())
    }

    /// Allocate a single-element GPU tensor holding a scalar value.
    /// Used for scalar-broadcast dispatch. The caller is responsible for
    /// ensuring the scalar is f32 (V1 only supports f32 broadcasting).
    /// The returned handle leaks — it will be freed as part of the M11 temp cleanup.
    fn materialize_scalar_tensor(
        &mut self,
        scalar_expr: &malus_sema::TypedExpr,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let scalar_val = self.lower_expr(scalar_expr)?;
        // Widen to f32 if needed.
        let f32_val = match &scalar_expr.ty {
            ResolvedTy::Scalar(s) if is_float_scalar(s) => scalar_val,
            ResolvedTy::Scalar(s) => {
                let t = scalar_cranelift_type(s);
                let wide = if t == I64 { scalar_val } else { self.builder.ins().sextend(I64, scalar_val) };
                self.builder.ins().fcvt_from_sint(F32, wide)
            }
            _ => return Err(CodegenError::UnsupportedExpr("non-scalar in materialize_scalar_tensor".into())),
        };
        // Stack-allocate a 1-element f32 data buffer.
        let data_slot = self.builder.create_sized_stack_slot(
            cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                4,
                2,
            )
        );
        self.builder.ins().stack_store(f32_val, data_slot, 0);

        // Shape slot: [1] as usize.
        let shape_slot = self.builder.create_sized_stack_slot(
            cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                8,
                3,
            )
        );
        let one_usize = self.builder.ins().iconst(I64, 1);
        self.builder.ins().stack_store(one_usize, shape_slot, 0);

        let data_ptr  = self.builder.ins().stack_addr(self.codegen.ptr_type(), data_slot, 0);
        let shape_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), shape_slot, 0);
        let dtype_val = self.builder.ins().iconst(I32, 0); // F32 = tag 0
        let ndims_val = self.builder.ins().iconst(self.codegen.ptr_type(), 1);
        let alloc_ref = self.import_func(self.codegen.rt_tensor_alloc_gpu);
        let call = self.builder.ins().call(alloc_ref, &[dtype_val, shape_ptr, ndims_val, data_ptr]);
        Ok(self.builder.inst_results(call).to_vec()[0])
    }

    fn lower_zeros_ones(
        &mut self,
        name: &str,
        args: &[TypedExpr],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let n = args.len() as u32;
        // Shape slot: n usize elements (8 bytes each), 8-byte aligned.
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
        let shape_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
        let ndims_val = self.builder.ins().iconst(self.codegen.ptr_type(), n as i64);
        let func_ref = if name == "zeros" {
            self.import_func(self.codegen.rt_tensor_alloc_zeros_gpu)
        } else {
            self.import_func(self.codegen.rt_tensor_alloc_ones_gpu)
        };
        let call = self.builder.ins().call(func_ref, &[shape_ptr, ndims_val]);
        Ok(self.builder.inst_results(call).to_vec()[0])
    }

    fn lower_eager_cpu_op(
        &mut self,
        name: &str,
        args: &[TypedExpr],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        if args.len() != 1 {
            return Err(CodegenError::UnsupportedExpr(
                format!("{name} expects exactly one tensor argument"),
            ));
        }
        let handle = self.lower_expr(&args[0])?;
        let func_ref = if name == "transpose" {
            self.import_func(self.codegen.rt_tensor_transpose)
        } else {
            self.import_func(self.codegen.rt_tensor_sum)
        };
        let call = self.builder.ins().call(func_ref, &[handle]);
        Ok(self.builder.inst_results(call).to_vec()[0])
    }

    /// Dispatch a kernel using pre-computed Cranelift Value handles.
    /// Used when an argument was materialized inline (e.g. scalar-broadcast temp).
    fn lower_kernel_dispatch_with_handles(
        &mut self,
        callee: &str,
        handles: &[cranelift_codegen::ir::Value],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let kernel_id = *self.codegen.kernel_ids.get(callee)
            .ok_or_else(|| CodegenError::UnknownKernel { name: callee.to_string() })?;
        let n = handles.len() as u32;
        let slot = self.builder.create_sized_stack_slot(
            cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                n * 8,
                3,
            )
        );
        for (i, &val) in handles.iter().enumerate() {
            self.builder.ins().stack_store(val, slot, (i as i32) * 8);
        }
        let id_val = self.builder.ins().iconst(I64, kernel_id as i64);
        let handles_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
        let n_val = self.builder.ins().iconst(self.codegen.ptr_type(), n as i64);
        let dispatch_ref = self.import_func(self.codegen.rt_kernel_dispatch);
        let call = self.builder.ins().call(dispatch_ref, &[id_val, handles_ptr, n_val]);
        Ok(self.builder.inst_results(call).to_vec()[0])
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

pub fn compile_and_run(
    program: &TypedProgram,
    symbols: &RuntimeSymbols,
    kernel_ids: &HashMap<String, u64>,
) -> Result<(), CodegenError> {
    if !program.fns.iter().any(|f| f.name == "main") {
        return Err(CodegenError::NoMainFunction);
    }

    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let isa = cranelift_native::builder()
        .map_err(|e| CodegenError::JitError(e.to_string()))?
        .finish(flags)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    jit_builder.symbol("tensor_alloc_gpu",       symbols.tensor_alloc_gpu       as *const u8);
    jit_builder.symbol("tensor_print",           symbols.tensor_print           as *const u8);
    jit_builder.symbol("tensor_free",            symbols.tensor_free            as *const u8);
    jit_builder.symbol("kernel_dispatch",        symbols.kernel_dispatch        as *const u8);
    jit_builder.symbol("gpu_barrier",            symbols.gpu_barrier            as *const u8);
    jit_builder.symbol("tensor_alloc_zeros_gpu", symbols.tensor_alloc_zeros_gpu as *const u8);
    jit_builder.symbol("tensor_alloc_ones_gpu",  symbols.tensor_alloc_ones_gpu  as *const u8);
    jit_builder.symbol("tensor_matmul",          symbols.tensor_matmul          as *const u8);
    jit_builder.symbol("tensor_transpose",       symbols.tensor_transpose       as *const u8);
    jit_builder.symbol("tensor_sum",             symbols.tensor_sum             as *const u8);
    jit_builder.symbol("tensor_len",             symbols.tensor_len             as *const u8);
    jit_builder.symbol("print_cstr",             print_cstr                     as *const u8);
    jit_builder.symbol("print_f32",              print_f32                      as *const u8);
    jit_builder.symbol("print_i64",              print_i64                      as *const u8);
    jit_builder.symbol("print_bool",             print_bool                     as *const u8);

    let mut module = JITModule::new(jit_builder);
    let ptr = module.target_config().pointer_type();
    let call_conv = module.target_config().default_call_conv;

    // Declare runtime extern signatures.
    // tensor_alloc_gpu(dtype: i32, shape_ptr: *const usize, ndims: usize, data: *const f32) -> i64
    let sig_alloc = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I32));  // dtype_tag
        s.params.push(AbiParam::new(ptr));  // shape_ptr
        s.params.push(AbiParam::new(ptr));  // ndims
        s.params.push(AbiParam::new(ptr));  // data
        s.returns.push(AbiParam::new(I64));
        s
    };
    // tensor_alloc_{zeros,ones}_gpu(shape_ptr: *const usize, ndims: usize) -> i64
    let sig_alloc_shape = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(ptr));  // shape_ptr
        s.params.push(AbiParam::new(ptr));  // ndims
        s.returns.push(AbiParam::new(I64));
        s
    };
    // tensor_{matmul}(a: i64, b: i64) -> i64
    let sig_binop_tensor = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // tensor_{transpose,sum,len,print,free}(h: i64) -> i64
    let sig_unary_tensor_ret = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
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
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(ptr));
        s.params.push(AbiParam::new(ptr));
        s.returns.push(AbiParam::new(I64));
        s
    };
    let sig_barrier = Signature::new(call_conv);
    let sig_print_cstr = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(ptr));
        s
    };
    let sig_print_f32 = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(F32));
        s
    };
    let sig_print_i64 = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s
    };
    let sig_print_bool = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I8));
        s
    };

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
    let rt_tensor_alloc_zeros_gpu = module.declare_function("tensor_alloc_zeros_gpu", Linkage::Import, &sig_alloc_shape)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_alloc_ones_gpu = module.declare_function("tensor_alloc_ones_gpu", Linkage::Import, &sig_alloc_shape)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_matmul = module.declare_function("tensor_matmul", Linkage::Import, &sig_binop_tensor)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_transpose = module.declare_function("tensor_transpose", Linkage::Import, &sig_unary_tensor_ret)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_sum = module.declare_function("tensor_sum", Linkage::Import, &sig_unary_tensor_ret)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_len = module.declare_function("tensor_len", Linkage::Import, &sig_unary_tensor_ret)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_print_cstr = module.declare_function("print_cstr", Linkage::Import, &sig_print_cstr)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_print_f32 = module.declare_function("print_f32", Linkage::Import, &sig_print_f32)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_print_i64 = module.declare_function("print_i64", Linkage::Import, &sig_print_i64)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_print_bool = module.declare_function("print_bool", Linkage::Import, &sig_print_bool)
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
        kernel_ids,
        rt_tensor_alloc_gpu,
        rt_tensor_print,
        rt_tensor_free,
        rt_kernel_dispatch,
        rt_gpu_barrier,
        rt_tensor_alloc_zeros_gpu,
        rt_tensor_alloc_ones_gpu,
        rt_tensor_matmul,
        rt_tensor_transpose,
        rt_tensor_sum,
        rt_tensor_len,
        rt_print_cstr,
        rt_print_f32,
        rt_print_i64,
        rt_print_bool,
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
