// Cranelift JIT backend for `fn` bodies (CPU host functions).
// Lowers malus's typed IR to Cranelift IR and JIT-compiles to native code.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use cranelift_codegen::ir::{AbiParam, Block, Function, InstBuilder, Signature, UserFuncName};
use cranelift_codegen::ir::types::{F32, I8, I16, I32, I64};
use cranelift_codegen::settings::{self, Configurable};

use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};

use malus_sema::{ResolvedTy, TypedExpr, TypedExprKind, TypedFn, TypedMatchArm, TypedProgram, TypedStmt};
use malus_syntax::ast::{elementwise_builtin_name, scalar_broadcast_builtin_name, BinOp, Lit, ScalarTy, UnaryOp};

// ── libc malloc/free shims (M10 heap allocation for structs/enums) ────────────
//
// These are plain C ABI wrappers so the JIT can call the process's libc
// without depending on malus-runtime (preserving ADR-0008's spirit).

extern "C" fn libc_malloc(size: usize) -> *mut u8 {
    unsafe { libc_alloc(size) }
}

extern "C" fn libc_free(ptr: *mut u8) {
    unsafe { libc_dealloc(ptr) }
}

#[cfg(target_os = "macos")]
extern "C" {
    #[link_name = "malloc"]
    fn libc_alloc(size: usize) -> *mut u8;
    #[link_name = "free"]
    fn libc_dealloc(ptr: *mut u8);
}

#[cfg(not(target_os = "macos"))]
extern "C" {
    #[link_name = "malloc"]
    fn libc_alloc(size: usize) -> *mut u8;
    #[link_name = "free"]
    fn libc_dealloc(ptr: *mut u8);
}

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
    // M9 RC ABI — wired now, unused by M9 CTMM, consumed by M10.
    pub tensor_retain:          extern "C" fn(i64),
    pub tensor_release:         extern "C" fn(i64),
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
        // Structs, enums, and arrays are heap-allocated; represented as opaque i64 pointer.
        ResolvedTy::Struct { .. } | ResolvedTy::Enum { .. } | ResolvedTy::Array { .. } => Ok(Some(I64)),
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
    // M9 RC ABI — wired but not called by M9 generated code.
    rt_tensor_retain: FuncId,
    rt_tensor_release: FuncId,
    // M10: heap allocation for structs and enums.
    rt_malloc: FuncId,
    rt_heap_free: FuncId,
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
        // Do NOT seal the entry block eagerly — loop headers have back-edges
        // from the loop body block that are emitted later, so we must call
        // `seal_all_blocks()` once all control-flow edges are in place.

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
            loop_stack: Vec::new(),
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

        // Seal all blocks now that every control-flow edge (including loop
        // back-edges) has been emitted.
        translator.builder.seal_all_blocks();
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
    /// Stack of (continue_blk, break_blk) for the innermost enclosing loops.
    /// Pushed when entering a For/While/ForIn body, popped on exit.
    loop_stack: Vec<(Block, Block)>,
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

            // ── Control flow (M9) ─────────────────────────────────────────────────
            //
            // Early return from inside a branch/loop body is out of scope for V1;
            // branch bodies always fall through to the merge block.
            // `brif` takes any integer and branches on nonzero — the I8 result of
            // a comparison (`bmask`, 0 = false, -1 = true) works correctly.

            TypedStmt::If { condition, then_body, else_body } => {
                let cond_val = self.lower_expr(condition)?;

                let then_blk = self.builder.create_block();
                let merge_blk = self.builder.create_block();

                if let Some(eb) = else_body {
                    let else_blk = self.builder.create_block();
                    self.builder.ins().brif(cond_val, then_blk, &[], else_blk, &[]);

                    // then branch
                    self.builder.switch_to_block(then_blk);
                    let mut then_term = false;
                    for s in then_body {
                        if then_term { break; }
                        then_term = self.lower_stmt(s)?;
                    }
                    if !then_term { self.builder.ins().jump(merge_blk, &[]); }

                    // else branch
                    self.builder.switch_to_block(else_blk);
                    let mut else_term = false;
                    for s in eb {
                        if else_term { break; }
                        else_term = self.lower_stmt(s)?;
                    }
                    if !else_term { self.builder.ins().jump(merge_blk, &[]); }

                    if then_term && else_term {
                        // merge_blk has no predecessors — don't switch into it.
                        return Ok(true);
                    }
                } else {
                    self.builder.ins().brif(cond_val, then_blk, &[], merge_blk, &[]);

                    // then branch (no-else: merge_blk already a predecessor via brif)
                    self.builder.switch_to_block(then_blk);
                    let mut then_term = false;
                    for s in then_body {
                        if then_term { break; }
                        then_term = self.lower_stmt(s)?;
                    }
                    if !then_term { self.builder.ins().jump(merge_blk, &[]); }
                }

                self.builder.switch_to_block(merge_blk);
                Ok(false)
            }

            TypedStmt::For { var, start, end, body } => {
                // Declare the loop variable as a Cranelift SSA Variable (I64).
                let loop_var = Variable::from_u32(self.next_var as u32);
                self.next_var += 1;
                self.builder.declare_var(loop_var, I64);
                let start_val = self.lower_expr(start)?;
                self.builder.def_var(loop_var, start_val);
                self.var_map.insert(var.clone(), loop_var);

                let header_blk = self.builder.create_block();
                let body_blk   = self.builder.create_block();
                // Separate increment block so `continue` increments before re-testing.
                let incr_blk   = self.builder.create_block();
                let exit_blk   = self.builder.create_block();

                // Jump from the pre-header into the loop header.
                self.builder.ins().jump(header_blk, &[]);

                // Loop header: test loop_var < end.
                self.builder.switch_to_block(header_blk);
                let cur = self.builder.use_var(loop_var);
                let end_val = self.lower_expr(end)?;
                let cmp = self.builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, cur, end_val);
                // `icmp` returns I8; brif branches on nonzero.
                self.builder.ins().brif(cmp, body_blk, &[], exit_blk, &[]);

                // Loop body: `continue` → incr_blk, `break` → exit_blk.
                self.builder.switch_to_block(body_blk);
                self.loop_stack.push((incr_blk, exit_blk));
                let mut body_terminated = false;
                for s in body {
                    if body_terminated { break; }
                    body_terminated = self.lower_stmt(s)?;
                }
                self.loop_stack.pop();
                if !body_terminated {
                    self.builder.ins().jump(incr_blk, &[]);
                }

                // Increment block: advance loop variable, jump back to header.
                self.builder.switch_to_block(incr_blk);
                let cur2 = self.builder.use_var(loop_var);
                let one  = self.builder.ins().iconst(I64, 1);
                let next = self.builder.ins().iadd(cur2, one);
                self.builder.def_var(loop_var, next);
                self.builder.ins().jump(header_blk, &[]);

                self.builder.switch_to_block(exit_blk);
                Ok(false)
            }

            TypedStmt::While { condition, body } => {
                let header_blk = self.builder.create_block();
                let body_blk   = self.builder.create_block();
                let exit_blk   = self.builder.create_block();

                self.builder.ins().jump(header_blk, &[]);

                // Loop header: evaluate condition.
                self.builder.switch_to_block(header_blk);
                let cond_val = self.lower_expr(condition)?;
                self.builder.ins().brif(cond_val, body_blk, &[], exit_blk, &[]);

                // Loop body: `continue` → header_blk, `break` → exit_blk.
                self.builder.switch_to_block(body_blk);
                self.loop_stack.push((header_blk, exit_blk));
                let mut body_terminated = false;
                for s in body {
                    if body_terminated { break; }
                    body_terminated = self.lower_stmt(s)?;
                }
                self.loop_stack.pop();
                if !body_terminated {
                    self.builder.ins().jump(header_blk, &[]);
                }

                self.builder.switch_to_block(exit_blk);
                Ok(false)
            }

            // ── M11: for-in loop over fixed arrays ────────────────────────────────
            TypedStmt::ForIn { var, iter, body } => {
                // Load the array pointer and determine the element type + length.
                let arr_ptr = self.lower_expr(iter)?;
                let (elem_ty, len) = match &iter.ty {
                    malus_sema::ResolvedTy::Array { elem, len } => (*elem.clone(), *len),
                    _ => return Err(CodegenError::UnsupportedExpr("ForIn requires Array type".into())),
                };
                let cl_ty = match cranelift_type(&elem_ty)? {
                    Some(t) => t,
                    None => return Err(CodegenError::UnsupportedExpr("ForIn: unit-typed array element".into())),
                };

                // Declare the loop variable (the element binding).
                let elem_var = Variable::from_u32(self.next_var as u32);
                self.next_var += 1;
                self.builder.declare_var(elem_var, cl_ty);
                self.var_map.insert(var.clone(), elem_var);

                // Declare the index variable (i64, not visible to the body).
                let idx_var = Variable::from_u32(self.next_var as u32);
                self.next_var += 1;
                self.builder.declare_var(idx_var, I64);
                let zero = self.builder.ins().iconst(I64, 0);
                self.builder.def_var(idx_var, zero);

                let header_blk = self.builder.create_block();
                let body_blk   = self.builder.create_block();
                // Separate increment block so `continue` advances the index first.
                let incr_blk   = self.builder.create_block();
                let exit_blk   = self.builder.create_block();

                self.builder.ins().jump(header_blk, &[]);

                // Header: test i < len.
                self.builder.switch_to_block(header_blk);
                let cur_idx = self.builder.use_var(idx_var);
                let len_val = self.builder.ins().iconst(I64, len as i64);
                let cmp = self.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, cur_idx, len_val,
                );
                self.builder.ins().brif(cmp, body_blk, &[], exit_blk, &[]);

                // Body: load element at arr_ptr + idx * 8.
                self.builder.switch_to_block(body_blk);
                let body_idx = self.builder.use_var(idx_var);
                let eight = self.builder.ins().iconst(I64, 8);
                let byte_offset_val = self.builder.ins().imul(body_idx, eight);
                let elem_ptr = self.builder.ins().iadd(arr_ptr, byte_offset_val);
                let elem_val = self.builder.ins().load(cl_ty, cranelift_codegen::ir::MemFlags::trusted(), elem_ptr, 0);
                self.builder.def_var(elem_var, elem_val);

                let body = body.clone();
                self.loop_stack.push((incr_blk, exit_blk));
                let mut body_terminated = false;
                for s in &body {
                    if body_terminated { break; }
                    body_terminated = self.lower_stmt(s)?;
                }
                self.loop_stack.pop();
                if !body_terminated {
                    self.builder.ins().jump(incr_blk, &[]);
                }

                // Increment block: advance index, jump back to header.
                self.builder.switch_to_block(incr_blk);
                let cur_idx2 = self.builder.use_var(idx_var);
                let one = self.builder.ins().iconst(I64, 1);
                let next_idx = self.builder.ins().iadd(cur_idx2, one);
                self.builder.def_var(idx_var, next_idx);
                self.builder.ins().jump(header_blk, &[]);

                self.builder.switch_to_block(exit_blk);
                Ok(false)
            }

            // ── M12: break / continue ────────────────────────────────────────────
            TypedStmt::Break => {
                let (_, break_blk) = *self.loop_stack.last()
                    .ok_or_else(|| CodegenError::UnsupportedExpr("break outside loop".into()))?;
                self.builder.ins().jump(break_blk, &[]);
                Ok(true) // block terminated
            }

            TypedStmt::Continue => {
                let (continue_blk, _) = *self.loop_stack.last()
                    .ok_or_else(|| CodegenError::UnsupportedExpr("continue outside loop".into()))?;
                self.builder.ins().jump(continue_blk, &[]);
                Ok(true) // block terminated
            }

            // ── M10 RC nodes (wired, not emitted by M9 CTMM) ─────────────────────
            TypedStmt::Retain { name } => {
                let handle = self.use_var(name)?;
                self.call_runtime_retain(handle);
                Ok(false)
            }

            TypedStmt::Release { name } => {
                let handle = self.use_var(name)?;
                self.call_runtime_release(handle);
                Ok(false)
            }

            // ── M10/M11: aggregate types ──────────────────────────────────────────
            TypedStmt::DropStruct { name, droppable_fields } => {
                let ptr = self.use_var(name)?;
                // Recursively release droppable fields, then free the struct box.
                let droppable_fields = droppable_fields.clone();
                for (slot_idx, field_ty) in &droppable_fields {
                    let offset = (*slot_idx as i32) * 8;
                    self.emit_drop_field(ptr, offset, field_ty)?;
                }
                self.call_heap_free(ptr);
                Ok(false)
            }

            TypedStmt::DropEnum { name, variants } => {
                let ptr = self.use_var(name)?;
                let variants = variants.clone();
                self.emit_drop_enum_box(ptr, &variants)?;
                Ok(false)
            }

            TypedStmt::DropArray { name, elem_ty, len } => {
                let ptr = self.use_var(name)?;
                let elem_ty = elem_ty.clone();
                let len = *len;
                // If the element type owns heap resources, release each element.
                if elem_ty.is_tensor() || elem_ty.is_struct() || elem_ty.is_enum() || elem_ty.is_array() {
                    self.emit_counted_drop_loop(ptr, &elem_ty, len)?;
                }
                self.call_heap_free(ptr);
                Ok(false)
            }

            TypedStmt::Match { scrutinee, arms } => {
                self.lower_match(scrutinee, arms)
            }
        }
    }

    fn lower_match(&mut self, scrutinee: &TypedExpr, arms: &[TypedMatchArm]) -> Result<bool, CodegenError> {
        let scrut_ptr = self.lower_expr(scrutinee)?;
        // Load the u32 tag stored at offset 0 (as i32 for Cranelift).
        let tag = self.builder.ins().load(I32, cranelift_codegen::ir::MemFlags::trusted(), scrut_ptr, 0);

        // Create a merge block (only used if at least one arm falls through).
        let merge_blk = self.builder.create_block();
        let mut all_diverge = true;

        // Build arm blocks and per-arm end-of-chain fallback block.
        // Chain: arm0_test → arm0_body | arm1_test → arm1_body | ... | unreachable
        let mut arm_test_blks: Vec<cranelift_codegen::ir::Block> = Vec::new();
        for _ in arms {
            arm_test_blks.push(self.builder.create_block());
        }
        let unreachable_blk = self.builder.create_block();

        // Jump into the first test block.
        if let Some(&first) = arm_test_blks.first() {
            self.builder.ins().jump(first, &[]);
        }

        for (i, arm) in arms.iter().enumerate() {
            let test_blk = arm_test_blks[i];
            let next_blk = arm_test_blks.get(i + 1).copied().unwrap_or(unreachable_blk);
            let body_blk = self.builder.create_block();

            self.builder.switch_to_block(test_blk);
            self.builder.seal_block(test_blk);

            let expected_tag = self.builder.ins().iconst(I32, arm.variant_index as i64);
            let cmp = self.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                tag,
                expected_tag,
            );
            self.builder.ins().brif(cmp, body_blk, &[], next_blk, &[]);

            // Arm body.
            self.builder.switch_to_block(body_blk);
            self.builder.seal_block(body_blk);

            // Bind payload fields from pointer (offset 8 + j*8 per field).
            for (j, (binding_name, binding_ty)) in arm.bindings.iter().enumerate() {
                let offset = 8 + (j as i32) * 8;
                let cl_ty = match cranelift_type(binding_ty)? {
                    Some(t) => t,
                    None => continue,
                };
                // Load using the binding's native Cranelift type (matches store in EnumInit).
                let field_val = self.builder.ins().load(cl_ty, cranelift_codegen::ir::MemFlags::trusted(), scrut_ptr, offset);
                let var = Variable::from_u32(self.next_var as u32);
                self.next_var += 1;
                self.builder.declare_var(var, cl_ty);
                self.builder.def_var(var, field_val);
                self.var_map.insert(binding_name.clone(), var);
            }

            let mut arm_diverges = false;
            for s in &arm.body {
                if self.lower_stmt(s)? {
                    arm_diverges = true;
                    break;
                }
            }
            if arm_diverges {
                // Arm ends in return — don't jump to merge.
            } else {
                self.builder.ins().jump(merge_blk, &[]);
                all_diverge = false;
            }
        }

        // Unreachable block (match is exhaustive at compile time).
        self.builder.switch_to_block(unreachable_blk);
        self.builder.seal_block(unreachable_blk);
        self.builder.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());

        if !all_diverge {
            self.builder.switch_to_block(merge_blk);
            self.builder.seal_block(merge_blk);
        }

        Ok(all_diverge)
    }

    fn use_var(&mut self, name: &str) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let var = self.var_map.get(name)
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr(format!("unknown variable: {name}")))?;
        Ok(self.builder.use_var(var))
    }

    // ── Recursive aggregate drop helpers (M11) ────────────────────────────────

    /// Emit code to release one field at `byte_offset` within `base_ptr`.
    /// - Tensor  → `tensor_release(field_value)`
    /// - Struct  → recursively release struct fields, then `heap_free`
    /// - Enum    → tag-switch, per-arm release, then `heap_free`
    /// All other types carry no heap resources and are skipped.
    fn emit_drop_field(
        &mut self,
        base_ptr: cranelift_codegen::ir::Value,
        byte_offset: i32,
        ty: &malus_sema::ResolvedTy,
    ) -> Result<(), CodegenError> {
        use malus_sema::ResolvedTy;
        match ty {
            ResolvedTy::Tensor { .. } => {
                let handle = self.builder.ins().load(
                    I64, cranelift_codegen::ir::MemFlags::trusted(), base_ptr, byte_offset,
                );
                self.call_runtime_release(handle);
            }
            ResolvedTy::Struct { fields, .. } => {
                let nested_ptr = self.builder.ins().load(
                    I64, cranelift_codegen::ir::MemFlags::trusted(), base_ptr, byte_offset,
                );
                let droppable: Vec<(usize, ResolvedTy)> = fields.iter()
                    .enumerate()
                    .filter_map(|(i, (_, fty))| {
                        if fty.is_tensor() || fty.is_struct() || fty.is_enum() {
                            Some((i, fty.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                for (fidx, fty) in &droppable {
                    self.emit_drop_field(nested_ptr, (*fidx as i32) * 8, fty)?;
                }
                self.call_heap_free(nested_ptr);
            }
            ResolvedTy::Enum { variants, .. } => {
                let nested_ptr = self.builder.ins().load(
                    I64, cranelift_codegen::ir::MemFlags::trusted(), base_ptr, byte_offset,
                );
                let drop_variants: Vec<(u32, Vec<(usize, ResolvedTy)>)> = variants.iter()
                    .enumerate()
                    .map(|(tag, (_, vfields))| {
                        let droppable = vfields.iter()
                            .enumerate()
                            .filter_map(|(i, (_, vty))| {
                                if vty.is_tensor() || vty.is_struct() || vty.is_enum() {
                                    Some((i, vty.clone()))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        (tag as u32, droppable)
                    })
                    .collect();
                self.emit_drop_enum_box(nested_ptr, &drop_variants)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Emit a tag-switch drop for an enum box pointer.
    /// Each variant arm releases its droppable fields; all arms jump to a merge
    /// block that calls `heap_free` exactly once.
    fn emit_drop_enum_box(
        &mut self,
        ptr: cranelift_codegen::ir::Value,
        variants: &[(u32, Vec<(usize, malus_sema::ResolvedTy)>)],
    ) -> Result<(), CodegenError> {
        let tag = self.builder.ins().load(
            I32, cranelift_codegen::ir::MemFlags::trusted(), ptr, 0,
        );

        let merge_blk = self.builder.create_block();
        let arm_test_blks: Vec<_> = variants.iter().map(|_| self.builder.create_block()).collect();
        let unreachable_blk = self.builder.create_block();

        if let Some(&first) = arm_test_blks.first() {
            self.builder.ins().jump(first, &[]);
        } else {
            // No variants — just free the box and return.
            self.call_heap_free(ptr);
            return Ok(());
        }

        let variants_clone: Vec<(u32, Vec<(usize, malus_sema::ResolvedTy)>)> = variants.to_vec();
        for (i, (variant_tag, droppable_fields)) in variants_clone.iter().enumerate() {
            let test_blk = arm_test_blks[i];
            let next_blk = arm_test_blks.get(i + 1).copied().unwrap_or(unreachable_blk);
            let body_blk = self.builder.create_block();

            self.builder.switch_to_block(test_blk);
            self.builder.seal_block(test_blk);

            let expected = self.builder.ins().iconst(I32, *variant_tag as i64);
            let cmp = self.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal, tag, expected,
            );
            self.builder.ins().brif(cmp, body_blk, &[], next_blk, &[]);

            self.builder.switch_to_block(body_blk);
            self.builder.seal_block(body_blk);

            // Release droppable fields (payload starts at offset 8).
            let droppable = droppable_fields.clone();
            for (slot_idx, field_ty) in &droppable {
                let offset = 8 + (*slot_idx as i32) * 8;
                self.emit_drop_field(ptr, offset, field_ty)?;
            }
            self.builder.ins().jump(merge_blk, &[]);
        }

        self.builder.switch_to_block(unreachable_blk);
        self.builder.seal_block(unreachable_blk);
        self.builder.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());

        self.builder.switch_to_block(merge_blk);
        self.builder.seal_block(merge_blk);
        self.call_heap_free(ptr);

        Ok(())
    }

    /// Emit a counted loop that calls `emit_drop_field` for each element of a
    /// fixed array at `arr_ptr`.  Elements are at `arr_ptr + i * 8`.
    /// Only called when `elem_ty` owns heap resources.
    fn emit_counted_drop_loop(
        &mut self,
        arr_ptr: cranelift_codegen::ir::Value,
        elem_ty: &malus_sema::ResolvedTy,
        len: usize,
    ) -> Result<(), CodegenError> {
        let idx_var = Variable::from_u32(self.next_var as u32);
        self.next_var += 1;
        self.builder.declare_var(idx_var, I64);
        let zero = self.builder.ins().iconst(I64, 0);
        self.builder.def_var(idx_var, zero);

        let header_blk = self.builder.create_block();
        let body_blk   = self.builder.create_block();
        let exit_blk   = self.builder.create_block();

        self.builder.ins().jump(header_blk, &[]);

        self.builder.switch_to_block(header_blk);
        let cur = self.builder.use_var(idx_var);
        let len_val = self.builder.ins().iconst(I64, len as i64);
        let cmp = self.builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, cur, len_val,
        );
        self.builder.ins().brif(cmp, body_blk, &[], exit_blk, &[]);

        self.builder.switch_to_block(body_blk);
        self.builder.seal_block(body_blk);
        // Compute byte offset: arr_ptr + i * 8
        let body_cur = self.builder.use_var(idx_var);
        let eight = self.builder.ins().iconst(I64, 8);
        let byte_off = self.builder.ins().imul(body_cur, eight);
        let elem_ptr = self.builder.ins().iadd(arr_ptr, byte_off);
        // Elements are values stored by value — load the handle from the slot.
        let handle = self.builder.ins().load(
            I64, cranelift_codegen::ir::MemFlags::trusted(), elem_ptr, 0,
        );
        // Drop the element at ptr=handle (the slot IS the pointer for agg types).
        let elem_ty_clone = elem_ty.clone();
        match &elem_ty_clone {
            malus_sema::ResolvedTy::Tensor { .. } => {
                self.call_runtime_release(handle);
            }
            malus_sema::ResolvedTy::Struct { fields, .. } => {
                let droppable: Vec<(usize, malus_sema::ResolvedTy)> = fields.iter().enumerate()
                    .filter_map(|(i, (_, ty))| {
                        if ty.is_tensor() || ty.is_struct() || ty.is_enum() || ty.is_array() {
                            Some((i, ty.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                for (slot_idx, field_ty) in &droppable {
                    let offset = (*slot_idx as i32) * 8;
                    self.emit_drop_field(handle, offset, field_ty)?;
                }
                self.call_heap_free(handle);
            }
            malus_sema::ResolvedTy::Enum { variants, .. } => {
                let drop_variants: Vec<(u32, Vec<(usize, malus_sema::ResolvedTy)>)> = variants.iter()
                    .enumerate()
                    .map(|(vi, (_, vfields))| {
                        let droppable: Vec<(usize, malus_sema::ResolvedTy)> = vfields.iter().enumerate()
                            .filter_map(|(fi, (_, fty))| {
                                if fty.is_tensor() || fty.is_struct() || fty.is_enum() || fty.is_array() {
                                    Some((fi, fty.clone()))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        (vi as u32, droppable)
                    })
                    .collect();
                self.emit_drop_enum_box(handle, &drop_variants)?;
            }
            malus_sema::ResolvedTy::Array { elem, len } => {
                let inner_elem = *elem.clone();
                let inner_len = *len;
                if inner_elem.is_tensor() || inner_elem.is_struct() || inner_elem.is_enum() || inner_elem.is_array() {
                    self.emit_counted_drop_loop(handle, &inner_elem, inner_len)?;
                }
                self.call_heap_free(handle);
            }
            _ => {}
        }
        // Advance index.
        let cur2 = self.builder.use_var(idx_var);
        let one = self.builder.ins().iconst(I64, 1);
        let next = self.builder.ins().iadd(cur2, one);
        self.builder.def_var(idx_var, next);
        self.builder.ins().jump(header_blk, &[]);
        self.builder.seal_block(header_blk);

        self.builder.switch_to_block(exit_blk);
        self.builder.seal_block(exit_blk);

        Ok(())
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
                    // tensor ⊕ scalar — materialize scalar as 1-elem tensor, dispatch scalar builtin,
                    // then immediately free the temp (Metal retains the buffer until GPU finishes).
                    (ResolvedTy::Tensor { dtype }, ResolvedTy::Scalar(sd)) if dtype == sd && *dtype == ScalarTy::F32 => {
                        if let Some(name) = scalar_broadcast_builtin_name(op, true) {
                            let tensor_handle = self.lower_expr(lhs)?;
                            let scalar_handle = self.materialize_scalar_tensor(rhs)?;
                            let result = self.lower_kernel_dispatch_with_handles(name, &[tensor_handle, scalar_handle])?;
                            self.call_runtime_free(scalar_handle);
                            Ok(result)
                        } else {
                            Err(CodegenError::UnsupportedExpr(format!("scalar broadcast {:?} not supported", op)))
                        }
                    }
                    // scalar ⊕ tensor — scalar-on-left: buffer layout is [tensor, scalar, out]
                    (ResolvedTy::Scalar(sd), ResolvedTy::Tensor { dtype }) if sd == dtype && *dtype == ScalarTy::F32 => {
                        if let Some(name) = scalar_broadcast_builtin_name(op, false) {
                            let tensor_handle = self.lower_expr(rhs)?;
                            let scalar_handle = self.materialize_scalar_tensor(lhs)?;
                            let result = self.lower_kernel_dispatch_with_handles(name, &[tensor_handle, scalar_handle])?;
                            self.call_runtime_free(scalar_handle);
                            Ok(result)
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

            TypedExprKind::TensorLiteral { dtype, elements, shape, .. } => {
                let len = elements.len() as u32;
                // Stack slot for f32 elements (len * 4 bytes, 4-byte aligned).
                let data_slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        (len * 4).max(4),
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

                // Shape slot: ndims usizes (each 8 bytes, 8-byte aligned).
                // For 1-D: [N].  For 2-D: [rows, cols].
                let ndims = shape.len();
                let shape_slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        (ndims as u32 * 8).max(8),
                        3,
                    )
                );
                for (i, &dim) in shape.iter().enumerate() {
                    let dim_val = self.builder.ins().iconst(I64, dim as i64);
                    self.builder.ins().stack_store(dim_val, shape_slot, (i as i32) * 8);
                }

                let data_ptr  = self.builder.ins().stack_addr(self.codegen.ptr_type(), data_slot, 0);
                let shape_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), shape_slot, 0);
                let dtype_val = self.builder.ins().iconst(I32, dtype_tag(dtype) as i64);
                let ndims_val = self.builder.ins().iconst(self.codegen.ptr_type(), ndims as i64);

                let alloc_ref = self.import_func(self.codegen.rt_tensor_alloc_gpu);
                let call = self.builder.ins().call(alloc_ref, &[dtype_val, shape_ptr, ndims_val, data_ptr]);
                let results = self.builder.inst_results(call).to_vec();
                Ok(results[0])
            }

            TypedExprKind::Index { base, indices } => {
                match &base.ty {
                    ResolvedTy::Array { .. } => {
                        let arr_ptr = self.lower_expr(base)?;
                        let idx_val = self.lower_expr(&indices[0])?;
                        let eight = self.builder.ins().iconst(I64, 8);
                        let byte_off = self.builder.ins().imul(idx_val, eight);
                        let elem_ptr = self.builder.ins().iadd(arr_ptr, byte_off);
                        let cl_ty = match cranelift_type(&expr.ty)? {
                            Some(t) => t,
                            None => return Err(CodegenError::UnsupportedExpr("Index: unit element type".into())),
                        };
                        let val = self.builder.ins().load(cl_ty, cranelift_codegen::ir::MemFlags::trusted(), elem_ptr, 0);
                        Ok(val)
                    }
                    _ => Err(CodegenError::UnsupportedExpr("Index: only Array indexing is supported".into())),
                }
            }

            TypedExprKind::ArrayLiteral { elements } => {
                let n = elements.len();
                // Heap-allocate N * 8 bytes (uniform 8-byte slots like structs).
                let size = (n * 8) as i64;
                let size_val = self.builder.ins().iconst(self.codegen.ptr_type(), size);
                let arr_ptr = self.call_malloc(size_val);
                for (i, elem_expr) in elements.iter().enumerate() {
                    let val = self.lower_expr(elem_expr)?;
                    self.builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        val,
                        arr_ptr,
                        (i as i32) * 8,
                    );
                }
                Ok(arr_ptr)
            }

            TypedExprKind::FieldAccess { base, field } => {
                if field == "len" && base.ty.is_tensor() {
                    let handle = self.lower_expr(base)?;
                    let len_ref = self.import_func(self.codegen.rt_tensor_len);
                    let call = self.builder.ins().call(len_ref, &[handle]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if let ResolvedTy::Struct { fields, .. } = &base.ty {
                    let slot_idx = fields.iter().position(|(n, _)| n == field)
                        .ok_or_else(|| CodegenError::UnsupportedExpr(format!("struct has no field '{field}'")))?;
                    let ptr = self.lower_expr(base)?;
                    let offset = (slot_idx as i32) * 8;
                    let cl_ty = match cranelift_type(&expr.ty)? {
                        Some(t) => t,
                        None => return Err(CodegenError::UnsupportedExpr("unit field access".into())),
                    };
                    // Load using the field's native Cranelift type (matches store).
                    let val = self.builder.ins().load(cl_ty, cranelift_codegen::ir::MemFlags::trusted(), ptr, offset);
                    Ok(val)
                } else {
                    Err(CodegenError::UnsupportedExpr(format!("FieldAccess .{field} not yet supported")))
                }
            }

            TypedExprKind::StructInit { fields, name } => {
                let n_slots = fields.len();
                let size = (n_slots * 8) as i64;
                let size_val = self.builder.ins().iconst(self.codegen.ptr_type(), size);
                let ptr = self.call_malloc(size_val);
                for (i, field_expr) in fields.iter().enumerate() {
                    let val = self.lower_expr(field_expr)?;
                    // Store each field in its native Cranelift type at slot i*8.
                    // Each slot is 8 bytes wide; smaller types leave the upper bytes unused.
                    self.builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        val,
                        ptr,
                        (i as i32) * 8,
                    );
                }
                let _ = name;
                Ok(ptr)
            }

            TypedExprKind::EnumInit { variant_index, payload, max_payload_slots, .. } => {
                // Layout: 8-byte tag slot (i32 tag + 4 bytes padding) + max_payload_slots * 8 bytes.
                let total_slots = 1 + max_payload_slots;
                let size = (total_slots * 8) as i64;
                let size_val = self.builder.ins().iconst(self.codegen.ptr_type(), size);
                let ptr = self.call_malloc(size_val);
                // Store u32 tag at offset 0 (as i32).
                let tag_val = self.builder.ins().iconst(I32, *variant_index as i64);
                self.builder.ins().store(
                    cranelift_codegen::ir::MemFlags::trusted(),
                    tag_val,
                    ptr,
                    0,
                );
                // Store payload fields at offsets 8 + j*8, each in its native type.
                for (j, field_expr) in payload.iter().enumerate() {
                    let val = self.lower_expr(field_expr)?;
                    self.builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        val,
                        ptr,
                        8 + (j as i32) * 8,
                    );
                }
                Ok(ptr)
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
    /// ensuring the scalar is f32 (V1 only supports f32 broadcasting)
    /// and for freeing the returned handle after dispatch.
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

    fn call_runtime_retain(&mut self, handle: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_tensor_retain);
        self.builder.ins().call(func_ref, &[handle]);
    }

    fn call_runtime_release(&mut self, handle: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_tensor_release);
        self.builder.ins().call(func_ref, &[handle]);
    }

    fn call_runtime_barrier(&mut self) {
        let func_ref = self.import_func(self.codegen.rt_gpu_barrier);
        self.builder.ins().call(func_ref, &[]);
    }

    fn call_malloc(&mut self, size: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        let func_ref = self.import_func(self.codegen.rt_malloc);
        let call = self.builder.ins().call(func_ref, &[size]);
        self.builder.inst_results(call).to_vec()[0]
    }

    fn call_heap_free(&mut self, ptr: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_heap_free);
        self.builder.ins().call(func_ref, &[ptr]);
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
    jit_builder.symbol("tensor_retain",          symbols.tensor_retain          as *const u8);
    jit_builder.symbol("tensor_release",         symbols.tensor_release         as *const u8);
    jit_builder.symbol("print_cstr",             print_cstr                     as *const u8);
    jit_builder.symbol("print_f32",              print_f32                      as *const u8);
    jit_builder.symbol("print_i64",              print_i64                      as *const u8);
    jit_builder.symbol("print_bool",             print_bool                     as *const u8);
    // libc malloc/free — process symbols, available without malus-runtime.
    jit_builder.symbol("malloc",                 libc_malloc                    as *const u8);
    jit_builder.symbol("free",                   libc_free                      as *const u8);

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
    // tensor_retain/tensor_release have the same signature as tensor_free: (i64) -> ()
    let rt_tensor_retain = module.declare_function("tensor_retain", Linkage::Import, &sig_free)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_release = module.declare_function("tensor_release", Linkage::Import, &sig_free)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // malloc(size: usize) -> *mut u8   (i64 on 64-bit)
    let sig_malloc = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(ptr));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // free(ptr: *mut u8)
    let sig_heap_free = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s
    };
    let rt_malloc = module.declare_function("malloc", Linkage::Import, &sig_malloc)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_heap_free = module.declare_function("free", Linkage::Import, &sig_heap_free)
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
        rt_tensor_retain,
        rt_tensor_release,
        rt_malloc,
        rt_heap_free,
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
