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
    #[link_name = "calloc"]
    fn libc_calloc(nmemb: usize, size: usize) -> *mut u8;
    #[link_name = "free"]
    fn libc_dealloc(ptr: *mut u8);
}

#[cfg(not(target_os = "macos"))]
extern "C" {
    #[link_name = "malloc"]
    fn libc_alloc(size: usize) -> *mut u8;
    #[link_name = "calloc"]
    fn libc_calloc(nmemb: usize, size: usize) -> *mut u8;
    #[link_name = "free"]
    fn libc_dealloc(ptr: *mut u8);
}

// ── Aggregate ARC shims (M13 heap allocation for struct/enum boxes) ───────────
//
// All struct/enum boxes have an 8-byte ARC header (AtomicUsize refcount) at
// offset 0.  Field/tag storage begins at byte 8 and beyond.
// These functions are local to codegen-cpu so malus-runtime stays unaware
// (ADR-0008).

extern "C" fn aggregate_alloc(size: i64) -> i64 {
    let ptr = unsafe { libc_calloc(1, size as usize) };
    if ptr.is_null() { panic!("aggregate_alloc: out of memory"); }
    unsafe {
        (*(ptr as *mut std::sync::atomic::AtomicUsize))
            .store(1, std::sync::atomic::Ordering::Relaxed);
    }
    ptr as i64
}

extern "C" fn aggregate_retain(ptr: i64) {
    if ptr == 0 { return; }
    unsafe {
        (*(ptr as *mut std::sync::atomic::AtomicUsize))
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

extern "C" fn aggregate_release(ptr: i64) {
    if ptr == 0 { return; }
    let prev = unsafe {
        (*(ptr as *mut std::sync::atomic::AtomicUsize))
            .fetch_sub(1, std::sync::atomic::Ordering::Release)
    };
    if prev == 1 {
        unsafe { libc_dealloc(ptr as *mut u8); }
    }
}

// M28: `List<T>` release, returning the PREVIOUS refcount (unlike
// `aggregate_release`'s `()`) so codegen can conditionally release element
// tensors only when this was genuinely the last reference (ADR-0034). Every
// existing `aggregate_release` call site is reached only where CTMM has
// already statically proven single ownership (struct/tuple/enum fields are
// unconditionally released right before it, since refcount is always exactly
// 1 there) — `List` cannot make that same static guarantee (it may alias
// across a call boundary, e.g. `Module::parameters` returning a model's own
// field), so its codegen must check the *result* of the decrement instead of
// assuming it. Frees the box itself here (same as `aggregate_release`); the
// caller must read any data it needs (length, element handles) BEFORE calling
// this, since the box may no longer exist once it returns.
extern "C" fn list_release(ptr: i64) -> i64 {
    if ptr == 0 { return 0; }
    let prev = unsafe {
        (*(ptr as *mut std::sync::atomic::AtomicUsize))
            .fetch_sub(1, std::sync::atomic::Ordering::Release)
    };
    if prev == 1 {
        unsafe { libc_dealloc(ptr as *mut u8); }
    }
    prev as i64
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
    pub tensor_len:             extern "C" fn(i64) -> i64,
    // M9 RC ABI — wired now, unused by M9 CTMM, consumed by M10.
    pub tensor_retain:          extern "C" fn(i64),
    pub tensor_release:         extern "C" fn(i64),
    // M14 tape ABI.
    pub tape_record_binop:      extern "C" fn(i32, i64, i64, i64),
    pub tape_record_unary:      extern "C" fn(i32, i64, i64),
    pub tape_register_leaf:     extern "C" fn(i64),
    pub tape_pause:             extern "C" fn(),
    pub tape_resume:            extern "C" fn(),
    pub tape_clear:             extern "C" fn(),
    pub tape_get_grad:          extern "C" fn(i64) -> i64,
    pub backward:               extern "C" fn(i64),
    // M15 tape ABI.
    pub tape_zero_grad:         extern "C" fn(*const i64, usize),
    pub tape_record_reduce:        extern "C" fn(i32, i64, i64, i64, i64),
    // M17 shapes + batched matmul.
    pub tensor_reshape:            extern "C" fn(i64, *const usize, usize) -> i64,
    pub tensor_permute:            extern "C" fn(i64, *const usize, usize) -> i64,
    pub tape_record_perm:          extern "C" fn(i32, i64, i64, *const usize, usize),
    // M18 transformer stdlib.
    pub tensor_causal_mask:        extern "C" fn(i64) -> i64,
    pub tape_record_layernorm:     extern "C" fn(i32, i64, i64, i64, i64),
    pub tape_record_cross_entropy: extern "C" fn(i32, i64, i64, i64, i64),
    // M19 randn.
    pub tensor_randn:              extern "C" fn(*const usize, usize) -> i64,
    pub tape_record_embedding:     extern "C" fn(i32, i64, i64, i64),
    // M22 string I/O.
    pub malus_str_box:             extern "C" fn(*const u8, usize) -> i64,
    pub malus_read_file:           extern "C" fn(i64) -> i64,
    pub malus_str_len:             extern "C" fn(i64) -> i64,
    pub malus_str_char_at:         extern "C" fn(i64, i64) -> i64,
    pub malus_str_from_char:       extern "C" fn(i64) -> i64,
    // M22 rand_uniform — no args, returns f32.
    pub malus_rand_uniform:        extern "C" fn() -> f32,
    // M22 Buffer<i32>.
    pub malus_buffer_i32:          extern "C" fn(i64) -> i64,
    pub malus_buffer_get_i32:      extern "C" fn(i64, i64) -> i64,
    pub malus_buffer_set_i32:      extern "C" fn(i64, i64, i64),
    pub malus_buffer_free:         extern "C" fn(i64),
    pub malus_buffer_freeze_i32:   extern "C" fn(i64) -> i64,
    // M22 rand_int + tensor_get_f32.
    pub malus_rand_int:            extern "C" fn(i64) -> i64,
    pub malus_tensor_get_f32:      extern "C" fn(i64, i64) -> f32,
    // M25 metadata accessors (no cpu_compute_inc).
    pub tensor_ndim:               extern "C" fn(i64) -> i64,
    pub tensor_dim:                extern "C" fn(i64, i64) -> i64,
    // M25 extended dispatch ABI (kernel_dispatch_v2).
    pub kernel_dispatch_v2:        extern "C" fn(u64, *const i64, usize, *const usize, *const usize, *const usize, usize, i32, *const std::ffi::c_void, usize) -> i64,
    // M26 — registers a backward kernel's finalized JIT pointer by BwdSlot
    // (ADR-0032). Called directly from Rust in compile_and_run after
    // finalize_definitions(), never injected into the JIT module itself —
    // malus source code never calls it.
    pub tape_register_backward_fn: extern "C" fn(i32, usize),
    // M26 — gradient-check test infra (record_diff builtin).
    pub malus_record_diff: extern "C" fn(f32),
}

// M26 — (BwdSlot discriminant, malus-stdlib fn name) pairs. The discriminant
// numbering must match `malus_runtime::tape::BwdSlot` exactly; drift is
// caught the same way as OpTag's (see malus-runtime tests.rs).
const BWD_SLOT_FNS: &[(i32, &str)] = &[
    (0,  "__add_bwd_a"),
    (1,  "__add_bwd_b"),
    (2,  "__sub_bwd_a"),
    (3,  "__sub_bwd_b"),
    (4,  "__mul_bwd_a"),
    (5,  "__mul_bwd_b"),
    (6,  "__div_bwd_a"),
    (7,  "__div_bwd_b"),
    (8,  "__sigmoid_bwd"),
    (9,  "__relu_bwd"),
    (10, "__tanh_bwd"),
    (11, "__broadcast_mul_fwd"), // ExpBwd: dx = dout * e
    (12, "__broadcast_div_fwd"), // LogBwd: dx = dout / x
    (13, "__sqrt_bwd"),
    (14, "__abs_bwd"),
    (15, "__scale_fwd"),         // NegBwd: dx = dout * -1.0
    (16, "__sum_bwd"),
    (17, "__permute_bwd_2d"),
    (18, "__permute_bwd_3d"),
    (19, "__reduce_sum_axis_bwd"),
    (20, "__reduce_mean_axis_bwd"),
    (21, "__reduce_max_axis_bwd"),
    (22, "__reduce_var_axis_bwd"),
    (23, "__softmax_bwd"),
    (24, "__layernorm_bwd"),
    (25, "__gelu_bwd"),
    (26, "__cross_entropy_bwd"),
    (27, "__embedding_bwd"),
    (28, "__matmul_bwd_a"),
    (29, "__matmul_bwd_b"),
    (30, "__broadcast_add_fwd"), // GradAcc
];

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

// Local mirror of malus_runtime::StrBox for the print_str shim.
// Must stay layout-compatible with the runtime's repr(C) StrBox.
#[repr(C)]
struct StrBoxLayout { ptr: *const u8, len: usize }

// M22: print_str(handle: i64) — print a runtime str value (StrBox handle).
// Local shim — same pattern as print_cstr; does not touch Metal state.
extern "C" fn print_str(handle: i64) {
    if handle == 0 { return; }
    // SAFETY: handle is a valid *const StrBox produced by malus_str_box /
    // malus_read_file / malus_str_from_char.  All are heap-allocated and leaked.
    let sb = unsafe { &*(handle as *const StrBoxLayout) };
    let bytes = unsafe { std::slice::from_raw_parts(sb.ptr, sb.len) };
    print!("{}", std::str::from_utf8(bytes).unwrap_or("<invalid utf-8>"));
}

// ── Scalar math shims ─────────────────────────────────────────────────────────

/// f32 power — shim for `**` operator; exponent already cast to f32 by codegen.
extern "C" fn malus_powf(base: f32, exp: f32) -> f32 { base.powf(exp) }

// ── dtype_tag — ScalarTy enum discriminant order ──────────────────────────────

// M14: OpTag discriminants — must stay in sync with malus_runtime::tape::OpTag.
// Drift is caught by the test_optag_from_tag_drift test in malus-runtime.
const OPTAG_MATMUL:    i32 = 0;
const OPTAG_ADD:       i32 = 1;
const OPTAG_SUB:       i32 = 2;
const OPTAG_MUL:       i32 = 3;
const OPTAG_DIV:       i32 = 4;
const OPTAG_SIGMOID:   i32 = 5;
const OPTAG_RELU:      i32 = 6;
const OPTAG_TANH:      i32 = 7;
const OPTAG_EXP:       i32 = 8;
const OPTAG_LOG:       i32 = 9;
const OPTAG_SQRT:      i32 = 10;
const OPTAG_ABS:       i32 = 11;
const OPTAG_SUM:              i32 = 12;
const OPTAG_TRANSPOSE:        i32 = 13;
const OPTAG_NEG:              i32 = 14;
const OPTAG_REDUCE_SUM_AXIS:  i32 = 15;
const OPTAG_REDUCE_MEAN_AXIS: i32 = 16;
const OPTAG_REDUCE_MAX_AXIS:  i32 = 17;
const OPTAG_REDUCE_VAR_AXIS:  i32 = 18;
const OPTAG_RESHAPE:          i32 = 19;
const OPTAG_SOFTMAX:          i32 = 20;
const OPTAG_LAYERNORM:        i32 = 21;
const OPTAG_GELU:             i32 = 22;
const OPTAG_CROSS_ENTROPY:    i32 = 23;
const OPTAG_EMBEDDING:        i32 = 24;

fn binop_to_optag(op: &BinOp) -> Option<i32> {
    match op {
        BinOp::Matmul => Some(OPTAG_MATMUL),
        BinOp::Add    => Some(OPTAG_ADD),
        BinOp::Sub    => Some(OPTAG_SUB),
        BinOp::Mul    => Some(OPTAG_MUL),
        BinOp::Div    => Some(OPTAG_DIV),
        _ => None,
    }
}

fn unary_builtin_to_optag(name: &str) -> Option<i32> {
    match name {
        "sigmoid"   => Some(OPTAG_SIGMOID),
        "relu"      => Some(OPTAG_RELU),
        "tanh"      => Some(OPTAG_TANH),
        "exp"       => Some(OPTAG_EXP),
        "log"       => Some(OPTAG_LOG),
        "sqrt"      => Some(OPTAG_SQRT),
        "abs"       => Some(OPTAG_ABS),
        "gelu"      => Some(OPTAG_GELU),
        "sum"       => Some(OPTAG_SUM),
        "transpose" => Some(OPTAG_TRANSPOSE),
        _ => None,
    }
}

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
        // Str is a leaked StrBox handle — opaque i64 pointer.
        ResolvedTy::Str => Ok(Some(I64)),
        // Tuples are heap-allocated; represented as opaque i64 pointer (same as Struct).
        ResolvedTy::Tuple(_) => Ok(Some(I64)),
        // Structs, enums, and arrays are heap-allocated; represented as opaque i64 pointer.
        ResolvedTy::Struct { .. } | ResolvedTy::Enum { .. } | ResolvedTy::Array { .. } => Ok(Some(I64)),
        // Buffer is a heap-allocated BufferData; opaque i64 pointer.
        ResolvedTy::Buffer { .. } => Ok(Some(I64)),
        // M28: List<T> is a reference-counted aggregate box (ARC header + length
        // word + elements); opaque i64 pointer, same as Struct/Array. ADR-0034.
        ResolvedTy::List { .. } => Ok(Some(I64)),
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
    // M13: aggregate ARC (alloc/retain/release with 8-byte header).
    rt_aggregate_alloc: FuncId,
    rt_aggregate_retain: FuncId,
    rt_aggregate_release: FuncId,
    // M28: List<T> release (returns previous refcount — ADR-0034).
    rt_list_release: FuncId,
    // M14: tape ABI.
    rt_tape_record_binop:  FuncId,
    rt_tape_record_unary:  FuncId,
    rt_tape_register_leaf: FuncId,
    rt_tape_pause:         FuncId,
    rt_tape_resume:        FuncId,
    rt_tape_clear:         FuncId,
    rt_tape_get_grad:      FuncId,
    rt_backward:           FuncId,
    // M15: tape ABI.
    rt_tape_zero_grad:     FuncId,
    rt_tape_record_reduce:      FuncId,
    // M17: shapes + batched matmul.
    rt_tensor_reshape:          FuncId,
    rt_tensor_permute:          FuncId,
    rt_tape_record_perm:        FuncId,
    // M18: transformer stdlib.
    rt_tensor_causal_mask:         FuncId,
    rt_tape_record_layernorm:      FuncId,
    rt_tape_record_cross_entropy:  FuncId,
    // M19: randn.
    rt_tensor_randn:               FuncId,
    rt_tape_record_embedding:      FuncId,
    // M20: scalar power operator (**).
    rt_powf:                       FuncId,
    // M22: string I/O.
    rt_malus_str_box:              FuncId,
    rt_malus_read_file:            FuncId,
    rt_malus_str_len:              FuncId,
    rt_malus_str_char_at:          FuncId,
    rt_malus_str_from_char:        FuncId,
    rt_print_str:                  FuncId,
    // M22: rand_uniform.
    rt_malus_rand_uniform:         FuncId,
    // M26: record_diff (gradient-check test infra).
    rt_malus_record_diff:          FuncId,
    // M22: Buffer<i32>.
    rt_malus_buffer_i32:           FuncId,
    rt_malus_buffer_get_i32:       FuncId,
    rt_malus_buffer_set_i32:       FuncId,
    rt_malus_buffer_free:          FuncId,
    rt_malus_buffer_freeze_i32:    FuncId,
    // M22: rand_int + tensor_get_f32.
    rt_malus_rand_int:             FuncId,
    rt_malus_tensor_get_f32:       FuncId,
    // M25: metadata accessors + kernel_dispatch_v2.
    rt_tensor_ndim:                FuncId,
    rt_tensor_dim:                 FuncId,
    rt_kernel_dispatch_v2:         FuncId,
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
                // M28: `return self.field` where `field: List<T>` hands the caller
                // an independent owned reference to a value `self` (a borrow —
                // ADR-0025) still owns. Retain here so the aliasing is genuine
                // (the caller's eventual `DropList` release balances this),
                // rather than leaving `self`'s field as the value's only owner
                // while the caller believes it holds its own reference
                // (ADR-0034). Scoped to exactly this shape — a bare struct-field
                // read in return position — since that's the only List-aliasing
                // return site the V4 fence exercises (`Module::parameters`).
                if expr.ty.is_list() && matches!(&expr.kind, TypedExprKind::FieldAccess { .. }) {
                    self.call_aggregate_retain(val);
                }
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

            TypedStmt::Assign { target, expr } => {
                use malus_sema::TypedAssignTarget;
                match target {
                    TypedAssignTarget::Ident(name) => {
                        // Ident: CTMM already inserted a drop-before-assign for the old binding.
                        // Just rebind the Cranelift variable.
                        let val = self.lower_expr(expr)?;
                        let var = self.var_map.get(name.as_str())
                            .copied()
                            .ok_or_else(|| CodegenError::UnsupportedExpr(
                                format!("assign to unknown variable: {name}")))?;
                        self.builder.def_var(var, val);
                    }
                    TypedAssignTarget::Index { base, index, elem_ty } => {
                        // Array element assign: arr_ptr + idx*8 (no ARC header).
                        // Evaluate RHS before releasing old slot (CTMM may have hoisted it).
                        let arr_ptr = self.use_var(base)?;
                        let idx_val = self.lower_expr(index)?;
                        let eight = self.builder.ins().iconst(I64, 8);
                        let byte_off = self.builder.ins().imul(idx_val, eight);
                        let slot_addr = self.builder.ins().iadd(arr_ptr, byte_off);
                        let new_val = self.lower_expr(expr)?;
                        // Release old slot element (reuse emit_drop_field: slot_addr + offset 0).
                        self.emit_drop_field(slot_addr, 0, elem_ty)?;
                        self.builder.ins().store(
                            cranelift_codegen::ir::MemFlags::trusted(), new_val, slot_addr, 0,
                        );
                    }
                    TypedAssignTarget::ListIndex { base, index, elem_ty } => {
                        // M28: List element assign — same shape as Array's Index arm,
                        // but elements start at offset 16 (8-byte ARC header + 8-byte
                        // length word), not 0 (ADR-0034). The container's own
                        // retain/release lifecycle (DropList) is independent of this
                        // per-slot element release.
                        let list_ptr = self.use_var(base)?;
                        let idx_val = self.lower_expr(index)?;
                        let eight = self.builder.ins().iconst(I64, 8);
                        let byte_off = self.builder.ins().imul(idx_val, eight);
                        let elem_off = self.builder.ins().iadd_imm(byte_off, 16);
                        let slot_addr = self.builder.ins().iadd(list_ptr, elem_off);
                        let new_val = self.lower_expr(expr)?;
                        self.emit_drop_field(slot_addr, 0, elem_ty)?;
                        self.builder.ins().store(
                            cranelift_codegen::ir::MemFlags::trusted(), new_val, slot_addr, 0,
                        );
                    }
                    TypedAssignTarget::Field { base, slot_idx, field_ty } => {
                        // Struct field assign: ptr + 8 + slot_idx*8 (8-byte ARC header).
                        let struct_ptr = self.use_var(base)?;
                        let offset = 8 + (*slot_idx as i32) * 8;
                        let new_val = self.lower_expr(expr)?;
                        // Release old field value (emit_drop_field handles Tensor/Variable/Struct/Enum).
                        self.emit_drop_field(struct_ptr, offset, field_ty)?;
                        self.builder.ins().store(
                            cranelift_codegen::ir::MemFlags::trusted(), new_val, struct_ptr, offset,
                        );
                    }
                    TypedAssignTarget::BufferIndex { base, index, .. } => {
                        // Buffer element assign: call malus_buffer_set_i32(handle, idx, val).
                        let handle = self.use_var(base)?;
                        let idx_val = self.lower_expr(index)?;
                        let new_val = self.lower_expr(expr)?;
                        let func_ref = self.import_func(self.codegen.rt_malus_buffer_set_i32);
                        self.builder.ins().call(func_ref, &[handle, idx_val, new_val]);
                    }
                }
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
                // Load the array/list pointer and determine the element type,
                // length, and element base offset. `Array` is headerless (offset
                // 0, compile-time length); `List` (M28) has an ARC header +
                // length word (offset 16, runtime length read from the box —
                // ADR-0034).
                let arr_ptr = self.lower_expr(iter)?;
                let (elem_ty, len_val, elem_base_offset) = match &iter.ty {
                    malus_sema::ResolvedTy::Array { elem, len } => {
                        let len_val = self.builder.ins().iconst(I64, *len as i64);
                        (*elem.clone(), len_val, 0i32)
                    }
                    malus_sema::ResolvedTy::List { elem } => {
                        let len_val = self.builder.ins().load(
                            I64, cranelift_codegen::ir::MemFlags::trusted(), arr_ptr, 8,
                        );
                        (*elem.clone(), len_val, 16i32)
                    }
                    _ => return Err(CodegenError::UnsupportedExpr("ForIn requires Array or List type".into())),
                };
                let cl_ty = match cranelift_type(&elem_ty)? {
                    Some(t) => t,
                    None => return Err(CodegenError::UnsupportedExpr("ForIn: unit-typed element".into())),
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
                let cmp = self.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, cur_idx, len_val,
                );
                self.builder.ins().brif(cmp, body_blk, &[], exit_blk, &[]);

                // Body: load element at arr_ptr + elem_base_offset + idx * 8.
                self.builder.switch_to_block(body_blk);
                let body_idx = self.builder.use_var(idx_var);
                let eight = self.builder.ins().iconst(I64, 8);
                let byte_offset_val = self.builder.ins().imul(body_idx, eight);
                let elem_off = self.builder.ins().iadd_imm(byte_offset_val, elem_base_offset as i64);
                let elem_ptr = self.builder.ins().iadd(arr_ptr, elem_off);
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

            // ── M13: aggregate ARC nodes ──────────────────────────────────────────
            TypedStmt::RetainAgg { name } => {
                let ptr = self.use_var(name)?;
                self.call_aggregate_retain(ptr);
                Ok(false)
            }

            TypedStmt::ReleaseAgg { name } => {
                let ptr = self.use_var(name)?;
                self.call_aggregate_release(ptr);
                Ok(false)
            }

            // ── M10/M11: aggregate types ──────────────────────────────────────────
            TypedStmt::DropStruct { name, droppable_fields, .. } => {
                let ptr = self.use_var(name)?;
                // Recursively release droppable fields (at 8 + slot*8 per M13 layout),
                // then release the struct box via aggregate ARC.
                let droppable_fields = droppable_fields.clone();
                for (slot_idx, field_ty) in &droppable_fields {
                    let offset = 8 + (*slot_idx as i32) * 8;
                    self.emit_drop_field(ptr, offset, field_ty)?;
                }
                self.call_aggregate_release(ptr);
                Ok(false)
            }

            TypedStmt::DropEnum { name, variants } => {
                let ptr = self.use_var(name)?;
                let variants: Vec<(u32, Vec<(usize, malus_sema::ResolvedTy)>, Vec<usize>)> = variants.clone();
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

            // M28: `List<T>` release (ADR-0034). Unlike `DropArray`, this is a
            // genuine reference count, not an unconditional free — `List` values
            // may alias across a call boundary (e.g. `Module::parameters`
            // returning a model's own field by identity) that neither CTMM nor
            // M29's (intraprocedural-only) borrow-inference can prove safe to
            // free unconditionally.
            //
            // Ordering is load-bearing: we PEEK the refcount (a plain read, safe
            // under this single-threaded JIT execution model — no other malus
            // code can concurrently mutate the same box), and if it looks like 1
            // (we're about to be the last reference), release each element
            // *before* the one call that actually performs the atomic decrement
            // (`call_list_release`, at the very end, common to both branches).
            // That call may deallocate the box, so nothing after it may read
            // through `ptr` again — reading the length and releasing elements
            // must happen strictly before it.
            TypedStmt::DropList { name, elem_ty } => {
                let ptr = self.use_var(name)?;
                let elem_ty = elem_ty.clone();
                let refcount = self.builder.ins().load(I64, cranelift_codegen::ir::MemFlags::trusted(), ptr, 0);
                let one = self.builder.ins().iconst(I64, 1);
                let is_last = self.builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, refcount, one);

                let last_blk = self.builder.create_block();
                let merge_blk = self.builder.create_block();
                self.builder.ins().brif(is_last, last_blk, &[], merge_blk, &[]);

                self.builder.switch_to_block(last_blk);
                self.builder.seal_block(last_blk);
                if elem_ty.is_tensor() {
                    let len_val = self.builder.ins().load(I64, cranelift_codegen::ir::MemFlags::trusted(), ptr, 8);
                    self.emit_list_element_drop_loop(ptr, len_val)?;
                }
                // Struct/Enum/Array-typed List elements are out of V4 scope (ADR-0034
                // — the capstone only ever instantiates `List<Tensor<f32>>`); silently
                // skipped rather than attempted, matching the documented fence.
                self.builder.ins().jump(merge_blk, &[]);

                self.builder.switch_to_block(merge_blk);
                self.builder.seal_block(merge_blk);
                self.call_list_release(ptr);
                Ok(false)
            }

            // ── M13.5: tuples ─────────────────────────────────────────────────────
            TypedStmt::DropTuple { name, droppable_fields } => {
                let ptr = self.use_var(name)?;
                let droppable_fields = droppable_fields.clone();
                for (slot_idx, field_ty) in &droppable_fields {
                    let offset = 8 + (*slot_idx as i32) * 8;
                    self.emit_drop_field(ptr, offset, field_ty)?;
                }
                self.call_aggregate_release(ptr);
                Ok(false)
            }

            // ── M22: Buffer<i32> ──────────────────────────────────────────────────
            TypedStmt::DropBuffer { name } => {
                let handle = self.use_var(name)?;
                let func_ref = self.import_func(self.codegen.rt_malus_buffer_free);
                self.builder.ins().call(func_ref, &[handle]);
                Ok(false)
            }

            TypedStmt::LetTuple { names, expr } => {
                // Detect whether the source is a named local binding (Ident).
                // If so, CTMM will insert DropTuple for it — don't double-free the box.
                // For temporaries (TupleInit, Call, etc.) we must free the box here.
                let expr_is_ident = matches!(&expr.kind, TypedExprKind::Ident(_));
                let ptr = self.lower_expr(expr)?;
                // Extract each field into a new Cranelift variable.
                // Fields are at ptr + 8 + i*8 (8-byte ARC header + field slots).
                let names = names.clone();
                for (i, (name, ty)) in names.iter().enumerate() {
                    let cl_ty = match cranelift_type(ty)? {
                        Some(t) => t,
                        None => continue,
                    };
                    let offset = 8 + (i as i32) * 8;
                    let val = self.builder.ins().load(
                        cl_ty, cranelift_codegen::ir::MemFlags::trusted(), ptr, offset,
                    );
                    // Tensor fields: retain so DropTuple's tensor_release balances
                    // with the Drop for the extracted binding. Only needed when DropTuple fires.
                    if expr_is_ident && ty.is_tensor() {
                        self.call_runtime_retain(val);
                    }
                    let var = Variable::from_u32(self.next_var as u32);
                    self.next_var += 1;
                    self.builder.declare_var(var, cl_ty);
                    self.builder.def_var(var, val);
                    self.var_map.insert(name.clone(), var);
                }
                // Free the box only for temporaries — named bindings get DropTuple from CTMM.
                if !expr_is_ident {
                    self.call_aggregate_release(ptr);
                }
                Ok(false)
            }

            TypedStmt::Match { scrutinee, arms } => {
                self.lower_match(scrutinee, arms)
            }

            // ── M14: no_grad scope ───────────────────────────────────────────────
            TypedStmt::NoGrad { body } => {
                self.call_tape_pause();
                for s in body {
                    self.lower_stmt(s)?;
                }
                self.call_tape_resume();
                Ok(false)
            }

            // ── M24: kernel shared memory ─────────────────────────────────────────
            // LetShared is a kernel-body-only statement; codegen-cpu never lowers
            // kernel bodies, so this arm is unreachable in practice.
            TypedStmt::LetShared { .. } => {
                unreachable!("LetShared is only valid in kernel bodies (lowered by codegen-gpu)");
            }
        }
    }

    fn lower_match(&mut self, scrutinee: &TypedExpr, arms: &[TypedMatchArm]) -> Result<bool, CodegenError> {
        let scrut_ptr = self.lower_expr(scrutinee)?;
        // Load the u32 tag stored at offset 8 (after the 8-byte ARC header).
        let tag = self.builder.ins().load(I32, cranelift_codegen::ir::MemFlags::trusted(), scrut_ptr, 8);

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

            // Bind payload fields from pointer (offset 16 + j*8 per field after M13 ARC header).
            for (j, (binding_name, binding_ty)) in arm.bindings.iter().enumerate() {
                let offset = 16 + (j as i32) * 8;
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
                    self.emit_drop_field(nested_ptr, 8 + (*fidx as i32) * 8, fty)?;
                }
                self.call_aggregate_release(nested_ptr);
            }
            ResolvedTy::Enum { variants, .. } => {
                let nested_ptr = self.builder.ins().load(
                    I64, cranelift_codegen::ir::MemFlags::trusted(), base_ptr, byte_offset,
                );
                let drop_variants: Vec<(u32, Vec<(usize, ResolvedTy)>, Vec<usize>)> = variants.iter()
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
                        (tag as u32, droppable, vec![])
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
        variants: &[(u32, Vec<(usize, malus_sema::ResolvedTy)>, Vec<usize>)],
    ) -> Result<(), CodegenError> {
        // Tag is at offset 8 (after the 8-byte ARC header).
        let tag = self.builder.ins().load(
            I32, cranelift_codegen::ir::MemFlags::trusted(), ptr, 8,
        );

        let merge_blk = self.builder.create_block();
        let arm_test_blks: Vec<_> = variants.iter().map(|_| self.builder.create_block()).collect();
        let unreachable_blk = self.builder.create_block();

        if let Some(&first) = arm_test_blks.first() {
            self.builder.ins().jump(first, &[]);
        } else {
            // No variants — release the box and return.
            self.call_aggregate_release(ptr);
            return Ok(());
        }

        let variants_clone: Vec<(u32, Vec<(usize, malus_sema::ResolvedTy)>, Vec<usize>)> = variants.to_vec();
        for (i, (variant_tag, droppable_fields, _)) in variants_clone.iter().enumerate() {
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

            // Release droppable fields (payload starts at offset 16 after ARC header + tag).
            let droppable = droppable_fields.clone();
            for (slot_idx, field_ty) in &droppable {
                let offset = 16 + (*slot_idx as i32) * 8;
                self.emit_drop_field(ptr, offset, field_ty)?;
            }
            self.builder.ins().jump(merge_blk, &[]);
        }

        self.builder.switch_to_block(unreachable_blk);
        self.builder.seal_block(unreachable_blk);
        self.builder.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());

        self.builder.switch_to_block(merge_blk);
        self.builder.seal_block(merge_blk);
        self.call_aggregate_release(ptr);

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
                    let offset = 8 + (*slot_idx as i32) * 8;
                    self.emit_drop_field(handle, offset, field_ty)?;
                }
                self.call_aggregate_release(handle);
            }
            malus_sema::ResolvedTy::Enum { variants, .. } => {
                let drop_variants: Vec<(u32, Vec<(usize, malus_sema::ResolvedTy)>, Vec<usize>)> = variants.iter()
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
                        (vi as u32, droppable, vec![])
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

    /// M28: emit a loop that releases each `Tensor`-typed element of a
    /// `List<Tensor<f32>>` at `list_ptr`, for a RUNTIME-known `len_val` (unlike
    /// `emit_counted_drop_loop`'s compile-time `len: usize` — `List` carries its
    /// length in the box, not the type). Elements start at offset 16 (ARC header
    /// + length word), not `Array`'s 0. Only called when the caller has already
    /// confirmed `elem_ty.is_tensor()` — V4 scope is `List<Tensor<f32>>` only
    /// (ADR-0034); non-tensor List elements are not released here.
    fn emit_list_element_drop_loop(
        &mut self,
        list_ptr: cranelift_codegen::ir::Value,
        len_val: cranelift_codegen::ir::Value,
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
        let cmp = self.builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, cur, len_val,
        );
        self.builder.ins().brif(cmp, body_blk, &[], exit_blk, &[]);

        self.builder.switch_to_block(body_blk);
        self.builder.seal_block(body_blk);
        let body_cur = self.builder.use_var(idx_var);
        let eight = self.builder.ins().iconst(I64, 8);
        let byte_off = self.builder.ins().imul(body_cur, eight);
        let elem_off = self.builder.ins().iadd_imm(byte_off, 16);
        let elem_ptr = self.builder.ins().iadd(list_ptr, elem_off);
        let handle = self.builder.ins().load(
            I64, cranelift_codegen::ir::MemFlags::trusted(), elem_ptr, 0,
        );
        self.call_runtime_release(handle);
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
                    // Tensor ⊕ Tensor — matmul or broadcast-aware element-wise, plus
                    // tape_record_binop when the result is grad-tracked (M27:
                    // `expr.grad_tracked`, set by `grad_inference.rs`, replaces the old
                    // the old distinct Variable type; ADR-0030).
                    (ResolvedTy::Tensor { dtype: ld }, ResolvedTy::Tensor { dtype: rd })
                        if ld == rd && *ld == ScalarTy::F32 =>
                    {
                        let a = self.lower_expr(lhs)?;
                        let b = self.lower_expr(rhs)?;
                        let out = if *op == BinOp::Matmul {
                            let matmul_ref = self.import_func(self.codegen.rt_tensor_matmul);
                            let call = self.builder.ins().call(matmul_ref, &[a, b]);
                            self.builder.inst_results(call).to_vec()[0]
                        } else if let Some(kernel_name) = elementwise_builtin_name(op) {
                            self.lower_broadcast_binop(kernel_name, a, b)?
                        } else {
                            return Err(CodegenError::UnsupportedExpr(format!("binop {:?} on tensors not supported", op)));
                        };
                        if expr.grad_tracked {
                            let op_tag = binop_to_optag(op).ok_or_else(|| {
                                CodegenError::UnsupportedExpr(format!("BinOp {:?} not supported on tape", op))
                            })?;
                            self.call_tape_record_binop(op_tag, a, b, out);
                        }
                        Ok(out)
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
                match op {
                    UnaryOp::Neg if operand.ty.is_tensor() => {
                        // Tensor neg — forward: malus_mul_scalar(x, -1.0).
                        // No Neg GPU kernel; reuse scalar-broadcast Mul with a -1 scalar tensor.
                        // tape_record_unary only when grad-tracked (M27 grad_inference.rs;
                        // replaces the old the old distinct Variable type; ADR-0030).
                        let x = self.lower_expr(operand)?;
                        let neg_one = self.builder.ins().f32const(-1.0_f32);
                        let scalar_handle = self.emit_scalar_tensor(neg_one);
                        let out = self.lower_kernel_dispatch_with_handles("malus_mul_scalar", &[x, scalar_handle])?;
                        self.call_runtime_free(scalar_handle);
                        if operand.grad_tracked {
                            self.call_tape_record_unary(OPTAG_NEG, x, out);
                        }
                        Ok(out)
                    }
                    UnaryOp::Neg => {
                        let val = self.lower_expr(operand)?;
                        match &operand.ty {
                            ResolvedTy::Scalar(s) if is_float_scalar(s) => Ok(self.builder.ins().fneg(val)),
                            ResolvedTy::Scalar(_) => Ok(self.builder.ins().ineg(val)),
                            _ => Err(CodegenError::UnsupportedExpr("Neg on non-scalar".into())),
                        }
                    }
                    UnaryOp::Not => {
                        let val = self.lower_expr(operand)?;
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
                    if expr.grad_tracked {
                        // Grad-tracked input: lower arg first so we have x for tape_record_unary.
                        let op_tag = unary_builtin_to_optag(callee).ok_or_else(|| {
                            CodegenError::UnsupportedExpr(format!("no OpTag for grad-tracked builtin '{callee}'"))
                        })?;
                        let x = self.lower_expr(&args[0])?;
                        let out = self.lower_kernel_dispatch_with_handles(callee, &[x])?;
                        self.call_tape_record_unary(op_tag, x, out);
                        Ok(out)
                    } else {
                        let arg_refs: Vec<&TypedExpr> = args.iter().collect();
                        self.lower_kernel_dispatch(callee, &arg_refs)
                    }
                } else if callee == "zeros" || callee == "ones" || callee == "randn" {
                    self.lower_zeros_ones(callee, args)
                } else if callee == "reshape" || callee == "transpose" || callee == "permute" {
                    self.lower_shape_op(callee, args, expr.grad_tracked)
                } else if callee == "sum" && args.len() == 1 {
                    // sum(t) — whole-tensor sum (no axis).
                    if expr.grad_tracked {
                        let x = self.lower_expr(&args[0])?;
                        let out = self.lower_eager_cpu_op_with_handle("sum", x)?;
                        self.call_tape_record_unary(OPTAG_SUM, x, out);
                        Ok(out)
                    } else {
                        self.lower_eager_cpu_op("sum", args)
                    }
                } else if callee == "sum" || callee == "mean" || callee == "max" || callee == "var" {
                    // sum(t, axis=N, keepdim=K) or mean/max/var(t, axis=N, keepdim=K) — axis reduction.
                    // args are positional [tensor, axis, keepdim] after sema normalization.
                    let op_tag = match callee.as_str() {
                        "sum"  => OPTAG_REDUCE_SUM_AXIS,
                        "mean" => OPTAG_REDUCE_MEAN_AXIS,
                        "max"  => OPTAG_REDUCE_MAX_AXIS,
                        "var"  => OPTAG_REDUCE_VAR_AXIS,
                        _      => unreachable!(),
                    };
                    self.lower_axis_reduction(op_tag, expr.grad_tracked, args)
                } else if callee == "softmax" {
                    self.lower_softmax(expr.grad_tracked, args)
                } else if callee == "layernorm" {
                    self.lower_layernorm(expr.grad_tracked, args)
                } else if callee == "gelu" {
                    let x = self.lower_expr(&args[0])?;
                    let func_id = self.codegen.func_ids.get("__gelu_fwd")
                        .copied()
                        .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __gelu_fwd not found".into()))?;
                    let func_ref = self.import_func(func_id);
                    let call = self.builder.ins().call(func_ref, &[x]);
                    let out  = self.builder.inst_results(call).to_vec()[0];
                    if expr.grad_tracked {
                        self.call_tape_record_unary(OPTAG_GELU, x, out);
                    }
                    Ok(out)
                } else if callee == "embedding" {
                    self.lower_embedding(args, expr.grad_tracked)
                } else if callee == "cross_entropy" {
                    self.lower_cross_entropy(args, expr.grad_tracked)
                } else if callee == "causal_mask" {
                    self.lower_causal_mask(args)
                } else if callee == "variable" {
                    // variable(t) — leaf marker: retain + tape_register_leaf.
                    let handle = self.lower_expr(&args[0])?;
                    self.call_runtime_retain(handle);
                    self.call_tape_register_leaf(handle);
                    Ok(handle)
                } else if callee == "backward" {
                    // backward(loss) — walk the tape in reverse, accumulate grads.
                    let handle = self.lower_expr(&args[0])?;
                    self.call_backward(handle);
                    Ok(self.builder.ins().iconst(I64, 0))
                } else if callee == "zero_grad" {
                    // zero_grad(v1, v2, ...) — clear accumulated grads for given Variables.
                    let n = args.len() as u32;
                    if n == 0 {
                        return Ok(self.builder.ins().iconst(I64, 0));
                    }
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
                    let handles_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
                    let n_val = self.builder.ins().iconst(self.codegen.ptr_type(), n as i64);
                    self.call_tape_zero_grad(handles_ptr, n_val);
                    Ok(self.builder.ins().iconst(I64, 0))
                } else if callee == "tensor_print" {
                    let handle = self.lower_expr(&args[0])?;
                    self.call_runtime_print(handle);
                    Ok(self.builder.ins().iconst(I64, 0))
                } else if callee == "read_file" {
                    let path = self.lower_expr(&args[0])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_read_file);
                    let call = self.builder.ins().call(func_ref, &[path]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if callee == "str_len" {
                    let s = self.lower_expr(&args[0])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_str_len);
                    let call = self.builder.ins().call(func_ref, &[s]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if callee == "len" {
                    // M28: len(lst) — the length word lives inline in the box at
                    // offset 8 (ARC header + length — ADR-0034); no runtime call
                    // needed.
                    let list_ptr = self.lower_expr(&args[0])?;
                    Ok(self.builder.ins().load(I64, cranelift_codegen::ir::MemFlags::trusted(), list_ptr, 8))
                } else if callee == "str_char_at" {
                    let s = self.lower_expr(&args[0])?;
                    let i = self.lower_expr(&args[1])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_str_char_at);
                    let call = self.builder.ins().call(func_ref, &[s, i]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if callee == "str_from_char" {
                    let c = self.lower_expr(&args[0])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_str_from_char);
                    let call = self.builder.ins().call(func_ref, &[c]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if callee == "rand_uniform" {
                    let func_ref = self.import_func(self.codegen.rt_malus_rand_uniform);
                    let call = self.builder.ins().call(func_ref, &[]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if callee == "record_diff" {
                    let v = self.lower_expr(&args[0])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_record_diff);
                    self.builder.ins().call(func_ref, &[v]);
                    Ok(self.builder.ins().iconst(I64, 0))
                } else if callee == "rand_int" {
                    let n = self.lower_expr(&args[0])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_rand_int);
                    let call = self.builder.ins().call(func_ref, &[n]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if callee == "buffer_i32" {
                    let n = self.lower_expr(&args[0])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_buffer_i32);
                    let call = self.builder.ins().call(func_ref, &[n]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if callee == "freeze" {
                    let handle = self.lower_expr(&args[0])?;
                    let func_ref = self.import_func(self.codegen.rt_malus_buffer_freeze_i32);
                    let call = self.builder.ins().call(func_ref, &[handle]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
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
                // Determine element size: i64 = 8 bytes, everything else (f32, i32) = 4 bytes.
                let elem_size: u32 = match dtype { ScalarTy::I64 => 8, _ => 4 };
                let align: u8      = if elem_size == 8 { 3 } else { 2 };
                let data_slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        (len * elem_size).max(elem_size),
                        align,
                    )
                );
                for (i, elem) in elements.iter().enumerate() {
                    let val = self.lower_expr(elem)?;
                    // For integer tensor dtypes, store native int values (no f32 conversion).
                    // For float dtypes, convert any integer literals to f32.
                    let stored = match dtype {
                        ScalarTy::I32 | ScalarTy::I64 => val,
                        _ => match &elem.ty {
                            ResolvedTy::Scalar(s) if !is_float_scalar(s) => {
                                self.builder.ins().fcvt_from_sint(F32, val)
                            }
                            _ => val,
                        },
                    };
                    self.builder.ins().stack_store(stored, data_slot, (i as i32) * (elem_size as i32));
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
                // M25: detect `t.shape[i]` / `t.strides[i]` — fuse into tensor_dim(t, i).
                // sema types .shape as Array<I64,8>; the array itself is not materialised.
                if let TypedExprKind::FieldAccess { base: fa_base, field } = &base.kind {
                    if (field == "shape" || field == "strides") && fa_base.ty.is_tensor() {
                        let handle = self.lower_expr(fa_base)?;
                        let idx_val = self.lower_expr(&indices[0])?;
                        let dim_ref = self.import_func(self.codegen.rt_tensor_dim);
                        let call = self.builder.ins().call(dim_ref, &[handle, idx_val]);
                        return Ok(self.builder.inst_results(call).to_vec()[0]);
                    }
                }
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
                    ResolvedTy::Buffer { .. } => {
                        // Buffer[i] — call malus_buffer_get_i32(handle, idx) -> i64
                        let buf_handle = self.lower_expr(base)?;
                        let idx_val = self.lower_expr(&indices[0])?;
                        let func_ref = self.import_func(self.codegen.rt_malus_buffer_get_i32);
                        let call = self.builder.ins().call(func_ref, &[buf_handle, idx_val]);
                        Ok(self.builder.inst_results(call).to_vec()[0])
                    }
                    ResolvedTy::Tensor { .. } => {
                        // Tensor<f32>[i] — flat row-major read; gpu_barrier injected by CTMM.
                        let ten_handle = self.lower_expr(base)?;
                        let idx_val = self.lower_expr(&indices[0])?;
                        let func_ref = self.import_func(self.codegen.rt_malus_tensor_get_f32);
                        let call = self.builder.ins().call(func_ref, &[ten_handle, idx_val]);
                        Ok(self.builder.inst_results(call).to_vec()[0])
                    }
                    // M28: List<T>[i] — same read shape as Array, but elements start
                    // at offset 16 (ARC header + length word), not 0 (ADR-0034).
                    ResolvedTy::List { .. } => {
                        let list_ptr = self.lower_expr(base)?;
                        let idx_val = self.lower_expr(&indices[0])?;
                        let eight = self.builder.ins().iconst(I64, 8);
                        let byte_off = self.builder.ins().imul(idx_val, eight);
                        let elem_off = self.builder.ins().iadd_imm(byte_off, 16);
                        let elem_ptr = self.builder.ins().iadd(list_ptr, elem_off);
                        let cl_ty = match cranelift_type(&expr.ty)? {
                            Some(t) => t,
                            None => return Err(CodegenError::UnsupportedExpr("Index: unit element type".into())),
                        };
                        let val = self.builder.ins().load(cl_ty, cranelift_codegen::ir::MemFlags::trusted(), elem_ptr, 0);
                        Ok(val)
                    }
                    _ => Err(CodegenError::UnsupportedExpr("Index: only Array, Buffer, Tensor<f32>, or List indexing is supported".into())),
                }
            }

            TypedExprKind::ArrayLiteral { elements } => {
                let n = elements.len();
                // M28: a List<T> literal is a reference-counted aggregate — ARC
                // header (8 bytes) + length word (8 bytes) + N element slots
                // (ADR-0034) — NOT Array's headerless `[e0, e1, ...]` layout.
                // Disambiguated purely by this expr's resolved type (List vs
                // Array); both share the same `ArrayLiteral` typed-IR node.
                if expr.ty.is_list() {
                    let size = (16 + n * 8) as i64;
                    let list_ptr = self.call_aggregate_alloc(size);
                    let len_val = self.builder.ins().iconst(I64, n as i64);
                    self.builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(), len_val, list_ptr, 8,
                    );
                    for (i, elem_expr) in elements.iter().enumerate() {
                        let val = self.lower_expr(elem_expr)?;
                        self.builder.ins().store(
                            cranelift_codegen::ir::MemFlags::trusted(),
                            val,
                            list_ptr,
                            16 + (i as i32) * 8,
                        );
                    }
                    return Ok(list_ptr);
                }
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
                } else if field == "ndim" && base.ty.is_tensor() {
                    // M25: x.ndim → tensor_ndim(handle) -> i64
                    let handle = self.lower_expr(base)?;
                    let ndim_ref = self.import_func(self.codegen.rt_tensor_ndim);
                    let call = self.builder.ins().call(ndim_ref, &[handle]);
                    Ok(self.builder.inst_results(call).to_vec()[0])
                } else if (field == "shape" || field == "strides") && base.ty.is_tensor() {
                    // M25: x.shape / x.strides — sema types as Array<i64,8>; consumed only
                    // via x.shape[i] which is caught in the Index arm below.  Getting the array
                    // itself is not representable at runtime (no descriptor pointer in the host);
                    // if this arm is ever reached without an enclosing index the codegen emits 0.
                    // (Sema already rejected bare .shape / .strides accesses other than .shape[k].)
                    self.lower_expr(base)?; // lower for side effects (retain counts etc.)
                    Ok(self.builder.ins().iconst(I64, 0))
                } else if field == "data" && base.ty.is_tensor() {
                    // .data is a detach (M27): same handle, just no longer grad-tracked
                    // in the typed IR (grad_inference.rs forces it false; ADR-0030).
                    self.lower_expr(base)
                } else if field == "grad" && base.ty.is_tensor() {
                    // .grad — call tape_get_grad; returns an owned Tensor handle (D5).
                    let handle = self.lower_expr(base)?;
                    Ok(self.call_tape_get_grad(handle))
                } else if let ResolvedTy::Struct { fields, .. } = &base.ty {
                    let slot_idx = fields.iter().position(|(n, _)| n == field)
                        .ok_or_else(|| CodegenError::UnsupportedExpr(format!("struct has no field '{field}'")))?;
                    let ptr = self.lower_expr(base)?;
                    // Offset by 8 to skip the ARC header (M13 layout).
                    let offset = 8 + (slot_idx as i32) * 8;
                    let cl_ty = match cranelift_type(&expr.ty)? {
                        Some(t) => t,
                        None => return Err(CodegenError::UnsupportedExpr("unit field access".into())),
                    };
                    let val = self.builder.ins().load(cl_ty, cranelift_codegen::ir::MemFlags::trusted(), ptr, offset);
                    Ok(val)
                } else {
                    Err(CodegenError::UnsupportedExpr(format!("FieldAccess .{field} not yet supported")))
                }
            }

            TypedExprKind::StructInit { fields, name } => {
                let n_slots = fields.len();
                // 8-byte ARC header + n_slots * 8 bytes for fields (M13 layout).
                let size = (8 + n_slots * 8) as i64;
                let ptr = self.call_aggregate_alloc(size);
                for (i, field_expr) in fields.iter().enumerate() {
                    let val = self.lower_expr(field_expr)?;
                    // Offset by 8 for ARC header; slot i at byte 8 + i*8.
                    self.builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        val,
                        ptr,
                        8 + (i as i32) * 8,
                    );
                }
                let _ = name;
                Ok(ptr)
            }

            TypedExprKind::EnumInit { variant_index, payload, max_payload_slots, .. } => {
                // Layout: 8-byte ARC header + 8-byte tag slot + max_payload_slots * 8 bytes.
                let total = 8 + (1 + max_payload_slots) * 8;
                let ptr = self.call_aggregate_alloc(total as i64);
                // Store u32 tag at offset 8 (after ARC header).
                let tag_val = self.builder.ins().iconst(I32, *variant_index as i64);
                self.builder.ins().store(
                    cranelift_codegen::ir::MemFlags::trusted(),
                    tag_val,
                    ptr,
                    8,
                );
                // Store payload fields at offsets 16 + j*8 (after ARC header + tag).
                for (j, field_expr) in payload.iter().enumerate() {
                    let val = self.lower_expr(field_expr)?;
                    self.builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        val,
                        ptr,
                        16 + (j as i32) * 8,
                    );
                }
                Ok(ptr)
            }

            // ── M13.5: tuples ────────────────────────────────────────────────────
            TypedExprKind::TupleInit { elements } => {
                // Layout: 8-byte ARC header + N * 8 bytes (one slot per element).
                let n = elements.len();
                let size = (8 + n * 8) as i64;
                let ptr = self.call_aggregate_alloc(size);
                for (i, elem_expr) in elements.iter().enumerate() {
                    let val = self.lower_expr(elem_expr)?;
                    self.builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        val,
                        ptr,
                        8 + (i as i32) * 8,
                    );
                }
                Ok(ptr)
            }

            TypedExprKind::TupleIndex { base, index } => {
                let ptr = self.lower_expr(base)?;
                let offset = 8 + (*index as i32) * 8;
                let cl_ty = match cranelift_type(&expr.ty)? {
                    Some(t) => t,
                    None => return Err(CodegenError::UnsupportedExpr("tuple index has unit type".into())),
                };
                let val = self.builder.ins().load(
                    cl_ty, cranelift_codegen::ir::MemFlags::trusted(), ptr, offset,
                );
                Ok(val)
            }

            // M25: host-side kernel launch — `kernel[grid=[..], tg=[..], out=[..]](tensors, scalars)`.
            // Lowers to kernel_dispatch_v2(id, handles_ptr, hc, grid_ptr, tg_ptr,
            //   out_shape_ptr, out_ndim, out_dtype_tag, uniforms_ptr, uniforms_bytes).
            TypedExprKind::KernelLaunch { kernel, grid, tg, out_shape, tensor_args, scalar_args } => {
                let kernel_id = *self.codegen.kernel_ids.get(kernel.as_str())
                    .ok_or_else(|| CodegenError::UnknownKernel { name: kernel.clone() })?;

                // 1. Handles slot: [i64; hc]
                let hc = tensor_args.len() as u32;
                let handles_slot = if hc > 0 {
                    let slot = self.builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            hc * 8,
                            3,
                        )
                    );
                    for (i, ta) in tensor_args.iter().enumerate() {
                        let v = self.lower_expr(ta)?;
                        self.builder.ins().stack_store(v, slot, (i as i32) * 8);
                    }
                    slot
                } else {
                    // Degenerate: create a 1-slot dummy; runtime ignores it when hc==0.
                    self.builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            8,
                            3,
                        )
                    )
                };

                // 2. Grid slot: [usize; 3]  (each element stored as usize=i64)
                let grid_slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        24,
                        3,
                    )
                );
                {
                    let grid_ptr_v = self.lower_expr(grid)?;
                    for dim in 0..3i32 {
                        let off = self.builder.ins().iconst(self.codegen.ptr_type(), (dim as i64) * 8);
                        let src = self.builder.ins().iadd(grid_ptr_v, off);
                        let v = self.builder.ins().load(I64, cranelift_codegen::ir::MemFlags::trusted(), src, 0);
                        self.builder.ins().stack_store(v, grid_slot, dim * 8);
                    }
                }

                // 3. Tg slot: [usize; 3]
                let tg_slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        24,
                        3,
                    )
                );
                {
                    let tg_ptr_v = self.lower_expr(tg)?;
                    for dim in 0..3i32 {
                        let off = self.builder.ins().iconst(self.codegen.ptr_type(), (dim as i64) * 8);
                        let src = self.builder.ins().iadd(tg_ptr_v, off);
                        let v = self.builder.ins().load(I64, cranelift_codegen::ir::MemFlags::trusted(), src, 0);
                        self.builder.ins().stack_store(v, tg_slot, dim * 8);
                    }
                }

                // 4. Out shape slot + ndim + dtype_tag.
                // If out_shape is provided: use it (Array<i64, N>); out_ndim = runtime len from type.
                // If absent: pass null/0 — runtime uses first input's shape.
                let (out_shape_ptr, out_ndim_val, out_dtype_val) = if let Some(os) = out_shape {
                    // The out_shape expr is an Array<i64, 3> (same form as grid/tg).
                    let os_arr = self.lower_expr(os)?;
                    let out_slot = self.builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            24,
                            3,
                        )
                    );
                    for dim in 0..3i32 {
                        let off = self.builder.ins().iconst(self.codegen.ptr_type(), (dim as i64) * 8);
                        let src = self.builder.ins().iadd(os_arr, off);
                        let v = self.builder.ins().load(I64, cranelift_codegen::ir::MemFlags::trusted(), src, 0);
                        self.builder.ins().stack_store(v, out_slot, dim * 8);
                    }
                    let ptr_v = self.builder.ins().stack_addr(self.codegen.ptr_type(), out_slot, 0);
                    // Infer out_ndim by stripping trailing literal-0 dimensions.
                    // Writing out=[d0, d1, 0] means 2D; out=[d0, 0, 0] means 1D.
                    // Use 0 (not 1) as the sentinel — a size-0 dimension is impossible;
                    // keepdim=True can produce trailing-1 dims which must NOT be stripped.
                    let out_ndim: usize = if let TypedExprKind::ArrayLiteral { elements } = &os.kind {
                        let mut n = elements.len();
                        while n > 1 {
                            if matches!(elements[n-1].kind, TypedExprKind::Lit(malus_syntax::ast::Lit::Int(0))) {
                                n -= 1;
                            } else {
                                break;
                            }
                        }
                        n
                    } else {
                        3
                    };
                    let ndim_v = self.builder.ins().iconst(self.codegen.ptr_type(), out_ndim as i64);
                    // dtype from expr's Tensor<dtype>
                    let dt = match &expr.ty {
                        ResolvedTy::Tensor { dtype } => dtype_tag(dtype),
                        _ => 0,
                    };
                    let dt_v = self.builder.ins().iconst(I32, dt as i64);
                    (ptr_v, ndim_v, dt_v)
                } else {
                    let null_ptr = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
                    let zero_ndim = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
                    let dt = match &expr.ty {
                        ResolvedTy::Tensor { dtype } => dtype_tag(dtype),
                        _ => 0,
                    };
                    let dt_v = self.builder.ins().iconst(I32, dt as i64);
                    (null_ptr, zero_ndim, dt_v)
                };

                // 5. Uniforms blob: pack scalar_args sequentially (each as f32 or i32/i64).
                let (uniforms_ptr, uniforms_bytes_val) = if scalar_args.is_empty() {
                    let null_ptr = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
                    let zero = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
                    (null_ptr, zero)
                } else {
                    // Each uniform is 4 bytes (f32 or i32); i64 uniforms truncated to i64 (8 bytes).
                    // Use 8-byte slots for alignment-safety (codegen-gpu packs in declaration order).
                    let ub = (scalar_args.len() * 4) as u32;
                    let uni_slot = self.builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            ub.max(4),
                            2,
                        )
                    );
                    let mut byte_off: i32 = 0;
                    for sa in scalar_args.iter() {
                        let v = self.lower_expr(sa)?;
                        match &sa.ty {
                            ResolvedTy::Scalar(s) if is_float_scalar(s) => {
                                self.builder.ins().stack_store(v, uni_slot, byte_off);
                                byte_off += 4;
                            }
                            ResolvedTy::Scalar(_) => {
                                // integer: truncate to i32 and store
                                let v32 = self.builder.ins().ireduce(I32, v);
                                self.builder.ins().stack_store(v32, uni_slot, byte_off);
                                byte_off += 4;
                            }
                            _ => return Err(CodegenError::UnsupportedExpr("KernelLaunch: scalar_arg is not a scalar".into())),
                        }
                    }
                    let uni_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), uni_slot, 0);
                    let ub_val = self.builder.ins().iconst(self.codegen.ptr_type(), ub as i64);
                    (uni_ptr, ub_val)
                };

                // 6. Call kernel_dispatch_v2.
                let id_val = self.builder.ins().iconst(I64, kernel_id as i64);
                let handles_ptr_v = self.builder.ins().stack_addr(self.codegen.ptr_type(), handles_slot, 0);
                let hc_val = self.builder.ins().iconst(self.codegen.ptr_type(), hc as i64);
                let grid_ptr_v = self.builder.ins().stack_addr(self.codegen.ptr_type(), grid_slot, 0);
                let tg_ptr_v = self.builder.ins().stack_addr(self.codegen.ptr_type(), tg_slot, 0);

                let dispatch_ref = self.import_func(self.codegen.rt_kernel_dispatch_v2);
                let call = self.builder.ins().call(dispatch_ref, &[
                    id_val, handles_ptr_v, hc_val,
                    grid_ptr_v, tg_ptr_v,
                    out_shape_ptr, out_ndim_val, out_dtype_val,
                    uniforms_ptr, uniforms_bytes_val,
                ]);
                Ok(self.builder.inst_results(call).to_vec()[0])
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
            Lit::Str(s) => {
                // Materialise a runtime Str handle from a compile-time literal.
                // Emit the string bytes as static anonymous data (with NUL for
                // safety, but StrBox.len does NOT include the NUL).
                let len = s.len();
                let data_ptr = self.emit_static_cstr(s);
                let len_val = self.builder.ins().iconst(I64, len as i64);
                let func_ref = self.import_func(self.codegen.rt_malus_str_box);
                let call = self.builder.ins().call(func_ref, &[data_ptr, len_val]);
                Ok(self.builder.inst_results(call).to_vec()[0])
            }
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
            BinOp::Pow => {
                // f32 ** {f32|i32|i64} → f32.  LHS is always f32; cast RHS to f32 if needed.
                let r_ty = self.builder.func.dfg.value_type(r);
                let r_f32 = if r_ty == F32 {
                    r
                } else {
                    self.builder.ins().fcvt_from_sint(F32, r)
                };
                let powf_ref = self.import_func(self.codegen.rt_powf);
                let call = self.builder.ins().call(powf_ref, &[l, r_f32]);
                Ok(self.builder.inst_results(call).to_vec()[0])
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
            ResolvedTy::Str => {
                let func_ref = self.import_func(self.codegen.rt_print_str);
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

    fn emit_scalar_tensor(
        &mut self,
        f32_val: cranelift_codegen::ir::Value,
    ) -> cranelift_codegen::ir::Value {
        let data_slot = self.builder.create_sized_stack_slot(
            cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                4,
                2,
            )
        );
        self.builder.ins().stack_store(f32_val, data_slot, 0);
        let shape_slot = self.builder.create_sized_stack_slot(
            cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                8,
                3,
            )
        );
        let one = self.builder.ins().iconst(I64, 1);
        self.builder.ins().stack_store(one, shape_slot, 0);
        let data_ptr  = self.builder.ins().stack_addr(self.codegen.ptr_type(), data_slot, 0);
        let shape_ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), shape_slot, 0);
        let dtype_val = self.builder.ins().iconst(I32, 0); // F32 = tag 0
        let ndims_val = self.builder.ins().iconst(self.codegen.ptr_type(), 1);
        let alloc_ref = self.import_func(self.codegen.rt_tensor_alloc_gpu);
        let call = self.builder.ins().call(alloc_ref, &[dtype_val, shape_ptr, ndims_val, data_ptr]);
        self.builder.inst_results(call).to_vec()[0]
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
        } else if name == "ones" {
            self.import_func(self.codegen.rt_tensor_alloc_ones_gpu)
        } else {
            // randn — same ABI as zeros/ones
            self.import_func(self.codegen.rt_tensor_randn)
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
        self.lower_eager_cpu_op_with_handle(name, handle)
    }

    fn lower_eager_cpu_op_with_handle(
        &mut self,
        _name: &str,
        handle: cranelift_codegen::ir::Value,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        // M26: sum(t) routes to the malus __reduce_all_sum_fwd kernel instead
        // of the retired CPU rt_tensor_sum (ADR-0031/0032).
        let func_id = self.codegen.func_ids.get("__reduce_all_sum_fwd")
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __reduce_all_sum_fwd not found".into()))?;
        let func_ref = self.import_func(func_id);
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

    fn call_aggregate_alloc(&mut self, size: i64) -> cranelift_codegen::ir::Value {
        let size_val = self.builder.ins().iconst(I64, size);
        let func_ref = self.import_func(self.codegen.rt_aggregate_alloc);
        let call = self.builder.ins().call(func_ref, &[size_val]);
        self.builder.inst_results(call).to_vec()[0]
    }

    fn call_aggregate_retain(&mut self, ptr: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_aggregate_retain);
        self.builder.ins().call(func_ref, &[ptr]);
    }

    fn call_aggregate_release(&mut self, ptr: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_aggregate_release);
        self.builder.ins().call(func_ref, &[ptr]);
    }

    /// M28: `List<T>` release — returns the refcount *before* the decrement, so
    /// callers can branch on whether this was the last reference (ADR-0034).
    fn call_list_release(&mut self, ptr: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        let func_ref = self.import_func(self.codegen.rt_list_release);
        let call = self.builder.ins().call(func_ref, &[ptr]);
        self.builder.inst_results(call).to_vec()[0]
    }

    // M14 tape helpers.

    fn call_tape_record_binop(
        &mut self,
        op_tag: i32,
        a: cranelift_codegen::ir::Value,
        b: cranelift_codegen::ir::Value,
        out: cranelift_codegen::ir::Value,
    ) {
        let tag_val = self.builder.ins().iconst(I32, op_tag as i64);
        let func_ref = self.import_func(self.codegen.rt_tape_record_binop);
        self.builder.ins().call(func_ref, &[tag_val, a, b, out]);
    }

    fn call_tape_record_unary(
        &mut self,
        op_tag: i32,
        x: cranelift_codegen::ir::Value,
        out: cranelift_codegen::ir::Value,
    ) {
        let tag_val = self.builder.ins().iconst(I32, op_tag as i64);
        let func_ref = self.import_func(self.codegen.rt_tape_record_unary);
        self.builder.ins().call(func_ref, &[tag_val, x, out]);
    }

    fn call_tape_register_leaf(&mut self, handle: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_tape_register_leaf);
        self.builder.ins().call(func_ref, &[handle]);
    }

    fn call_tape_pause(&mut self) {
        let func_ref = self.import_func(self.codegen.rt_tape_pause);
        self.builder.ins().call(func_ref, &[]);
    }

    fn call_tape_resume(&mut self) {
        let func_ref = self.import_func(self.codegen.rt_tape_resume);
        self.builder.ins().call(func_ref, &[]);
    }

    fn call_tape_get_grad(&mut self, handle: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        let func_ref = self.import_func(self.codegen.rt_tape_get_grad);
        let call = self.builder.ins().call(func_ref, &[handle]);
        self.builder.inst_results(call).to_vec()[0]
    }

    fn call_backward(&mut self, loss: cranelift_codegen::ir::Value) {
        let func_ref = self.import_func(self.codegen.rt_backward);
        self.builder.ins().call(func_ref, &[loss]);
    }

    fn call_tape_zero_grad(
        &mut self,
        handles_ptr: cranelift_codegen::ir::Value,
        count: cranelift_codegen::ir::Value,
    ) {
        let func_ref = self.import_func(self.codegen.rt_tape_zero_grad);
        self.builder.ins().call(func_ref, &[handles_ptr, count]);
    }

    // M16 helpers.

    /// Call __broadcast_{add,sub,mul,div}_fwd(a, b) -> out.
    /// kernel_name is the MSL kernel name like "malus_add".
    fn lower_broadcast_binop(
        &mut self,
        kernel_name: &str,
        a: cranelift_codegen::ir::Value,
        b: cranelift_codegen::ir::Value,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let fwd_name = match kernel_name {
            "malus_add" => "__broadcast_add_fwd",
            "malus_sub" => "__broadcast_sub_fwd",
            "malus_mul" => "__broadcast_mul_fwd",
            "malus_div" => "__broadcast_div_fwd",
            _ => return Err(CodegenError::UnsupportedExpr(format!("no broadcast fn for kernel '{kernel_name}'"))),
        };
        let func_id = self.codegen.func_ids.get(fwd_name)
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr(format!("stdlib {fwd_name} not found")))?;
        let func_ref = self.import_func(func_id);
        let call = self.builder.ins().call(func_ref, &[a, b]);
        Ok(self.builder.inst_results(call).to_vec()[0])
    }

    /// Lower an axis reduction (sum/mean/max/var with axis/keepdim args).
    /// args = [tensor, axis_expr, keepdim_expr] after sema normalization.
    fn lower_axis_reduction(
        &mut self,
        op_tag: i32,
        grad_tracked: bool,
        args: &[malus_sema::TypedExpr],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let x = self.lower_expr(&args[0])?;
        let axis = self.lower_expr(&args[1])?;
        let keepdim = self.lower_expr(&args[2])?;
        let fwd_name = match op_tag {
            OPTAG_REDUCE_SUM_AXIS  => "__reduce_sum_fwd",
            OPTAG_REDUCE_MEAN_AXIS => "__reduce_mean_fwd",
            OPTAG_REDUCE_MAX_AXIS  => "__reduce_max_fwd",
            OPTAG_REDUCE_VAR_AXIS  => "__reduce_var_fwd",
            _ => unreachable!("invalid axis reduction op_tag {}", op_tag),
        };
        let func_id = self.codegen.func_ids.get(fwd_name)
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr(format!("stdlib {fwd_name} not found")))?;
        let func_ref = self.import_func(func_id);
        let call = self.builder.ins().call(func_ref, &[x, axis, keepdim]);
        let out = self.builder.inst_results(call).to_vec()[0];
        if grad_tracked {
            self.call_tape_record_reduce(op_tag, x, out, axis, keepdim);
        }
        Ok(out)
    }

    fn call_tape_record_reduce(
        &mut self,
        op_tag: i32,
        x: cranelift_codegen::ir::Value,
        out: cranelift_codegen::ir::Value,
        axis: cranelift_codegen::ir::Value,   // I64 (from lowered Scalar(I64) literal)
        keepdim: cranelift_codegen::ir::Value, // I64
    ) {
        let tag_val = self.builder.ins().iconst(I32, op_tag as i64);
        let func_ref = self.import_func(self.codegen.rt_tape_record_reduce);
        // sig: (I32, I64, I64, I64, I64)
        self.builder.ins().call(func_ref, &[tag_val, x, out, axis, keepdim]);
    }

    // M17 helpers.

    fn call_tape_record_perm(
        &mut self,
        op_tag: i32,
        x: cranelift_codegen::ir::Value,
        out: cranelift_codegen::ir::Value,
        dims_ptr: cranelift_codegen::ir::Value, // ptr (*const usize)
        ndims: cranelift_codegen::ir::Value,    // ptr (usize)
    ) {
        let tag_val = self.builder.ins().iconst(I32, op_tag as i64);
        let func_ref = self.import_func(self.codegen.rt_tape_record_perm);
        self.builder.ins().call(func_ref, &[tag_val, x, out, dims_ptr, ndims]);
    }

    // M18 helpers.

    fn call_tape_record_layernorm(
        &mut self,
        op_tag: i32,
        x:     cranelift_codegen::ir::Value,
        out:   cranelift_codegen::ir::Value,
        var_h: cranelift_codegen::ir::Value,
        axis:  cranelift_codegen::ir::Value,
    ) {
        let tag_val = self.builder.ins().iconst(I32, op_tag as i64);
        let func_ref = self.import_func(self.codegen.rt_tape_record_layernorm);
        // sig: (I32, I64, I64, I64, I64)
        self.builder.ins().call(func_ref, &[tag_val, x, out, var_h, axis]);
    }

    fn call_tape_record_cross_entropy(
        &mut self,
        op_tag:  i32,
        logits:  cranelift_codegen::ir::Value,
        out:     cranelift_codegen::ir::Value,
        sm_h:    cranelift_codegen::ir::Value,
        targets: cranelift_codegen::ir::Value,
    ) {
        let tag_val = self.builder.ins().iconst(I32, op_tag as i64);
        let func_ref = self.import_func(self.codegen.rt_tape_record_cross_entropy);
        // sig: (I32, I64, I64, I64, I64)
        self.builder.ins().call(func_ref, &[tag_val, logits, out, sm_h, targets]);
    }

    /// softmax(t, axis=N) — axis-only builtin.
    /// grad_tracked: whether the result is grad-tracked (M27 grad_inference.rs;
    /// propagates from arg0's own grad-tracked status; ADR-0030).
    fn lower_softmax(
        &mut self,
        grad_tracked: bool,
        args: &[malus_sema::TypedExpr],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let x    = self.lower_expr(&args[0])?;
        let axis = self.lower_expr(&args[1])?;
        let func_id = self.codegen.func_ids.get("__softmax_fwd")
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __softmax_fwd not found".into()))?;
        let func_ref = self.import_func(func_id);
        let call = self.builder.ins().call(func_ref, &[x, axis]);
        let out  = self.builder.inst_results(call).to_vec()[0];
        if grad_tracked {
            let keepdim_zero = self.builder.ins().iconst(I64, 0);
            self.call_tape_record_reduce(OPTAG_SOFTMAX, x, out, axis, keepdim_zero);
        }
        Ok(out)
    }

    /// layernorm(t, axis=N) — axis-only builtin.
    fn lower_layernorm(
        &mut self,
        grad_tracked: bool,
        args: &[malus_sema::TypedExpr],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let x    = self.lower_expr(&args[0])?;
        let axis = self.lower_expr(&args[1])?;
        let func_id = self.codegen.func_ids.get("__layernorm_fwd")
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __layernorm_fwd not found".into()))?;
        let func_ref = self.import_func(func_id);
        let call      = self.builder.ins().call(func_ref, &[x, axis]);
        let tuple_ptr = self.builder.inst_results(call).to_vec()[0];
        let mem      = cranelift_codegen::ir::MemFlags::trusted();
        let normed   = self.builder.ins().load(I64, mem, tuple_ptr, 8);
        let variance = self.builder.ins().load(I64, mem, tuple_ptr, 16);
        self.call_aggregate_release(tuple_ptr);
        if grad_tracked {
            self.call_tape_record_layernorm(OPTAG_LAYERNORM, x, normed, variance, axis);
        }
        Ok(normed)
    }

    /// embedding(weight: Tensor<f32>, indices: Tensor<i32|i64>) -> Tensor<f32>
    fn lower_embedding(
        &mut self,
        args: &[malus_sema::TypedExpr],
        grad_tracked: bool,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let weight  = self.lower_expr(&args[0])?;
        let indices = self.lower_expr(&args[1])?;
        let func_id = self.codegen.func_ids.get("__embedding_fwd")
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __embedding_fwd not found".into()))?;
        let func_ref = self.import_func(func_id);
        let call = self.builder.ins().call(func_ref, &[weight, indices]);
        let out  = self.builder.inst_results(call).to_vec()[0];
        if grad_tracked {
            let tag  = self.builder.ins().iconst(I32, OPTAG_EMBEDDING as i64);
            let rec  = self.import_func(self.codegen.rt_tape_record_embedding);
            self.builder.ins().call(rec, &[tag, weight, indices, out]);
        }
        Ok(out)
    }

    /// cross_entropy(logits: Tensor<f32>, targets: Tensor<i32|i64>) -> Tensor<f32>
    fn lower_cross_entropy(
        &mut self,
        args: &[malus_sema::TypedExpr],
        grad_tracked: bool,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let logits  = self.lower_expr(&args[0])?;
        let targets = self.lower_expr(&args[1])?;
        let func_id = self.codegen.func_ids.get("__cross_entropy_fwd")
            .copied()
            .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __cross_entropy_fwd not found".into()))?;
        let func_ref  = self.import_func(func_id);
        let call      = self.builder.ins().call(func_ref, &[logits, targets]);
        let tuple_ptr = self.builder.inst_results(call).to_vec()[0];
        let mem   = cranelift_codegen::ir::MemFlags::trusted();
        let loss  = self.builder.ins().load(I64, mem, tuple_ptr, 8);
        let probs = self.builder.ins().load(I64, mem, tuple_ptr, 16);
        self.call_aggregate_release(tuple_ptr);
        if grad_tracked {
            self.call_tape_record_cross_entropy(OPTAG_CROSS_ENTROPY, logits, loss, probs, targets);
        } else {
            // Not grad-tracked: `probs` is a backward-only VJP aid with no other
            // owner (tape_record_cross_entropy normally retains it) — free it here
            // so it doesn't leak.
            self.call_runtime_free(probs);
        }
        Ok(loss)
    }

    /// causal_mask(T: i64) -> Tensor<f32>
    fn lower_causal_mask(
        &mut self,
        args: &[malus_sema::TypedExpr],
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let t = self.lower_expr(&args[0])?;
        let func_ref = self.import_func(self.codegen.rt_tensor_causal_mask);
        let call = self.builder.ins().call(func_ref, &[t]);
        Ok(self.builder.inst_results(call).to_vec()[0])
    }

    /// Lower reshape(t, d0..dn), transpose(t[, i, j]), or permute(t, p0..pn).
    /// args[0] = tensor; args[1..] = dim args (may be empty for no-arg transpose).
    fn lower_shape_op(
        &mut self,
        callee: &str,
        args: &[malus_sema::TypedExpr],
        grad_tracked: bool,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        let handle = self.lower_expr(&args[0])?;
        let dim_args = &args[1..];
        let n = dim_args.len() as u32;

        // Stdlib fast path: 2-D transpose (no dim args).
        if callee == "transpose" && n == 0 {
            let func_id = self.codegen.func_ids.get("__transpose_2d_fwd")
                .copied()
                .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __transpose_2d_fwd not found".into()))?;
            let func_ref = self.import_func(func_id);
            let call = self.builder.ins().call(func_ref, &[handle]);
            let out  = self.builder.inst_results(call).to_vec()[0];
            if grad_tracked {
                let null_ptr  = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
                let zero      = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
                self.call_tape_record_perm(OPTAG_TRANSPOSE, handle, out, null_ptr, zero);
            }
            return Ok(out);
        }

        // Stdlib fast path: 3-D permute.
        if callee == "permute" && n == 3 {
            let p0 = self.lower_expr(&dim_args[0])?;
            let p1 = self.lower_expr(&dim_args[1])?;
            let p2 = self.lower_expr(&dim_args[2])?;
            let func_id = self.codegen.func_ids.get("__permute_3d_fwd")
                .copied()
                .ok_or_else(|| CodegenError::UnsupportedExpr("stdlib __permute_3d_fwd not found".into()))?;
            let func_ref = self.import_func(func_id);
            let call = self.builder.ins().call(func_ref, &[handle, p0, p1, p2]);
            let out  = self.builder.inst_results(call).to_vec()[0];
            if grad_tracked {
                let slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 24, 3));
                self.builder.ins().stack_store(p0, slot, 0);
                self.builder.ins().stack_store(p1, slot, 8);
                self.builder.ins().stack_store(p2, slot, 16);
                let dims_ptr  = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
                let ndims_val = self.builder.ins().iconst(self.codegen.ptr_type(), 3);
                self.call_tape_record_perm(OPTAG_TRANSPOSE, handle, out, dims_ptr, ndims_val);
            }
            return Ok(out);
        }

        // Build stack slot for variadic dim args (identical idiom to lower_zeros_ones).
        let (dims_ptr, ndims_val) = if n == 0 {
            // No dim args (e.g. transpose(t) — 0-arg 2-D reverse).
            // Pass null ptr + 0 count; runtime normalize_perm handles this.
            let null_ptr = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
            let zero     = self.builder.ins().iconst(self.codegen.ptr_type(), 0);
            (null_ptr, zero)
        } else {
            let slot = self.builder.create_sized_stack_slot(
                cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    n * 8,
                    3,
                )
            );
            for (i, arg) in dim_args.iter().enumerate() {
                let val = self.lower_expr(arg)?;
                self.builder.ins().stack_store(val, slot, (i as i32) * 8);
            }
            let ptr = self.builder.ins().stack_addr(self.codegen.ptr_type(), slot, 0);
            let cnt = self.builder.ins().iconst(self.codegen.ptr_type(), n as i64);
            (ptr, cnt)
        };

        let out = if callee == "reshape" {
            let func_ref = self.import_func(self.codegen.rt_tensor_reshape);
            let call = self.builder.ins().call(func_ref, &[handle, dims_ptr, ndims_val]);
            self.builder.inst_results(call).to_vec()[0]
        } else {
            // Fallback: transpose(t, i, j) or other permute forms — use rt_tensor_permute.
            let func_ref = self.import_func(self.codegen.rt_tensor_permute);
            let call = self.builder.ins().call(func_ref, &[handle, dims_ptr, ndims_val]);
            self.builder.inst_results(call).to_vec()[0]
        };

        if grad_tracked {
            if callee == "reshape" {
                self.call_tape_record_unary(OPTAG_RESHAPE, handle, out);
            } else {
                // transpose and permute both record as OPTAG_TRANSPOSE with the raw dims.
                self.call_tape_record_perm(OPTAG_TRANSPOSE, handle, out, dims_ptr, ndims_val);
            }
        }
        Ok(out)
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
    // M13 aggregate ARC — local shims, not in RuntimeSymbols (ADR-0008).
    jit_builder.symbol("aggregate_alloc",        aggregate_alloc                as *const u8);
    jit_builder.symbol("aggregate_retain",       aggregate_retain               as *const u8);
    jit_builder.symbol("aggregate_release",      aggregate_release              as *const u8);
    // M28 List<T> release (returns previous refcount — ADR-0034).
    jit_builder.symbol("list_release",           list_release                   as *const u8);
    // M14 tape ABI — from RuntimeSymbols (live in malus-runtime).
    jit_builder.symbol("tape_record_binop",      symbols.tape_record_binop      as *const u8);
    jit_builder.symbol("tape_record_unary",      symbols.tape_record_unary      as *const u8);
    jit_builder.symbol("tape_register_leaf",     symbols.tape_register_leaf     as *const u8);
    jit_builder.symbol("tape_pause",             symbols.tape_pause             as *const u8);
    jit_builder.symbol("tape_resume",            symbols.tape_resume            as *const u8);
    jit_builder.symbol("tape_clear",             symbols.tape_clear             as *const u8);
    jit_builder.symbol("tape_get_grad",          symbols.tape_get_grad          as *const u8);
    jit_builder.symbol("backward",               symbols.backward               as *const u8);
    // M15 tape ABI.
    jit_builder.symbol("tape_zero_grad",         symbols.tape_zero_grad         as *const u8);
    jit_builder.symbol("tape_record_reduce",     symbols.tape_record_reduce     as *const u8);
    // M17 shapes + batched matmul.
    jit_builder.symbol("tensor_reshape",         symbols.tensor_reshape         as *const u8);
    jit_builder.symbol("tensor_permute",         symbols.tensor_permute         as *const u8);
    jit_builder.symbol("tape_record_perm",       symbols.tape_record_perm       as *const u8);
    // M18 transformer stdlib.
    jit_builder.symbol("tensor_causal_mask",        symbols.tensor_causal_mask        as *const u8);
    jit_builder.symbol("tape_record_layernorm",     symbols.tape_record_layernorm     as *const u8);
    jit_builder.symbol("tape_record_cross_entropy", symbols.tape_record_cross_entropy as *const u8);
    // M19 randn.
    jit_builder.symbol("tensor_randn",              symbols.tensor_randn              as *const u8);
    jit_builder.symbol("tape_record_embedding",     symbols.tape_record_embedding     as *const u8);
    // M20 scalar power operator.
    jit_builder.symbol("malus_powf",               malus_powf                        as *const u8);
    // M22 string I/O.
    jit_builder.symbol("malus_str_box",            symbols.malus_str_box             as *const u8);
    jit_builder.symbol("malus_read_file",          symbols.malus_read_file           as *const u8);
    jit_builder.symbol("malus_str_len",            symbols.malus_str_len             as *const u8);
    jit_builder.symbol("malus_str_char_at",        symbols.malus_str_char_at         as *const u8);
    jit_builder.symbol("malus_str_from_char",      symbols.malus_str_from_char       as *const u8);
    jit_builder.symbol("print_str",                print_str                         as *const u8);
    // M22 rand_uniform.
    jit_builder.symbol("malus_rand_uniform",        symbols.malus_rand_uniform        as *const u8);
    jit_builder.symbol("malus_record_diff",         symbols.malus_record_diff         as *const u8);
    // M22 Buffer<i32>.
    jit_builder.symbol("malus_buffer_i32",         symbols.malus_buffer_i32          as *const u8);
    jit_builder.symbol("malus_buffer_get_i32",     symbols.malus_buffer_get_i32      as *const u8);
    jit_builder.symbol("malus_buffer_set_i32",     symbols.malus_buffer_set_i32      as *const u8);
    jit_builder.symbol("malus_buffer_free",        symbols.malus_buffer_free         as *const u8);
    jit_builder.symbol("malus_buffer_freeze_i32",  symbols.malus_buffer_freeze_i32   as *const u8);
    // M22 rand_int + tensor_get_f32.
    jit_builder.symbol("malus_rand_int",            symbols.malus_rand_int            as *const u8);
    jit_builder.symbol("malus_tensor_get_f32",      symbols.malus_tensor_get_f32      as *const u8);
    // M25 metadata accessors + kernel_dispatch_v2.
    jit_builder.symbol("tensor_ndim",              symbols.tensor_ndim               as *const u8);
    jit_builder.symbol("tensor_dim",               symbols.tensor_dim                as *const u8);
    jit_builder.symbol("kernel_dispatch_v2",       symbols.kernel_dispatch_v2        as *const u8);

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
    // aggregate_alloc(size: i64) -> i64  (returns *mut u8 cast to i64)
    let sig_agg_alloc = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // aggregate_retain/release(ptr: i64) -> ()
    let sig_agg_rc = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s
    };
    let rt_aggregate_alloc = module.declare_function("aggregate_alloc", Linkage::Import, &sig_agg_alloc)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_aggregate_retain = module.declare_function("aggregate_retain", Linkage::Import, &sig_agg_rc)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_aggregate_release = module.declare_function("aggregate_release", Linkage::Import, &sig_agg_rc)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // M28: list_release(ptr: i64) -> i64  (previous refcount — ADR-0034)
    let sig_list_release = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    let rt_list_release = module.declare_function("list_release", Linkage::Import, &sig_list_release)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M14 tape ABI signatures.
    // tape_record_binop(op_tag: i32, a: i64, b: i64, out: i64)
    let sig_tape_binop = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I32));
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s
    };
    // tape_record_unary(op_tag: i32, x: i64, out: i64)
    let sig_tape_unary = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I32));
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s
    };
    let rt_tape_record_binop = module.declare_function("tape_record_binop", Linkage::Import, &sig_tape_binop)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tape_record_unary = module.declare_function("tape_record_unary", Linkage::Import, &sig_tape_unary)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // tape_register_leaf / backward / tape_clear use sig_free: (i64) -> ()
    let rt_tape_register_leaf = module.declare_function("tape_register_leaf", Linkage::Import, &sig_free)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_backward = module.declare_function("backward", Linkage::Import, &sig_free)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tape_clear = module.declare_function("tape_clear", Linkage::Import, &sig_barrier)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // tape_pause / tape_resume: () -> ()
    let rt_tape_pause = module.declare_function("tape_pause", Linkage::Import, &sig_barrier)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tape_resume = module.declare_function("tape_resume", Linkage::Import, &sig_barrier)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // tape_get_grad(handle: i64) -> i64  (same as sig_unary_tensor_ret)
    let rt_tape_get_grad = module.declare_function("tape_get_grad", Linkage::Import, &sig_unary_tensor_ret)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // tape_zero_grad(handles: *const i64, count: usize) -> ()
    let sig_tape_zero_grad = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(ptr));  // handles_ptr
        s.params.push(AbiParam::new(ptr));  // count
        s
    };
    let rt_tape_zero_grad = module.declare_function("tape_zero_grad", Linkage::Import, &sig_tape_zero_grad)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M16: tape_record_reduce(op: i32, x: i64, out: i64, axis: i64, keepdim: i64)
    let sig_tape_record_reduce = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I32)); // op_tag
        s.params.push(AbiParam::new(I64)); // x
        s.params.push(AbiParam::new(I64)); // out
        s.params.push(AbiParam::new(I64)); // axis
        s.params.push(AbiParam::new(I64)); // keepdim
        s
    };
    let rt_tape_record_reduce = module.declare_function("tape_record_reduce", Linkage::Import, &sig_tape_record_reduce)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M17: tensor_reshape / tensor_permute (handle, dims_ptr, ndims) -> i64
    //      tape_record_perm(op: i32, x: i64, out: i64, dims_ptr: *const usize, ndims: usize)
    let sig_shape_op = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64)); // handle
        s.params.push(AbiParam::new(ptr)); // dims_ptr (*const usize)
        s.params.push(AbiParam::new(ptr)); // ndims (usize)
        s.returns.push(AbiParam::new(I64));
        s
    };
    let rt_tensor_reshape = module.declare_function("tensor_reshape", Linkage::Import, &sig_shape_op)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tensor_permute = module.declare_function("tensor_permute", Linkage::Import, &sig_shape_op)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let sig_tape_record_perm = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I32)); // op_tag
        s.params.push(AbiParam::new(I64)); // x
        s.params.push(AbiParam::new(I64)); // out
        s.params.push(AbiParam::new(ptr)); // dims_ptr (*const usize)
        s.params.push(AbiParam::new(ptr)); // ndims (usize)
        s
    };
    let rt_tape_record_perm = module.declare_function("tape_record_perm", Linkage::Import, &sig_tape_record_perm)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M18: tensor_causal_mask(t_size: i64) -> i64
    let sig_unary_cpu = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    let rt_tensor_causal_mask = module.declare_function("tensor_causal_mask", Linkage::Import, &sig_unary_cpu)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // M18: tape_record_layernorm(op: i32, x: i64, out: i64, var_h: i64, axis: i64)
    //      tape_record_cross_entropy(op: i32, logits: i64, out: i64, sm_h: i64, targets: i64)
    let sig_tape_saved_axis = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I32)); // op_tag
        s.params.push(AbiParam::new(I64)); // x / logits
        s.params.push(AbiParam::new(I64)); // out
        s.params.push(AbiParam::new(I64)); // var_h / sm_h
        s.params.push(AbiParam::new(I64)); // axis / targets
        s
    };
    let rt_tape_record_layernorm = module.declare_function("tape_record_layernorm", Linkage::Import, &sig_tape_saved_axis)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_tape_record_cross_entropy = module.declare_function("tape_record_cross_entropy", Linkage::Import, &sig_tape_saved_axis)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M19: tensor_randn(shape_ptr, ndims) -> i64  (same shape as sig_alloc_shape)
    let rt_tensor_randn = module.declare_function("tensor_randn", Linkage::Import, &sig_alloc_shape)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // M19: tape_record_embedding(op: i32, weight: i64, indices: i64, out: i64)  (same as sig_tape_binop)
    let rt_tape_record_embedding = module.declare_function("tape_record_embedding", Linkage::Import, &sig_tape_binop)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M20: malus_powf(base: f32, exp: f32) -> f32
    let sig_powf = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(F32));
        s.params.push(AbiParam::new(F32));
        s.returns.push(AbiParam::new(F32));
        s
    };
    let rt_powf = module.declare_function("malus_powf", Linkage::Import, &sig_powf)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M22: string I/O signatures.
    // malus_str_box(ptr: *const u8, len: usize) -> i64
    let sig_str_box = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(ptr)); // ptr (*const u8)
        s.params.push(AbiParam::new(ptr)); // len (usize)
        s.returns.push(AbiParam::new(I64));
        s
    };
    // malus_read_file(path_handle: i64) -> i64
    let sig_read_file = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // malus_str_len(handle: i64) -> i64
    let sig_str_len = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // malus_str_char_at(handle: i64, idx: i64) -> i64
    let sig_str_char_at = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // malus_str_from_char(c: i64) -> i64
    let sig_str_from_char = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // print_str(handle: i64) -> ()  (same as sig_free)
    let rt_malus_str_box = module.declare_function("malus_str_box", Linkage::Import, &sig_str_box)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_read_file = module.declare_function("malus_read_file", Linkage::Import, &sig_read_file)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_str_len = module.declare_function("malus_str_len", Linkage::Import, &sig_str_len)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_str_char_at = module.declare_function("malus_str_char_at", Linkage::Import, &sig_str_char_at)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_str_from_char = module.declare_function("malus_str_from_char", Linkage::Import, &sig_str_from_char)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_print_str = module.declare_function("print_str", Linkage::Import, &sig_free)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M22 rand_uniform() -> f32.
    let sig_rand_uniform = {
        let mut s = Signature::new(call_conv);
        s.returns.push(AbiParam::new(F32));
        s
    };
    let rt_malus_rand_uniform = module.declare_function("malus_rand_uniform", Linkage::Import, &sig_rand_uniform)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M26 record_diff(value: f32) -> Unit.
    let sig_record_diff = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(F32));
        s
    };
    let rt_malus_record_diff = module.declare_function("malus_record_diff", Linkage::Import, &sig_record_diff)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M22 Buffer<i32> signatures.
    // malus_buffer_i32(len: i64) -> i64
    let sig_buffer_i32 = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // malus_buffer_get_i32(handle: i64, idx: i64) -> i64
    let sig_buffer_get = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(I64));
        s
    };
    // malus_buffer_set_i32(handle: i64, idx: i64, val: i64) -> ()
    let sig_buffer_set = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s
    };
    // malus_buffer_free(handle: i64) — same as sig_free.
    // malus_buffer_freeze_i32(handle: i64) -> i64 — same as sig_buffer_i32.
    let rt_malus_buffer_i32 = module.declare_function("malus_buffer_i32", Linkage::Import, &sig_buffer_i32)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_buffer_get_i32 = module.declare_function("malus_buffer_get_i32", Linkage::Import, &sig_buffer_get)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_buffer_set_i32 = module.declare_function("malus_buffer_set_i32", Linkage::Import, &sig_buffer_set)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_buffer_free = module.declare_function("malus_buffer_free", Linkage::Import, &sig_free)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    let rt_malus_buffer_freeze_i32 = module.declare_function("malus_buffer_freeze_i32", Linkage::Import, &sig_buffer_i32)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M22 rand_int(n: i64) -> i64
    let rt_malus_rand_int = module.declare_function("malus_rand_int", Linkage::Import, &sig_buffer_i32)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // M22 malus_tensor_get_f32(handle: i64, idx: i64) -> f32
    let sig_tensor_get_f32 = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64));
        s.params.push(AbiParam::new(I64));
        s.returns.push(AbiParam::new(F32));
        s
    };
    let rt_malus_tensor_get_f32 = module.declare_function("malus_tensor_get_f32", Linkage::Import, &sig_tensor_get_f32)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M25: tensor_ndim(handle: i64) -> i64  (same as sig_unary_tensor_ret)
    let rt_tensor_ndim = module.declare_function("tensor_ndim", Linkage::Import, &sig_unary_tensor_ret)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // M25: tensor_dim(handle: i64, i: i64) -> i64
    let sig_tensor_dim = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64)); // handle
        s.params.push(AbiParam::new(I64)); // axis index
        s.returns.push(AbiParam::new(I64));
        s
    };
    let rt_tensor_dim = module.declare_function("tensor_dim", Linkage::Import, &sig_tensor_dim)
        .map_err(|e| CodegenError::JitError(e.to_string()))?;
    // M25: kernel_dispatch_v2(kernel_id: u64→i64, handles: ptr, handle_count: ptr,
    //       grid_dims: ptr, tg_dims: ptr, out_shape: ptr, out_ndim: ptr,
    //       out_dtype_tag: i32, uniforms: ptr, uniforms_bytes: ptr) -> i64
    let sig_dispatch_v2 = {
        let mut s = Signature::new(call_conv);
        s.params.push(AbiParam::new(I64)); // kernel_id (u64 fits in i64)
        s.params.push(AbiParam::new(ptr)); // handles *const i64
        s.params.push(AbiParam::new(ptr)); // handle_count usize
        s.params.push(AbiParam::new(ptr)); // grid_dims *const usize
        s.params.push(AbiParam::new(ptr)); // tg_dims *const usize
        s.params.push(AbiParam::new(ptr)); // out_shape *const usize
        s.params.push(AbiParam::new(ptr)); // out_ndim usize
        s.params.push(AbiParam::new(I32)); // out_dtype_tag i32
        s.params.push(AbiParam::new(ptr)); // uniforms *const c_void
        s.params.push(AbiParam::new(ptr)); // uniforms_bytes usize
        s.returns.push(AbiParam::new(I64));
        s
    };
    let rt_kernel_dispatch_v2 = module.declare_function("kernel_dispatch_v2", Linkage::Import, &sig_dispatch_v2)
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
        rt_tensor_len,
        rt_print_cstr,
        rt_print_f32,
        rt_print_i64,
        rt_print_bool,
        rt_tensor_retain,
        rt_tensor_release,
        rt_malloc,
        rt_heap_free,
        rt_aggregate_alloc,
        rt_aggregate_retain,
        rt_aggregate_release,
        rt_list_release,
        rt_tape_record_binop,
        rt_tape_record_unary,
        rt_tape_register_leaf,
        rt_tape_pause,
        rt_tape_resume,
        rt_tape_clear,
        rt_tape_get_grad,
        rt_backward,
        rt_tape_zero_grad,
        rt_tape_record_reduce,
        rt_tensor_reshape,
        rt_tensor_permute,
        rt_tape_record_perm,
        rt_tensor_causal_mask,
        rt_tape_record_layernorm,
        rt_tape_record_cross_entropy,
        rt_tensor_randn,
        rt_tape_record_embedding,
        rt_powf,
        rt_malus_str_box,
        rt_malus_read_file,
        rt_malus_str_len,
        rt_malus_str_char_at,
        rt_malus_str_from_char,
        rt_print_str,
        rt_malus_rand_uniform,
        rt_malus_buffer_i32,
        rt_malus_buffer_get_i32,
        rt_malus_buffer_set_i32,
        rt_malus_buffer_free,
        rt_malus_buffer_freeze_i32,
        rt_malus_rand_int,
        rt_malus_tensor_get_f32,
        rt_tensor_ndim,
        rt_tensor_dim,
        rt_kernel_dispatch_v2,
        rt_malus_record_diff,
    };

    // Second pass: compile each fn body.
    let fns: Vec<TypedFn> = program.fns.clone();
    for typed_fn in &fns {
        cg.compile_fn(typed_fn)?;
    }

    cg.module.finalize_definitions()
        .map_err(|e| CodegenError::JitError(e.to_string()))?;

    // M26 (ADR-0032): register every backward kernel's finalized JIT pointer
    // before main runs, so tape.rs::backward can dispatch to it. Every name
    // in BWD_SLOT_FNS is a malus-stdlib item, present in func_ids regardless
    // of call-graph reachability (the first compilation pass declares every
    // fn in the combined stdlib+user program) — a missing entry means the
    // stdlib file was renamed/dropped, a real wiring bug worth panicking on.
    for &(slot, name) in BWD_SLOT_FNS {
        let func_id = *cg.func_ids.get(name).unwrap_or_else(|| {
            panic!("malus: backward kernel fn '{name}' (slot {slot}) not found in compiled program")
        });
        let ptr = cg.module.get_finalized_function(func_id);
        (symbols.tape_register_backward_fn)(slot, ptr as usize);
    }

    let main_id = cg.func_ids["main"];
    let main_ptr = cg.module.get_finalized_function(main_id);
    let main_fn = unsafe { std::mem::transmute::<_, fn()>(main_ptr) };
    main_fn();

    Ok(())
}
