use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use crate::metal::{
    invert_perm, normalize_perm, reshape_to, tensor_alloc_ones_gpu,
    tensor_alloc_zeros_gpu, tensor_retain, tensor_release, TensorBuffer,
};

// ── OpTag ─────────────────────────────────────────────────────────────────────
//
// Numeric tags are emitted by codegen-cpu and dispatched here.  The two-sided
// mapping is drift-tested in `tests.rs`.

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpTag {
    Matmul    = 0,
    Add       = 1,
    Sub       = 2,
    Mul       = 3,
    Div       = 4,
    Sigmoid   = 5,
    Relu      = 6,
    Tanh      = 7,
    Exp       = 8,
    Log       = 9,
    Sqrt      = 10,
    Abs       = 11,
    Sum              = 12,
    Transpose        = 13,
    Neg              = 14,
    // M16 axis reductions
    ReduceSumAxis    = 15,
    ReduceMeanAxis   = 16,
    ReduceMaxAxis    = 17,
    ReduceVarAxis    = 18,
    // M17 shapes + batched matmul
    Reshape          = 19,
    // M18 transformer stdlib
    Softmax          = 20,
    Layernorm        = 21,
    Gelu             = 22,
    CrossEntropy     = 23,
    // M19 embeddings
    Embedding        = 24,
}

impl OpTag {
    pub fn from_tag(tag: i32) -> Self {
        match tag {
            0  => OpTag::Matmul,
            1  => OpTag::Add,
            2  => OpTag::Sub,
            3  => OpTag::Mul,
            4  => OpTag::Div,
            5  => OpTag::Sigmoid,
            6  => OpTag::Relu,
            7  => OpTag::Tanh,
            8  => OpTag::Exp,
            9  => OpTag::Log,
            10 => OpTag::Sqrt,
            11 => OpTag::Abs,
            12 => OpTag::Sum,
            13 => OpTag::Transpose,
            14 => OpTag::Neg,
            15 => OpTag::ReduceSumAxis,
            16 => OpTag::ReduceMeanAxis,
            17 => OpTag::ReduceMaxAxis,
            18 => OpTag::ReduceVarAxis,
            19 => OpTag::Reshape,
            20 => OpTag::Softmax,
            21 => OpTag::Layernorm,
            22 => OpTag::Gelu,
            23 => OpTag::CrossEntropy,
            24 => OpTag::Embedding,
            _  => panic!("malus: unknown op tag {tag}"),
        }
    }
}

// ── BwdSlot ───────────────────────────────────────────────────────────────────
//
// M26 (ADR-0032): each VJP is a malus kernel + host fn living in
// malus-stdlib/stdlib/backward/.  codegen-cpu resolves each fn's finalized
// JIT pointer after compilation and registers it here by numeric slot —
// mirroring OpTag's codegen-cpu/malus-runtime dual-definition pattern
// (drift-tested in tests.rs).  `backward` below transmutes the registered
// pointer to the slot's known signature and calls it directly; no
// `bwd_kernel_id` on `TapeNode` is needed because kernel-id resolution
// happens inside the malus host fn itself, exactly like the forward path.

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BwdSlot {
    AddBwdA           = 0,
    AddBwdB           = 1,
    SubBwdA           = 2,
    SubBwdB           = 3,
    MulBwdA           = 4,
    MulBwdB           = 5,
    DivBwdA           = 6,
    DivBwdB           = 7,
    SigmoidBwd        = 8,
    ReluBwd           = 9,
    TanhBwd           = 10,
    ExpBwd            = 11,
    LogBwd            = 12,
    SqrtBwd           = 13,
    AbsBwd            = 14,
    NegBwd            = 15,
    SumBwd            = 16,
    // M33: one rank-generic slot holds the *forward* N-D permute host fn
    // (__permute_nd_fwd); the Transpose VJP calls it with the inverse perm.
    // Same forward-fn-as-backward-slot convention as ExpBwd/NegBwd/GradAcc.
    PermuteNdFwd      = 17,
    ReduceSumAxisBwd  = 18,
    ReduceMeanAxisBwd = 19,
    ReduceMaxAxisBwd  = 20,
    ReduceVarAxisBwd  = 21,
    SoftmaxBwd        = 22,
    LayernormBwd      = 23,
    GeluBwd           = 24,
    CrossEntropyBwd   = 25,
    EmbeddingBwd      = 26,
    MatmulBwdA        = 27,
    MatmulBwdB        = 28,
    GradAcc           = 29,
}

pub const N_BWD_SLOTS: usize = 30;

thread_local! {
    static BWD_SLOTS: RefCell<[usize; N_BWD_SLOTS]> = const { RefCell::new([0; N_BWD_SLOTS]) };
}

/// Register a backward kernel's finalized JIT function pointer by slot.
/// Called by codegen-cpu's `compile_and_run` after `finalize_definitions()`
/// (production), and by the `cpu_fallback` self-registration path (tests).
/// `extern "C"` for ABI consistency with the rest of `RuntimeSymbols`, but
/// it is always called directly from Rust on both sides — never injected
/// into the JIT, never called by JIT'd code.
#[no_mangle]
pub extern "C" fn tape_register_backward_fn(slot: i32, func_ptr: usize) {
    BWD_SLOTS.with(|s| {
        s.borrow_mut()[slot as usize] = func_ptr;
    });
}

pub(crate) fn bwd_slot(slot: BwdSlot) -> usize {
    let ptr = BWD_SLOTS.with(|s| s.borrow()[slot as usize]);
    if ptr != 0 {
        return ptr;
    }

    // Unregistered: compile_and_run always registers every slot in one
    // batch before backward() can ever run, so reaching here means this
    // call never went through that production wiring — malus-runtime's
    // own isolated tape tests calling backward() directly. Self-register
    // the cpu_fallback closures (once) and retry. Checking the slot value
    // first (not a separate "have I registered" flag) is load-bearing: it
    // guarantees this path never fires after production registration,
    // even on the very first backward() call in this thread.
    #[cfg(feature = "cpu_fallback")]
    {
        cpu_fallback::ensure_registered();
        let ptr = BWD_SLOTS.with(|s| s.borrow()[slot as usize]);
        assert!(ptr != 0, "malus: backward kernel slot {slot:?} not registered even after cpu_fallback self-registration");
        return ptr;
    }
    #[cfg(not(feature = "cpu_fallback"))]
    panic!(
        "malus: backward kernel slot {slot:?} not registered — compile_and_run must register \
         every malus-stdlib backward fn before calling backward()"
    );
}

// ── TapeNode ─────────────────────────────────────────────────────────────────
//
// For binops, saved = [a, b] (both retained).
// For unary ops, saved = [x] (input retained); node.output is retained and
// used directly in VJPs that need the forward output (sigmoid, tanh, exp, sqrt).
// For axis reductions, saved = [x], meta = [axis, keepdim] (non-handle scalars).
// tape_clear() releases every handle in saved + output; meta is not released.

struct TapeNode {
    op:     OpTag,
    saved:  Vec<i64>,
    output: i64,
    meta:   Vec<i64>,
}

// ── Thread-local tape state ───────────────────────────────────────────────────

thread_local! {
    static RECORDING: Cell<bool>                   = const { Cell::new(true) };
    static NODES:     RefCell<Vec<TapeNode>>        = RefCell::new(Vec::new());
    // LEAVES: handles registered by variable(); persists across backward calls.
    static LEAVES:    RefCell<HashSet<i64>>         = RefCell::new(HashSet::new());
    // LEAF_GRAD: persists across backward calls (accumulate semantics).
    // Cleared by zero_grad (M15) or tape_reset (tests/reset).
    static LEAF_GRAD: RefCell<HashMap<i64, i64>>    = RefCell::new(HashMap::new());
}

// ── Public tape ABI (extern "C" for JIT injection) ───────────────────────────

/// Record a binary op onto the tape.  Retains a, b, and out so they survive
/// until backward() runs.  No-op when recording is paused.
#[no_mangle]
pub extern "C" fn tape_record_binop(op_tag: i32, a: i64, b: i64, out: i64) {
    if !RECORDING.with(|r| r.get()) {
        return;
    }
    tensor_retain(a);
    tensor_retain(b);
    tensor_retain(out);
    let op = OpTag::from_tag(op_tag);
    NODES.with(|n| {
        n.borrow_mut().push(TapeNode { op, saved: vec![a, b], output: out, meta: vec![] });
    });
}

/// Record a unary op onto the tape.  Retains x and out.  No-op when paused.
#[no_mangle]
pub extern "C" fn tape_record_unary(op_tag: i32, x: i64, out: i64) {
    if !RECORDING.with(|r| r.get()) {
        return;
    }
    tensor_retain(x);
    tensor_retain(out);
    let op = OpTag::from_tag(op_tag);
    NODES.with(|n| {
        n.borrow_mut().push(TapeNode { op, saved: vec![x], output: out, meta: vec![] });
    });
}

/// Record an axis-reduction op onto the tape.  Retains x and out.
/// meta = [axis, keepdim]; axis/keepdim passed as i64 because malus int literals are I64.
/// No-op when paused.
#[no_mangle]
pub extern "C" fn tape_record_reduce(op_tag: i32, x: i64, out: i64, axis: i64, keepdim: i64) {
    if !RECORDING.with(|r| r.get()) {
        return;
    }
    tensor_retain(x);
    tensor_retain(out);
    let op = OpTag::from_tag(op_tag);
    NODES.with(|n| {
        n.borrow_mut().push(TapeNode {
            op,
            saved: vec![x],
            output: out,
            meta: vec![axis, keepdim],
        });
    });
}

/// Record a permutation op (transpose/permute) onto the tape.  Retains x and
/// out; copies dims from dims_ptr into meta as i64 values.  No-op when paused.
#[no_mangle]
pub extern "C" fn tape_record_perm(
    op_tag: i32,
    x: i64,
    out: i64,
    dims_ptr: *const usize,
    ndims: usize,
) {
    if !RECORDING.with(|r| r.get()) {
        return;
    }
    tensor_retain(x);
    tensor_retain(out);
    let op = OpTag::from_tag(op_tag);
    let meta: Vec<i64> = if ndims == 0 || dims_ptr.is_null() {
        vec![]
    } else {
        unsafe { std::slice::from_raw_parts(dims_ptr, ndims) }
            .iter().map(|&v| v as i64).collect()
    };
    NODES.with(|n| {
        n.borrow_mut().push(TapeNode { op, saved: vec![x], output: out, meta });
    });
}

/// Record a layernorm op.  `var_h` is the population-variance tensor allocated
/// by tensor_layernorm_axis with keepdim=1 at the reduced axis; the caller
/// transfers ownership (refcount=1 from alloc, no extra retain needed).
/// Retains x and out.  Releases var_h when recording is paused.
#[no_mangle]
pub extern "C" fn tape_record_layernorm(op_tag: i32, x: i64, out: i64, var_h: i64, axis: i64) {
    if !RECORDING.with(|r| r.get()) {
        tensor_release(var_h);
        return;
    }
    tensor_retain(x);
    tensor_retain(out);
    // var_h: ownership transferred from caller (refcount=1 from alloc); no retain.
    let op = OpTag::from_tag(op_tag);
    NODES.with(|n| {
        n.borrow_mut().push(TapeNode {
            op,
            saved: vec![x, var_h],
            output: out,
            meta: vec![axis],
        });
    });
}

/// Record a cross-entropy op.  `softmax_h` is the softmax output allocated by
/// tensor_cross_entropy; caller transfers ownership.  Retains logits, out, and
/// targets.  Releases softmax_h when recording is paused.
#[no_mangle]
pub extern "C" fn tape_record_cross_entropy(
    op_tag: i32,
    logits: i64,
    out: i64,
    softmax_h: i64,
    targets: i64,
) {
    if !RECORDING.with(|r| r.get()) {
        tensor_release(softmax_h);
        return;
    }
    tensor_retain(logits);
    tensor_retain(out);
    // softmax_h: ownership transferred from caller; no retain.
    tensor_retain(targets);
    let op = OpTag::from_tag(op_tag);
    NODES.with(|n| {
        n.borrow_mut().push(TapeNode {
            op,
            saved: vec![logits, softmax_h, targets],
            output: out,
            meta: vec![],
        });
    });
}

/// Record an embedding op.  Retains weight, indices, and out.  Indices are
/// non-differentiable but must be retained so backward can read them.
/// No-op when paused.
#[no_mangle]
pub extern "C" fn tape_record_embedding(op_tag: i32, weight: i64, indices: i64, out: i64) {
    if !RECORDING.with(|r| r.get()) {
        return;
    }
    tensor_retain(weight);
    tensor_retain(indices);
    tensor_retain(out);
    let op = OpTag::from_tag(op_tag);
    NODES.with(|n| {
        n.borrow_mut().push(TapeNode { op, saved: vec![weight, indices], output: out, meta: vec![] });
    });
}

/// Mark handle as a leaf.  Always executes regardless of RECORDING flag so
/// that variable() inside a no_grad body still registers its leaf for the
/// next training step.
#[no_mangle]
pub extern "C" fn tape_register_leaf(handle: i64) {
    LEAVES.with(|l| {
        l.borrow_mut().insert(handle);
    });
}

#[no_mangle]
pub extern "C" fn tape_pause() {
    RECORDING.with(|r| r.set(false));
}

#[no_mangle]
pub extern "C" fn tape_resume() {
    RECORDING.with(|r| r.set(true));
}

/// Release all tape nodes and their saved/output handles.  Leaves LEAVES and
/// LEAF_GRAD intact (leaf grads survive to be read by .grad).
#[no_mangle]
pub extern "C" fn tape_clear() {
    NODES.with(|n| {
        let mut nodes = n.borrow_mut();
        for node in nodes.iter() {
            for &h in &node.saved {
                tensor_release(h);
            }
            tensor_release(node.output);
        }
        nodes.clear();
    });
}

/// Return the accumulated gradient for a leaf as an owned handle.
/// Retains the registry handle before returning so CTMM's static Drop
/// of the .grad result won't corrupt the registry.
/// Returns a fresh zeros tensor (same shape as the leaf) if no gradient
/// has been accumulated yet.
#[no_mangle]
pub extern "C" fn tape_get_grad(handle: i64) -> i64 {
    LEAF_GRAD.with(|lg| {
        let lg = lg.borrow();
        match lg.get(&handle) {
            Some(&grad) => {
                tensor_retain(grad);
                grad
            }
            None => {
                let tb = unsafe { &*(handle as *const TensorBuffer) };
                tensor_alloc_zeros_gpu(tb.shape.as_ptr(), tb.shape.len())
            }
        }
    })
}

/// Walk the tape in reverse, accumulate gradients into leaf .grad slots,
/// then clear the tape.  seed grad is ones_like(loss).
///
/// Every VJP dispatches a malus backward kernel via the BwdSlot table
/// (ADR-0032) — no CPU tensor arithmetic happens in this function. No
/// leading gpu_barrier(): each malus host fn's own `t[i]` reads get a
/// barrier auto-inserted by CTMM, so synchronization is handled per-call,
/// not upfront.
#[no_mangle]
pub extern "C" fn backward(loss: i64) {
    // Snapshot node data (clones Vec<i64> contents only; no TensorBuffer copies).
    struct NodeSnap {
        op:     OpTag,
        saved:  Vec<i64>,
        output: i64,
        meta:   Vec<i64>,
    }
    let nodes: Vec<NodeSnap> = NODES.with(|n| {
        n.borrow()
            .iter()
            .map(|node| NodeSnap {
                op:     node.op,
                saved:  node.saved.clone(),
                output: node.output,
                meta:   node.meta.clone(),
            })
            .collect()
    });

    // Transient grad map: input_handle → owned grad tensor (fresh per backward).
    let mut grads: HashMap<i64, i64> = HashMap::new();

    // Seed: ones_like(loss).  Allocation, not arithmetic — doesn't count.
    {
        let tb = unsafe { &*(loss as *const TensorBuffer) };
        let seed = tensor_alloc_ones_gpu(tb.shape.as_ptr(), tb.shape.len());
        grads.insert(loss, seed);
    }

    // Reverse walk.
    for node in nodes.iter().rev() {
        let dout = match grads.get(&node.output).copied() {
            Some(g) => g,
            None    => continue,
        };

        match node.op {
            OpTag::Matmul => {
                let (a, b) = (node.saved[0], node.saved[1]);
                let fa: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::MatmulBwdA)) };
                let fb: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::MatmulBwdB)) };
                let da = fa(a, b, dout);
                let db = fb(a, b, dout);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Add => {
                let (a, b) = (node.saved[0], node.saved[1]);
                let fa: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::AddBwdA)) };
                let fb: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::AddBwdB)) };
                let da = fa(dout, a);
                let db = fb(dout, b);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Sub => {
                let (a, b) = (node.saved[0], node.saved[1]);
                let fa: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::SubBwdA)) };
                let fb: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::SubBwdB)) };
                let da = fa(dout, a);
                let db = fb(dout, b);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Mul => {
                let (a, b) = (node.saved[0], node.saved[1]);
                let fa: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::MulBwdA)) };
                let fb: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::MulBwdB)) };
                let da = fa(dout, a, b);
                let db = fb(dout, a, b);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Div => {
                let (a, b) = (node.saved[0], node.saved[1]);
                let fa: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::DivBwdA)) };
                let fb: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::DivBwdB)) };
                let da = fa(dout, a, b);
                let db = fb(dout, a, b);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Sigmoid => {
                let x = node.saved[0];
                let s = node.output;
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::SigmoidBwd)) };
                let dx = f(s, dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Relu => {
                let x = node.saved[0];
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::ReluBwd)) };
                let dx = f(x, dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Tanh => {
                let x = node.saved[0];
                let t = node.output;
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::TanhBwd)) };
                let dx = f(t, dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Exp => {
                // e = exp(x)  ->  dx = dout * e — reuses the forward
                // broadcast-mul kernel directly (same-shape multiply).
                let x = node.saved[0];
                let e = node.output;
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::ExpBwd)) };
                let dx = f(dout, e);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Log => {
                // l = log(x)  ->  dx = dout / x — reuses the forward
                // broadcast-div kernel directly (same-shape divide).
                let x = node.saved[0];
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::LogBwd)) };
                let dx = f(dout, x);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Sqrt => {
                let x = node.saved[0];
                let s = node.output;
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::SqrtBwd)) };
                let dx = f(s, dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Abs => {
                let x = node.saved[0];
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::AbsBwd)) };
                let dx = f(x, dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Sum => {
                // s = sum(x) (full-tensor)  ->  dx = fill_like(x, dout[0]).
                // dout[0] is read inside the malus host fn (CTMM-inserted
                // barrier), not here.
                let x = node.saved[0];
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::SumBwd)) };
                let dx = f(x, dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Transpose => {
                // B = permute(A, perm)  ->  dA = permute(dB, inverse_perm).
                // Computing the inverse perm is scalar index arithmetic over
                // a length<=8 list — orchestration, not tensor compute
                // (ADR-0031) — so it stays in Rust; the actual gather is the
                // rank-generic malus __permute_nd_kernel via the same forward
                // host fn the eager path uses (M33).
                let x = node.saved[0];
                let rank = tb(x).shape.len();
                let raw: Vec<usize> = node.meta.iter().map(|&v| v as usize).collect();
                let perm = normalize_perm(&raw, rank);
                let inv  = invert_perm(&perm);
                let dx = crate::metal::permute_nd_gpu(dout, &inv);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Reshape => {
                // y = reshape(x, new_shape)  ->  dx = reshape(dy, x.shape).
                // Zero-copy (same MTLBuffer) — orchestration, not compute;
                // stays the existing Rust helper (ADR-0031).
                let x = node.saved[0];
                let x_shape = tb(x).shape.clone();
                let dx = reshape_to(dout, &x_shape);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Neg => {
                // y = -x  ->  dx = -dout — reuses the scale-by-constant
                // kernel with c=-1.0.
                let x = node.saved[0];
                let f: extern "C" fn(i64, f32) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::NegBwd)) };
                let dx = f(dout, -1.0);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceSumAxis => {
                let x = node.saved[0];
                let axis = node.meta[0];
                let f: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::ReduceSumAxisBwd)) };
                let dx = f(x, dout, axis);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceMeanAxis => {
                let x = node.saved[0];
                let axis = node.meta[0];
                let f: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::ReduceMeanAxisBwd)) };
                let dx = f(x, dout, axis);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceMaxAxis => {
                let x = node.saved[0];
                let y = node.output;
                let axis = node.meta[0];
                let f: extern "C" fn(i64, i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::ReduceMaxAxisBwd)) };
                let dx = f(x, y, dout, axis);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceVarAxis => {
                let x = node.saved[0];
                let axis = node.meta[0];
                let f: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::ReduceVarAxisBwd)) };
                let dx = f(x, dout, axis);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Softmax => {
                let x = node.saved[0];
                let s = node.output;
                let axis = node.meta[0];
                let f: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::SoftmaxBwd)) };
                let dx = f(s, dout, axis);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Layernorm => {
                let x = node.saved[0];
                let var_h = node.saved[1];
                let y = node.output;
                let axis = node.meta[0];
                let f: extern "C" fn(i64, i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::LayernormBwd)) };
                let dx = f(y, var_h, dout, axis);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Gelu => {
                let x = node.saved[0];
                let f: extern "C" fn(i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::GeluBwd)) };
                let dx = f(x, dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::CrossEntropy => {
                let logits  = node.saved[0];
                let sm_h    = node.saved[1];
                let targets = node.saved[2];
                let f: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::CrossEntropyBwd)) };
                let dx = f(sm_h, targets, dout);
                accumulate_grad(&mut grads, logits, dx);
            }
            OpTag::Embedding => {
                let weight  = node.saved[0];
                let indices = node.saved[1];
                let f: extern "C" fn(i64, i64, i64) -> i64 =
                    unsafe { std::mem::transmute(bwd_slot(BwdSlot::EmbeddingBwd)) };
                let dweight = f(weight, indices, dout);
                accumulate_grad(&mut grads, weight, dweight);
            }
        }
    }

    // Fold each leaf's transient grad into the persistent registry (accumulate).
    // Stash handles to release outside the borrow so tape_on_release (called from
    // tensor_release) can borrow_mut LEAVES/LEAF_GRAD without hitting a re-entrant panic.
    let mut to_release: Vec<i64> = Vec::new();
    LEAVES.with(|leaves_cell| {
        LEAF_GRAD.with(|lg_cell| {
            let leaves = leaves_cell.borrow();
            let mut lg = lg_cell.borrow_mut();
            for &leaf in leaves.iter() {
                if let Some(new_grad) = grads.remove(&leaf) {
                    match lg.remove(&leaf) {
                        None => {
                            lg.insert(leaf, new_grad);
                        }
                        Some(old_grad) => {
                            let summed = grad_acc(old_grad, new_grad);
                            to_release.push(old_grad);
                            to_release.push(new_grad);
                            lg.insert(leaf, summed);
                        }
                    }
                }
            }
        });
    });
    for h in to_release {
        tensor_release(h);
    }

    // Release remaining transient grads (intermediates not in LEAVES).
    for (_, grad) in grads {
        tensor_release(grad);
    }

    tape_clear();
}

// ── Test / reset helper (not extern "C"; not JIT-injected) ───────────────────

/// Clear all tape state including the persistent leaf-grad registry and leaves
/// set.  Used by tests.  Not JIT-injected.
pub fn tape_reset() {
    LEAVES.with(|l| l.borrow_mut().clear());
    let mut to_release: Vec<i64> = Vec::new();
    LEAF_GRAD.with(|lg| {
        let mut lg = lg.borrow_mut();
        for (_, grad) in lg.drain() {
            to_release.push(grad);
        }
    });
    for g in to_release {
        tensor_release(g);
    }
    tape_clear();
    RECORDING.with(|r| r.set(true));
}

/// Lazily clear the accumulated gradient for each passed leaf: release the
/// stored grad tensor and remove the registry entry.  tape_get_grad already
/// returns a fresh zeros tensor when the entry is absent, so the observable
/// effect is identical to storing zeros.  Called by the JIT-lowered zero_grad
/// builtin at the start of each training step.
#[no_mangle]
pub extern "C" fn tape_zero_grad(handles: *const i64, count: usize) {
    if handles.is_null() || count == 0 {
        return;
    }
    let hs = unsafe { std::slice::from_raw_parts(handles, count) };
    let mut to_release: Vec<i64> = Vec::new();
    LEAF_GRAD.with(|lg| {
        let mut lg = lg.borrow_mut();
        for &h in hs {
            if let Some(grad) = lg.remove(&h) {
                to_release.push(grad);
            }
        }
    });
    for g in to_release {
        tensor_release(g);
    }
}

/// Called by tensor_release (metal.rs) on the 1→0 refcount transition,
/// before the TensorBuffer box is dropped.  Removes the handle from LEAVES
/// and releases + removes its LEAF_GRAD entry so the leaf registry never
/// outlives its tensor.  Not extern "C" — called directly from metal.rs
/// (same crate).
pub(crate) fn tape_on_release(handle: i64) {
    let grad = LEAF_GRAD.with(|lg| lg.borrow_mut().remove(&handle));
    LEAVES.with(|l| { l.borrow_mut().remove(&handle); });
    if let Some(g) = grad {
        // g is a different handle (a grad tensor); it is not a leaf, so the
        // recursive tensor_release → tape_on_release is a bounded lookup-miss.
        tensor_release(g);
    }
}

#[cfg(test)]
pub(crate) fn registry_lens() -> (usize, usize) {
    (
        LEAVES.with(|l| l.borrow().len()),
        LEAF_GRAD.with(|lg| lg.borrow().len()),
    )
}

// ── Shared helpers ─────────────────────────────────────────────────────────

fn tb(handle: i64) -> &'static TensorBuffer {
    unsafe { &*(handle as *const TensorBuffer) }
}

/// Add new_grad into old_grad via the registered GradAcc slot (the forward
/// broadcast-add kernel — gradients accumulated against the same node always
/// share its shape, so the broadcast machinery degenerates to a same-shape
/// add). Does not release operands; callers own that.
fn grad_acc(old: i64, new: i64) -> i64 {
    let f: extern "C" fn(i64, i64) -> i64 =
        unsafe { std::mem::transmute(bwd_slot(BwdSlot::GradAcc)) };
    f(old, new)
}

/// Add new_grad into grads[input_handle].  Takes ownership of new_grad.
fn accumulate_grad(grads: &mut HashMap<i64, i64>, input_handle: i64, new_grad: i64) {
    match grads.remove(&input_handle) {
        None => {
            grads.insert(input_handle, new_grad);
        }
        Some(old) => {
            let summed = grad_acc(old, new_grad);
            tensor_release(old);
            tensor_release(new_grad);
            grads.insert(input_handle, summed);
        }
    }
}

// ── CPU fallback (M26 / ADR-0031 / ADR-0032) ─────────────────────────────────
//
// Self-registers the legacy V3 Rust VJP closures into the BwdSlot table the
// first time `backward()` runs, so malus-runtime's own isolated tape tests
// (which exercise `backward()` directly, without going through
// compile_and_run/malus-stdlib) keep working unchanged. Production wiring
// (codegen-cpu) always registers every slot with real GPU kernel pointers
// before `backward()` is ever called, so this lazy path never fires there —
// and without the `cpu_fallback` feature, this module doesn't exist at all,
// making a stray CPU-compute path structurally impossible to reach in the
// canonical gate build (the M26 done-when).
#[cfg(feature = "cpu_fallback")]
mod cpu_fallback {
    use std::cell::Cell;

    use objc2_metal::MTLBuffer;

    use crate::metal::{
        broadcast_to_shape, permute_by_perm, sum_to_shape,
        tensor_alloc_gpu, tensor_matmul, tensor_reduce_mean_axis, tensor_reduce_sum_axis,
        tensor_release, tensor_scatter_add, Dtype,
    };

    use super::{tape_register_backward_fn, tb, BwdSlot};

    thread_local! {
        static REGISTERED: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn ensure_registered() {
        REGISTERED.with(|r| {
            if r.get() {
                return;
            }
            r.set(true);
            register_all();
        });
    }

    fn register_all() {
        tape_register_backward_fn(BwdSlot::AddBwdA as i32, add_bwd_a as *const () as usize);
        tape_register_backward_fn(BwdSlot::AddBwdB as i32, add_bwd_b as *const () as usize);
        tape_register_backward_fn(BwdSlot::SubBwdA as i32, sub_bwd_a as *const () as usize);
        tape_register_backward_fn(BwdSlot::SubBwdB as i32, sub_bwd_b as *const () as usize);
        tape_register_backward_fn(BwdSlot::MulBwdA as i32, mul_bwd_a as *const () as usize);
        tape_register_backward_fn(BwdSlot::MulBwdB as i32, mul_bwd_b as *const () as usize);
        tape_register_backward_fn(BwdSlot::DivBwdA as i32, div_bwd_a as *const () as usize);
        tape_register_backward_fn(BwdSlot::DivBwdB as i32, div_bwd_b as *const () as usize);
        tape_register_backward_fn(BwdSlot::SigmoidBwd as i32, sigmoid_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::ReluBwd as i32, relu_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::TanhBwd as i32, tanh_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::ExpBwd as i32, exp_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::LogBwd as i32, log_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::SqrtBwd as i32, sqrt_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::AbsBwd as i32, abs_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::NegBwd as i32, neg_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::SumBwd as i32, sum_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::PermuteNdFwd as i32, permute_nd_fwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::ReduceSumAxisBwd as i32, reduce_sum_axis_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::ReduceMeanAxisBwd as i32, reduce_mean_axis_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::ReduceMaxAxisBwd as i32, reduce_max_axis_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::ReduceVarAxisBwd as i32, reduce_var_axis_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::SoftmaxBwd as i32, softmax_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::LayernormBwd as i32, layernorm_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::GeluBwd as i32, gelu_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::CrossEntropyBwd as i32, cross_entropy_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::EmbeddingBwd as i32, embedding_bwd as *const () as usize);
        tape_register_backward_fn(BwdSlot::MatmulBwdA as i32, matmul_bwd_a as *const () as usize);
        tape_register_backward_fn(BwdSlot::MatmulBwdB as i32, matmul_bwd_b as *const () as usize);
        tape_register_backward_fn(BwdSlot::GradAcc as i32, elem_add as *const () as usize);
    }

    fn read(handle: i64) -> Vec<f32> {
        crate::metal::flush_if_pending(handle);
        let t = tb(handle);
        let ptr = t.buffer.contents().as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, t.len).to_vec() }
    }

    fn alloc_like(template: i64, data: &[f32]) -> i64 {
        let t = tb(template);
        tensor_alloc_gpu(0, t.shape.as_ptr(), t.shape.len(), data.as_ptr())
    }

    fn elem_add(a: i64, b: i64) -> i64 {
        crate::cpu_compute_inc();
        let (ad, bd) = (read(a), read(b));
        let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect();
        alloc_like(a, &out)
    }

    fn elem_sub(a: i64, b: i64) -> i64 {
        crate::cpu_compute_inc();
        let (ad, bd) = (read(a), read(b));
        let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x - y).collect();
        alloc_like(a, &out)
    }

    fn elem_cmp_eq(a: i64, b: i64) -> i64 {
        crate::cpu_compute_inc();
        let (ad, bd) = (read(a), read(b));
        let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| if x == y { 1.0 } else { 0.0 }).collect();
        alloc_like(a, &out)
    }

    fn elem_mul(a: i64, b: i64) -> i64 {
        crate::cpu_compute_inc();
        let (ad, bd) = (read(a), read(b));
        let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x * y).collect();
        alloc_like(a, &out)
    }

    fn elem_div(a: i64, b: i64) -> i64 {
        crate::cpu_compute_inc();
        let (ad, bd) = (read(a), read(b));
        let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x / y).collect();
        alloc_like(a, &out)
    }

    fn scalar_mul(a: i64, s: f32) -> i64 {
        crate::cpu_compute_inc();
        let ad = read(a);
        let out: Vec<f32> = ad.iter().map(|x| x * s).collect();
        alloc_like(a, &out)
    }

    fn elem_apply(a: i64, f: impl Fn(f32) -> f32) -> i64 {
        crate::cpu_compute_inc();
        let ad = read(a);
        let out: Vec<f32> = ad.iter().map(|&v| f(v)).collect();
        alloc_like(a, &out)
    }

    fn read_int_index(buf: *const u8, i: usize, dtype: Dtype) -> usize {
        match dtype {
            Dtype::I32 => unsafe { *(buf.add(i * 4) as *const i32) as usize }
            Dtype::I64 => unsafe { *(buf.add(i * 8) as *const i64) as usize }
            _ => panic!("malus: integer index tensor must be Tensor<i32> or Tensor<i64>, got {:?}", dtype),
        }
    }

    extern "C" fn add_bwd_a(dout: i64, a: i64) -> i64 { sum_to_shape(dout, &tb(a).shape.clone()) }
    extern "C" fn add_bwd_b(dout: i64, b: i64) -> i64 { sum_to_shape(dout, &tb(b).shape.clone()) }
    extern "C" fn sub_bwd_a(dout: i64, a: i64) -> i64 { sum_to_shape(dout, &tb(a).shape.clone()) }
    extern "C" fn sub_bwd_b(dout: i64, b: i64) -> i64 {
        let neg = scalar_mul(dout, -1.0);
        let r = sum_to_shape(neg, &tb(b).shape.clone());
        tensor_release(neg);
        r
    }

    extern "C" fn mul_bwd_a(dout: i64, a: i64, b: i64) -> i64 {
        let out_shape = tb(dout).shape.clone();
        let b_bc = broadcast_to_shape(b, &out_shape);
        let full = elem_mul(dout, b_bc);
        tensor_release(b_bc);
        let r = sum_to_shape(full, &tb(a).shape.clone());
        tensor_release(full);
        r
    }
    extern "C" fn mul_bwd_b(dout: i64, a: i64, b: i64) -> i64 {
        let out_shape = tb(dout).shape.clone();
        let a_bc = broadcast_to_shape(a, &out_shape);
        let full = elem_mul(a_bc, dout);
        tensor_release(a_bc);
        let r = sum_to_shape(full, &tb(b).shape.clone());
        tensor_release(full);
        r
    }
    extern "C" fn div_bwd_a(dout: i64, a: i64, b: i64) -> i64 {
        let out_shape = tb(dout).shape.clone();
        let b_bc = broadcast_to_shape(b, &out_shape);
        let full = elem_div(dout, b_bc);
        tensor_release(b_bc);
        let r = sum_to_shape(full, &tb(a).shape.clone());
        tensor_release(full);
        r
    }
    extern "C" fn div_bwd_b(dout: i64, a: i64, b: i64) -> i64 {
        let out_shape = tb(dout).shape.clone();
        let a_bc = broadcast_to_shape(a, &out_shape);
        let tmp = elem_mul(dout, a_bc);
        tensor_release(a_bc);
        let neg_tmp = scalar_mul(tmp, -1.0);
        tensor_release(tmp);
        let b_bc = broadcast_to_shape(b, &out_shape);
        let b_sq = elem_mul(b_bc, b_bc);
        tensor_release(b_bc);
        let full = elem_div(neg_tmp, b_sq);
        tensor_release(neg_tmp);
        tensor_release(b_sq);
        let r = sum_to_shape(full, &tb(b).shape.clone());
        tensor_release(full);
        r
    }

    extern "C" fn sigmoid_bwd(s: i64, dout: i64) -> i64 {
        let one_minus_s = elem_apply(s, |v| 1.0 - v);
        let tmp = elem_mul(dout, s);
        let dx = elem_mul(tmp, one_minus_s);
        tensor_release(tmp);
        tensor_release(one_minus_s);
        dx
    }
    extern "C" fn relu_bwd(x: i64, dout: i64) -> i64 {
        let mask = elem_apply(x, |v| if v > 0.0 { 1.0 } else { 0.0 });
        let dx = elem_mul(dout, mask);
        tensor_release(mask);
        dx
    }
    extern "C" fn tanh_bwd(t: i64, dout: i64) -> i64 {
        let one_minus_t_sq = elem_apply(t, |v| 1.0 - v * v);
        let dx = elem_mul(dout, one_minus_t_sq);
        tensor_release(one_minus_t_sq);
        dx
    }
    extern "C" fn exp_bwd(dout: i64, e: i64) -> i64 { elem_mul(dout, e) }
    extern "C" fn log_bwd(dout: i64, x: i64) -> i64 { elem_div(dout, x) }
    extern "C" fn sqrt_bwd(s: i64, dout: i64) -> i64 {
        let two_s = scalar_mul(s, 2.0);
        let dx = elem_div(dout, two_s);
        tensor_release(two_s);
        dx
    }
    extern "C" fn abs_bwd(x: i64, dout: i64) -> i64 {
        let sign = elem_apply(x, |v| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 });
        let dx = elem_mul(dout, sign);
        tensor_release(sign);
        dx
    }
    extern "C" fn neg_bwd(dout: i64, c: f32) -> i64 { scalar_mul(dout, c) }

    extern "C" fn sum_bwd(x: i64, dout: i64) -> i64 {
        let scalar_val = {
            crate::metal::flush_if_pending(dout);
            let t = tb(dout);
            let ptr = t.buffer.contents().as_ptr() as *const f32;
            unsafe { *ptr }
        };
        elem_apply(x, |_| scalar_val)
    }

    // Mock for the PermuteNdFwd slot: same 9-arg ABI as the JIT'd
    // __permute_nd_fwd — only the first rank(x) perm entries are meaningful.
    extern "C" fn permute_nd_fwd(
        x: i64,
        p0: i64, p1: i64, p2: i64, p3: i64,
        p4: i64, p5: i64, p6: i64, p7: i64,
    ) -> i64 {
        let rank = tb(x).shape.len();
        let all = [p0, p1, p2, p3, p4, p5, p6, p7];
        let perm: Vec<usize> = all[..rank].iter().map(|&v| v as usize).collect();
        permute_by_perm(x, &perm)
    }

    extern "C" fn reduce_sum_axis_bwd(x: i64, dout: i64, axis: i64) -> i64 {
        use crate::metal::unsqueeze_at;
        let axis = axis as usize;
        let x_shape = tb(x).shape.clone();
        let dout_exp = if x_shape.len() == tb(dout).shape.len() { tensor_retain_ret(dout) } else { unsqueeze_at(dout, axis) };
        let dx = broadcast_to_shape(dout_exp, &x_shape);
        tensor_release(dout_exp);
        dx
    }
    extern "C" fn reduce_mean_axis_bwd(x: i64, dout: i64, axis: i64) -> i64 {
        use crate::metal::unsqueeze_at;
        let axis = axis as usize;
        let x_shape = tb(x).shape.clone();
        let n = x_shape[axis] as f32;
        let dout_scaled = scalar_mul(dout, 1.0 / n);
        let dout_exp = if x_shape.len() == tb(dout_scaled).shape.len() {
            dout_scaled
        } else {
            let u = unsqueeze_at(dout_scaled, axis);
            tensor_release(dout_scaled);
            u
        };
        let dx = broadcast_to_shape(dout_exp, &x_shape);
        tensor_release(dout_exp);
        dx
    }
    extern "C" fn reduce_max_axis_bwd(x: i64, y: i64, dout: i64, axis: i64) -> i64 {
        use crate::metal::unsqueeze_at;
        let axis = axis as usize;
        let x_shape = tb(x).shape.clone();
        let out_exp = if x_shape.len() == tb(y).shape.len() { tensor_retain_ret(y) } else { unsqueeze_at(y, axis) };
        let out_bc = broadcast_to_shape(out_exp, &x_shape);
        tensor_release(out_exp);
        let mask = elem_cmp_eq(x, out_bc);
        tensor_release(out_bc);
        let dout_exp = if x_shape.len() == tb(dout).shape.len() { tensor_retain_ret(dout) } else { unsqueeze_at(dout, axis) };
        let dout_bc = broadcast_to_shape(dout_exp, &x_shape);
        tensor_release(dout_exp);
        let dx = elem_mul(dout_bc, mask);
        tensor_release(dout_bc);
        tensor_release(mask);
        dx
    }
    extern "C" fn reduce_var_axis_bwd(x: i64, dout: i64, axis_i64: i64) -> i64 {
        use crate::metal::unsqueeze_at;
        let axis = axis_i64 as usize;
        let x_shape = tb(x).shape.clone();
        let n = x_shape[axis] as f32;
        let mean_h = tensor_reduce_mean_axis(x, axis_i64, 1);
        let mean_bc = broadcast_to_shape(mean_h, &x_shape);
        tensor_release(mean_h);
        let x_minus_mean = elem_sub(x, mean_bc);
        tensor_release(mean_bc);
        let scaled = scalar_mul(x_minus_mean, 2.0 / n);
        tensor_release(x_minus_mean);
        let dout_exp = if x_shape.len() == tb(dout).shape.len() { tensor_retain_ret(dout) } else { unsqueeze_at(dout, axis) };
        let dout_bc = broadcast_to_shape(dout_exp, &x_shape);
        tensor_release(dout_exp);
        let dx = elem_mul(dout_bc, scaled);
        tensor_release(dout_bc);
        tensor_release(scaled);
        dx
    }

    extern "C" fn softmax_bwd(s: i64, dout: i64, axis: i64) -> i64 {
        let x_shape = tb(s).shape.clone();
        let dout_s = elem_mul(dout, s);
        let sum_ds = tensor_reduce_sum_axis(dout_s, axis, 1);
        tensor_release(dout_s);
        let sum_bc = broadcast_to_shape(sum_ds, &x_shape);
        tensor_release(sum_ds);
        let diff = elem_sub(dout, sum_bc);
        tensor_release(sum_bc);
        let dx = elem_mul(s, diff);
        tensor_release(diff);
        dx
    }
    extern "C" fn layernorm_bwd(y: i64, var_h: i64, dout: i64, axis: i64) -> i64 {
        let x_shape = tb(y).shape.clone();
        let inv_sigma_h = elem_apply(var_h, |v| 1.0 / (v + 1e-5_f32).sqrt());
        let inv_sigma_bc = broadcast_to_shape(inv_sigma_h, &x_shape);
        tensor_release(inv_sigma_h);
        let dy_mean_h = tensor_reduce_mean_axis(dout, axis, 1);
        let dy_mean_bc = broadcast_to_shape(dy_mean_h, &x_shape);
        tensor_release(dy_mean_h);
        let dy_y = elem_mul(dout, y);
        let dy_y_mean_h = tensor_reduce_mean_axis(dy_y, axis, 1);
        tensor_release(dy_y);
        let dy_y_mean_bc = broadcast_to_shape(dy_y_mean_h, &x_shape);
        tensor_release(dy_y_mean_h);
        let y_term = elem_mul(y, dy_y_mean_bc);
        tensor_release(dy_y_mean_bc);
        let tmp = elem_sub(dout, dy_mean_bc);
        tensor_release(dy_mean_bc);
        let numer = elem_sub(tmp, y_term);
        tensor_release(tmp);
        tensor_release(y_term);
        let dx = elem_mul(numer, inv_sigma_bc);
        tensor_release(numer);
        tensor_release(inv_sigma_bc);
        dx
    }
    extern "C" fn gelu_bwd(x: i64, dout: i64) -> i64 {
        const C0: f32 = 0.7978845608;
        const C1: f32 = 0.044715;
        let gelu_deriv = elem_apply(x, |xi| {
            let g  = C0 * (xi + C1 * xi * xi * xi);
            let t  = g.tanh();
            let gp = C0 * (1.0 + 3.0 * C1 * xi * xi);
            0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * gp
        });
        let dx = elem_mul(dout, gelu_deriv);
        tensor_release(gelu_deriv);
        dx
    }
    extern "C" fn cross_entropy_bwd(probs: i64, targets: i64, dout: i64) -> i64 {
        crate::cpu_compute_inc();
        let n = tb(probs).shape[0];
        let c = tb(probs).shape[1];
        let scale = read(dout)[0] / n as f32;
        let mut grad_data = read(probs);
        crate::metal::flush_if_pending(targets);
        let tgt_buf = tb(targets).buffer.contents().as_ptr() as *const u8;
        let tgt_dtype = tb(targets).dtype;
        for i in 0..n {
            let t = read_int_index(tgt_buf, i, tgt_dtype);
            grad_data[i * c + t] -= 1.0;
        }
        for v in grad_data.iter_mut() { *v *= scale; }
        alloc_like(probs, &grad_data)
    }
    extern "C" fn embedding_bwd(weight: i64, indices: i64, dout: i64) -> i64 {
        let vocab_size = tb(weight).shape[0] as i64;
        tensor_scatter_add(dout, indices, vocab_size)
    }

    extern "C" fn matmul_bwd_a(a: i64, b: i64, dout: i64) -> i64 {
        let (rank_a, rank_b) = (tb(a).shape.len(), tb(b).shape.len());
        if rank_a == 2 && rank_b == 2 {
            let bt = permute_by_perm(b, &[1, 0]);
            let da = tensor_matmul(dout, bt);
            tensor_release(bt);
            da
        } else if rank_a == 3 && rank_b == 3 {
            let bt = permute_by_perm(b, &[0, 2, 1]);
            let da = tensor_matmul(dout, bt);
            tensor_release(bt);
            da
        } else {
            let bt = permute_by_perm(b, &[1, 0]);
            let da = tensor_matmul(dout, bt);
            tensor_release(bt);
            da
        }
    }
    extern "C" fn matmul_bwd_b(a: i64, b: i64, dout: i64) -> i64 {
        let (rank_a, rank_b) = (tb(a).shape.len(), tb(b).shape.len());
        if rank_a == 2 && rank_b == 2 {
            let at = permute_by_perm(a, &[1, 0]);
            let db = tensor_matmul(at, dout);
            tensor_release(at);
            db
        } else if rank_a == 3 && rank_b == 3 {
            let at = permute_by_perm(a, &[0, 2, 1]);
            let db = tensor_matmul(at, dout);
            tensor_release(at);
            db
        } else {
            let at = permute_by_perm(a, &[0, 2, 1]);
            let db_3d = tensor_matmul(at, dout);
            tensor_release(at);
            let db = tensor_reduce_sum_axis(db_3d, 0, 0);
            tensor_release(db_3d);
            db
        }
    }

    fn tensor_retain_ret(h: i64) -> i64 {
        crate::metal::tensor_retain(h);
        h
    }
}
