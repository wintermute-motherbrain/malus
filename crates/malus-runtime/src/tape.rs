use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use objc2_metal::MTLBuffer;

use crate::metal::{
    broadcast_to_shape, gpu_barrier, invert_perm, normalize_perm, permute_by_perm,
    reshape_to, sum_to_shape, tensor_alloc_gpu, tensor_alloc_ones_gpu,
    tensor_alloc_zeros_gpu, tensor_matmul, tensor_reduce_mean_axis, tensor_reduce_sum_axis,
    tensor_retain, tensor_release, tensor_scatter_add, unsqueeze_at, TensorBuffer,
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
#[no_mangle]
pub extern "C" fn backward(loss: i64) {
    // Flush any pending GPU work so saved handles are readable on the CPU.
    gpu_barrier();

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

    // Seed: ones_like(loss).
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
                let (rank_a, rank_b) = (tb(a).shape.len(), tb(b).shape.len());
                if rank_a == 2 && rank_b == 2 {
                    // C (M,N) = A (M,K) @ B (K,N)
                    let bt = permute_by_perm(b, &[1, 0]);
                    let da = tensor_matmul(dout, bt);
                    tensor_release(bt);
                    let at = permute_by_perm(a, &[1, 0]);
                    let db = tensor_matmul(at, dout);
                    tensor_release(at);
                    accumulate_grad(&mut grads, a, da);
                    accumulate_grad(&mut grads, b, db);
                } else if rank_a == 3 && rank_b == 3 {
                    // C (B,M,N) = A (B,M,K) @ B (B,K,N)
                    let bt = permute_by_perm(b, &[0, 2, 1]);
                    let da = tensor_matmul(dout, bt);
                    tensor_release(bt);
                    let at = permute_by_perm(a, &[0, 2, 1]);
                    let db = tensor_matmul(at, dout);
                    tensor_release(at);
                    accumulate_grad(&mut grads, a, da);
                    accumulate_grad(&mut grads, b, db);
                } else {
                    // C (B,M,N) = A (B,M,K) @ B (K,N)
                    // dA (B,M,K) = dC (B,M,N) @ Bᵀ (N,K)   [3D @ 2D]
                    // dB (K,N)   = sum_b( Aᵀ[b] @ dC[b] ) = reduce_sum(Aᵀ (B,K,M) @ dC (B,M,N), axis=0)
                    let bt = permute_by_perm(b, &[1, 0]);
                    let da = tensor_matmul(dout, bt);     // (B,M,N) @ (N,K) → (B,M,K) via (3,2)
                    tensor_release(bt);
                    let at = permute_by_perm(a, &[0, 2, 1]);   // (B,K,M)
                    let db_3d = tensor_matmul(at, dout);       // (B,K,M) @ (B,M,N) → (B,K,N)
                    tensor_release(at);
                    let db = tensor_reduce_sum_axis(db_3d, 0, 0); // sum batch → (K,N)
                    tensor_release(db_3d);
                    accumulate_grad(&mut grads, a, da);
                    accumulate_grad(&mut grads, b, db);
                }
            }
            OpTag::Add => {
                // C = A + B  →  dA = reduce_to(dC, A.shape),  dB = reduce_to(dC, B.shape)
                // Identity fast-paths when shapes match (no broadcast).
                let (a, b) = (node.saved[0], node.saved[1]);
                let da = sum_to_shape(dout, &tb(a).shape.clone());
                let db = sum_to_shape(dout, &tb(b).shape.clone());
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Sub => {
                // C = A - B  →  dA = reduce_to(dC, A.shape),  dB = -reduce_to(dC, B.shape)
                let (a, b) = (node.saved[0], node.saved[1]);
                let da = sum_to_shape(dout, &tb(a).shape.clone());
                let neg_dout = scalar_mul(dout, -1.0);
                let db = sum_to_shape(neg_dout, &tb(b).shape.clone());
                tensor_release(neg_dout);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Mul => {
                // C = A * B  →  dA = reduce_to(dC * broadcast(B), A.shape)
                //               dB = reduce_to(broadcast(A) * dC, B.shape)
                let (a, b) = (node.saved[0], node.saved[1]);
                let out_shape = tb(dout).shape.clone();
                let a_shape = tb(a).shape.clone();
                let b_shape = tb(b).shape.clone();
                let b_bc = broadcast_to_shape(b, &out_shape);
                let da_full = elem_mul(dout, b_bc);
                tensor_release(b_bc);
                let da = sum_to_shape(da_full, &a_shape);
                tensor_release(da_full);
                let a_bc = broadcast_to_shape(a, &out_shape);
                let db_full = elem_mul(a_bc, dout);
                tensor_release(a_bc);
                let db = sum_to_shape(db_full, &b_shape);
                tensor_release(db_full);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Div => {
                // C = A / B  →  dA = reduce_to(dC / broadcast(B), A.shape)
                //               dB = reduce_to(-dC * broadcast(A) / broadcast(B)², B.shape)
                let (a, b) = (node.saved[0], node.saved[1]);
                let out_shape = tb(dout).shape.clone();
                let a_shape = tb(a).shape.clone();
                let b_shape = tb(b).shape.clone();
                let b_bc = broadcast_to_shape(b, &out_shape);
                let da_full = elem_div(dout, b_bc);
                tensor_release(b_bc);
                let da = sum_to_shape(da_full, &a_shape);
                tensor_release(da_full);
                let a_bc = broadcast_to_shape(a, &out_shape);
                let tmp = elem_mul(dout, a_bc);
                tensor_release(a_bc);
                let neg_tmp = scalar_mul(tmp, -1.0);
                tensor_release(tmp);
                let b_bc2 = broadcast_to_shape(b, &out_shape);
                let b_sq = elem_mul(b_bc2, b_bc2);
                tensor_release(b_bc2);
                let db_full = elem_div(neg_tmp, b_sq);
                tensor_release(neg_tmp);
                tensor_release(b_sq);
                let db = sum_to_shape(db_full, &b_shape);
                tensor_release(db_full);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Sigmoid => {
                // s = σ(x)  →  dx = dC * s * (1 - s)
                // node.output = s (forward output, retained by tape)
                let x = node.saved[0];
                let s = node.output;
                let one_minus_s = elem_apply(s, |v| 1.0 - v);
                let tmp = elem_mul(dout, s);
                let dx = elem_mul(tmp, one_minus_s);
                tensor_release(tmp);
                tensor_release(one_minus_s);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Relu => {
                // r = max(x, 0)  →  dx = dC * (x > 0)
                let x = node.saved[0];
                let mask = elem_apply(x, |v| if v > 0.0 { 1.0 } else { 0.0 });
                let dx = elem_mul(dout, mask);
                tensor_release(mask);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Tanh => {
                // t = tanh(x)  →  dx = dC * (1 - t²)
                let x = node.saved[0];
                let t = node.output;
                let one_minus_t_sq = elem_apply(t, |v| 1.0 - v * v);
                let dx = elem_mul(dout, one_minus_t_sq);
                tensor_release(one_minus_t_sq);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Exp => {
                // e = exp(x)  →  dx = dC * e
                let x = node.saved[0];
                let e = node.output;
                let dx = elem_mul(dout, e);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Log => {
                // l = log(x)  →  dx = dC / x
                let x = node.saved[0];
                let dx = elem_div(dout, x);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Sqrt => {
                // s = sqrt(x)  →  dx = dC / (2 * s)
                let x = node.saved[0];
                let s = node.output;
                let two_s = scalar_mul(s, 2.0);
                let dx = elem_div(dout, two_s);
                tensor_release(two_s);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Abs => {
                // a = |x|  →  dx = dC * sign(x)
                let x = node.saved[0];
                let sign = elem_apply(x, |v| {
                    if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
                });
                let dx = elem_mul(dout, sign);
                tensor_release(sign);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Sum => {
                // s = sum(x)  →  dx = ones_like(x) * dC[0]
                // dout is a [1] tensor; read the scalar value.
                let x = node.saved[0];
                let scalar_val = {
                    let tb = unsafe { &*(dout as *const TensorBuffer) };
                    let ptr = tb.buffer.contents().as_ptr() as *const f32;
                    unsafe { *ptr }
                };
                let dx = elem_apply(x, |_| scalar_val);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Transpose => {
                // B = permute(A, perm)  →  dA = permute(dB, inverse_perm)
                // meta holds the raw dim args recorded at forward time.
                let x = node.saved[0];
                let rank = tb(x).shape.len();
                let raw: Vec<usize> = node.meta.iter().map(|&v| v as usize).collect();
                let perm = normalize_perm(&raw, rank);
                let inv  = invert_perm(&perm);
                let dx   = permute_by_perm(dout, &inv);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Reshape => {
                // y = reshape(x, new_shape)  →  dx = reshape(dy, x.shape)
                let x = node.saved[0];
                let x_shape = tb(x).shape.clone();
                let dx = reshape_to(dout, &x_shape);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Neg => {
                // y = -x  →  dx = -dC
                let x = node.saved[0];
                let dx = scalar_mul(dout, -1.0);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceSumAxis => {
                // y = sum(x, axis, keepdim)  →  dx = broadcast_to(unsqueeze_if_needed(dout), x.shape)
                let x = node.saved[0];
                let axis = node.meta[0] as usize;
                let keepdim = node.meta[1] != 0;
                let x_shape = tb(x).shape.clone();
                let dout_exp = if keepdim { tensor_retain(dout); dout } else { unsqueeze_at(dout, axis) };
                let dx = broadcast_to_shape(dout_exp, &x_shape);
                tensor_release(dout_exp);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceMeanAxis => {
                // y = mean(x, axis, keepdim)  →  dx = broadcast_to(dout / N, x.shape)
                let x = node.saved[0];
                let axis = node.meta[0] as usize;
                let keepdim = node.meta[1] != 0;
                let x_shape = tb(x).shape.clone();
                let n = x_shape[axis] as f32;
                let dout_scaled = scalar_mul(dout, 1.0 / n);
                let dout_exp = if keepdim {
                    tensor_retain(dout_scaled); dout_scaled
                } else {
                    let u = unsqueeze_at(dout_scaled, axis);
                    tensor_release(dout_scaled);
                    u
                };
                let dx = broadcast_to_shape(dout_exp, &x_shape);
                tensor_release(dout_exp);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceMaxAxis => {
                // y = max(x, axis, keepdim)  →  dx = dout_bc * (x == max_val) mask
                // node.output holds the per-axis max values (retained by tape).
                let x = node.saved[0];
                let out = node.output;
                let axis = node.meta[0] as usize;
                let keepdim = node.meta[1] != 0;
                let x_shape = tb(x).shape.clone();
                let out_exp = if keepdim { tensor_retain(out); out } else { unsqueeze_at(out, axis) };
                let out_bc = broadcast_to_shape(out_exp, &x_shape);
                tensor_release(out_exp);
                let mask = elem_cmp_eq(x, out_bc);
                tensor_release(out_bc);
                let dout_exp = if keepdim { tensor_retain(dout); dout } else { unsqueeze_at(dout, axis) };
                let dout_bc = broadcast_to_shape(dout_exp, &x_shape);
                tensor_release(dout_exp);
                let dx = elem_mul(dout_bc, mask);
                tensor_release(dout_bc);
                tensor_release(mask);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::ReduceVarAxis => {
                // y = var(x, axis, keepdim) population  →  dx = dout * 2 * (x - mean) / N
                let x = node.saved[0];
                let axis = node.meta[0] as usize;
                let keepdim = node.meta[1] != 0;
                let x_shape = tb(x).shape.clone();
                let n = x_shape[axis] as f32;
                // Recompute mean (cold path; keepdim=1 for easy broadcasting).
                let mean_h = tensor_reduce_mean_axis(x, axis as i64, 1);
                let mean_bc = broadcast_to_shape(mean_h, &x_shape);
                tensor_release(mean_h);
                // 2 * (x - mean) / N
                let x_minus_mean = elem_sub(x, mean_bc);
                tensor_release(mean_bc);
                let coeff = 2.0 / n;
                let scaled = scalar_mul(x_minus_mean, coeff);
                tensor_release(x_minus_mean);
                // Expand dout to x.shape
                let dout_exp = if keepdim { tensor_retain(dout); dout } else { unsqueeze_at(dout, axis) };
                let dout_bc = broadcast_to_shape(dout_exp, &x_shape);
                tensor_release(dout_exp);
                let dx = elem_mul(dout_bc, scaled);
                tensor_release(dout_bc);
                tensor_release(scaled);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Softmax => {
                // s = softmax(x, axis)  →  dx = s ⊙ (dout − sum(dout⊙s, axis, keepdim=1))
                let x = node.saved[0];
                let s = node.output;
                let axis = node.meta[0];
                let x_shape = tb(x).shape.clone();
                let dout_s = elem_mul(dout, s);
                let sum_ds = tensor_reduce_sum_axis(dout_s, axis, 1);
                tensor_release(dout_s);
                let sum_bc = broadcast_to_shape(sum_ds, &x_shape);
                tensor_release(sum_ds);
                let diff = elem_sub(dout, sum_bc);
                tensor_release(sum_bc);
                let dx = elem_mul(s, diff);
                tensor_release(diff);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Layernorm => {
                // y = (x − μ) / σ,  σ = sqrt(var + 1e-5)
                // dx = (1/σ) · (dy − mean(dy, axis, k) − y ⊙ mean(dy⊙y, axis, k))
                let x = node.saved[0];
                let var_h = node.saved[1];
                let y = node.output;
                let axis = node.meta[0];
                let x_shape = tb(x).shape.clone();
                // 1/σ (keepdim shape matching var_h)
                let inv_sigma_h = elem_apply(var_h, |v| 1.0 / (v + 1e-5_f32).sqrt());
                let inv_sigma_bc = broadcast_to_shape(inv_sigma_h, &x_shape);
                tensor_release(inv_sigma_h);
                // mean(dy, axis, keepdim=1) broadcast
                let dy_mean_h = tensor_reduce_mean_axis(dout, axis, 1);
                let dy_mean_bc = broadcast_to_shape(dy_mean_h, &x_shape);
                tensor_release(dy_mean_h);
                // mean(dy ⊙ y, axis, keepdim=1) broadcast
                let dy_y = elem_mul(dout, y);
                let dy_y_mean_h = tensor_reduce_mean_axis(dy_y, axis, 1);
                tensor_release(dy_y);
                let dy_y_mean_bc = broadcast_to_shape(dy_y_mean_h, &x_shape);
                tensor_release(dy_y_mean_h);
                // y ⊙ mean(dy⊙y)
                let y_term = elem_mul(y, dy_y_mean_bc);
                tensor_release(dy_y_mean_bc);
                // dy − mean(dy) − y⊙mean(dy⊙y)
                let tmp = elem_sub(dout, dy_mean_bc);
                tensor_release(dy_mean_bc);
                let numer = elem_sub(tmp, y_term);
                tensor_release(tmp);
                tensor_release(y_term);
                let dx = elem_mul(numer, inv_sigma_bc);
                tensor_release(numer);
                tensor_release(inv_sigma_bc);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Gelu => {
                // y = 0.5·x·(1 + tanh(g)),  g = c0·(x + c1·x³)
                // dy/dx = 0.5·(1+t) + 0.5·x·(1−t²)·g',  t = tanh(g),  g' = c0·(1 + 3·c1·x²)
                let x = node.saved[0];
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
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::CrossEntropy => {
                // L = −mean(log(s[i, t[i]])),  s = softmax(logits, axis=1)
                // d_logits[i,j] = dout[0]/N · (s[i,j] − 1{j == t[i]})
                let logits  = node.saved[0];
                let sm_h    = node.saved[1];
                let targets = node.saved[2];
                let n = tb(logits).shape[0];
                let c = tb(logits).shape[1];
                let scale = read(dout)[0] / n as f32;
                let mut grad_data = read(sm_h);
                let tgt_buf = tb(targets).buffer.contents().as_ptr() as *const u8;
                let tgt_dtype = tb(targets).dtype;
                for i in 0..n {
                    let t = read_int_index_tape(tgt_buf, i, tgt_dtype);
                    grad_data[i * c + t] -= 1.0;
                }
                for v in grad_data.iter_mut() { *v *= scale; }
                let logits_shape = tb(logits).shape.clone();
                let dx = tensor_alloc_gpu(0, logits_shape.as_ptr(), logits_shape.len(), grad_data.as_ptr());
                accumulate_grad(&mut grads, logits, dx);
            }
            OpTag::Embedding => {
                // dweight[indices[t]] += dout[t, :] for t in 0..T (scatter-add)
                let weight  = node.saved[0];
                let indices = node.saved[1];
                let vocab_size = tb(weight).shape[0] as i64;
                let dweight = tensor_scatter_add(dout, indices, vocab_size);
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
                            let summed = elem_add(old_grad, new_grad);
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

// ── Integer index reader (for embedding/cross_entropy backward) ───────────────

use crate::metal::Dtype;

fn read_int_index_tape(buf: *const u8, i: usize, dtype: Dtype) -> usize {
    match dtype {
        Dtype::I32 => unsafe { *(buf.add(i * 4) as *const i32) as usize }
        Dtype::I64 => unsafe { *(buf.add(i * 8) as *const i64) as usize }
        _ => panic!("malus: integer index tensor must be Tensor<i32> or Tensor<i64>, got {:?}", dtype),
    }
}

// ── Private CPU math helpers ──────────────────────────────────────────────────
//
// All helpers allocate a new owned tensor; callers release temporaries.
// No GPU work: backward always calls gpu_barrier() first.

fn tb(handle: i64) -> &'static TensorBuffer {
    unsafe { &*(handle as *const TensorBuffer) }
}

fn read(handle: i64) -> Vec<f32> {
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

// ── Grad accumulation helper ──────────────────────────────────────────────────

/// Add new_grad into grads[input_handle].  Takes ownership of new_grad.
fn accumulate_grad(grads: &mut HashMap<i64, i64>, input_handle: i64, new_grad: i64) {
    match grads.remove(&input_handle) {
        None => {
            grads.insert(input_handle, new_grad);
        }
        Some(old) => {
            let summed = elem_add(old, new_grad);
            tensor_release(old);
            tensor_release(new_grad);
            grads.insert(input_handle, summed);
        }
    }
}
