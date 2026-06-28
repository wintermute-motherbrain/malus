use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use crate::metal::{
    gpu_barrier, tensor_alloc_gpu, tensor_alloc_ones_gpu, tensor_alloc_zeros_gpu,
    tensor_matmul, tensor_retain, tensor_release, tensor_transpose, TensorBuffer,
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
    Sum       = 12,
    Transpose = 13,
    Neg       = 14,
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
            _  => panic!("malus: unknown op tag {tag}"),
        }
    }
}

// ── TapeNode ─────────────────────────────────────────────────────────────────
//
// For binops, saved = [a, b] (both retained).
// For unary ops, saved = [x] (input retained); node.output is retained and
// used directly in VJPs that need the forward output (sigmoid, tanh, exp, sqrt).
// tape_clear() releases every handle in saved + output.

struct TapeNode {
    op:     OpTag,
    saved:  Vec<i64>,
    output: i64,
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
        n.borrow_mut().push(TapeNode { op, saved: vec![a, b], output: out });
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
        n.borrow_mut().push(TapeNode { op, saved: vec![x], output: out });
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
    }
    let nodes: Vec<NodeSnap> = NODES.with(|n| {
        n.borrow()
            .iter()
            .map(|node| NodeSnap {
                op:     node.op,
                saved:  node.saved.clone(),
                output: node.output,
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
                // C = A @ B  →  dA = dC @ Bᵀ,  dB = Aᵀ @ dC
                let (a, b) = (node.saved[0], node.saved[1]);
                let bt = tensor_transpose(b);
                let da = tensor_matmul(dout, bt);
                tensor_release(bt);
                let at = tensor_transpose(a);
                let db = tensor_matmul(at, dout);
                tensor_release(at);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Add => {
                // C = A + B  →  dA = dC,  dB = dC
                let (a, b) = (node.saved[0], node.saved[1]);
                tensor_retain(dout);
                tensor_retain(dout);
                accumulate_grad(&mut grads, a, dout);
                accumulate_grad(&mut grads, b, dout);
            }
            OpTag::Sub => {
                // C = A - B  →  dA = dC,  dB = -dC
                let (a, b) = (node.saved[0], node.saved[1]);
                tensor_retain(dout);
                let neg_dout = scalar_mul(dout, -1.0);
                accumulate_grad(&mut grads, a, dout);
                accumulate_grad(&mut grads, b, neg_dout);
            }
            OpTag::Mul => {
                // C = A * B  →  dA = dC * B,  dB = A * dC
                let (a, b) = (node.saved[0], node.saved[1]);
                let da = elem_mul(dout, b);
                let db = elem_mul(a, dout);
                accumulate_grad(&mut grads, a, da);
                accumulate_grad(&mut grads, b, db);
            }
            OpTag::Div => {
                // C = A / B  →  dA = dC / B,  dB = -dC * A / B²
                let (a, b) = (node.saved[0], node.saved[1]);
                let da = elem_div(dout, b);
                let b_sq = elem_mul(b, b);
                let tmp = elem_mul(dout, a);
                let neg_tmp = scalar_mul(tmp, -1.0);
                tensor_release(tmp);
                let db = elem_div(neg_tmp, b_sq);
                tensor_release(neg_tmp);
                tensor_release(b_sq);
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
                    let ptr = tb.buffer.contents() as *const f32;
                    unsafe { *ptr }
                };
                let dx = elem_apply(x, |_| scalar_val);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Transpose => {
                // B = Aᵀ  →  dA = dBᵀ
                let x = node.saved[0];
                let dx = tensor_transpose(dout);
                accumulate_grad(&mut grads, x, dx);
            }
            OpTag::Neg => {
                // y = -x  →  dx = -dC
                let x = node.saved[0];
                let dx = scalar_mul(dout, -1.0);
                accumulate_grad(&mut grads, x, dx);
            }
        }
    }

    // Fold each leaf's transient grad into the persistent registry (accumulate).
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
                            tensor_release(old_grad);
                            tensor_release(new_grad);
                            lg.insert(leaf, summed);
                        }
                    }
                }
            }
        });
    });

    // Release remaining transient grads (intermediates not in LEAVES).
    for (_, grad) in grads {
        tensor_release(grad);
    }

    tape_clear();
}

// ── Test / reset helper (not extern "C"; not JIT-injected) ───────────────────

/// Clear all tape state including the persistent leaf-grad registry and leaves
/// set.  Used by tests and future M15 full-reset paths.  Not JIT-injected.
pub fn tape_reset() {
    LEAVES.with(|l| l.borrow_mut().clear());
    LEAF_GRAD.with(|lg| {
        let mut lg = lg.borrow_mut();
        for (_, grad) in lg.drain() {
            tensor_release(grad);
        }
    });
    tape_clear();
    RECORDING.with(|r| r.set(true));
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
    let ptr = t.buffer.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, t.len).to_vec() }
}

fn alloc_like(template: i64, data: &[f32]) -> i64 {
    let t = tb(template);
    tensor_alloc_gpu(0, t.shape.as_ptr(), t.shape.len(), data.as_ptr())
}

fn elem_add(a: i64, b: i64) -> i64 {
    let (ad, bd) = (read(a), read(b));
    let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect();
    alloc_like(a, &out)
}

fn elem_mul(a: i64, b: i64) -> i64 {
    let (ad, bd) = (read(a), read(b));
    let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x * y).collect();
    alloc_like(a, &out)
}

fn elem_div(a: i64, b: i64) -> i64 {
    let (ad, bd) = (read(a), read(b));
    let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x / y).collect();
    alloc_like(a, &out)
}

fn scalar_mul(a: i64, s: f32) -> i64 {
    let ad = read(a);
    let out: Vec<f32> = ad.iter().map(|x| x * s).collect();
    alloc_like(a, &out)
}

fn elem_apply(a: i64, f: impl Fn(f32) -> f32) -> i64 {
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
