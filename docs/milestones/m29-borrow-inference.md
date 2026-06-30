# M29 — Borrow-Inference RC + Benchmark

**Crates:** `malus-sema`  
**Track:** frontend  
**Depends on:** M28 (all prior gates must be green)

Implement the Lobster-style borrow-inference pass: assign a single owner per tensor allocation; demote all other uses to zero-cost borrows; emit RC ops only for the ~5% of allocations that genuinely escape their creation scope (primarily tape-saved tensors). Then benchmark against f32 PyTorch-MPS to establish the V4 performance number. See ADR-0026.

## Done-When

1. RC-op-count gate: `retain_count + release_count ≤ 0.05 × alloc_count` over a nanoGPT train step (measured via counters analogous to the CPU-compute counter, added to `malus-runtime`).
2. nanoGPT trains end-to-end and all prior CI gates remain green (M26 full-step CPU-counter==0, M27 zero-Variable IR, M28 no-unroll lint, gradient_check within tol).
3. Benchmark result documented: nanoGPT step time on the target M-series Mac, compared to f32 PyTorch-MPS nanoGPT. The Nx ratio is the V4 performance number.
4. `cargo test --workspace` passes.

## Scope

### 1. Borrow-inference pass (`malus-sema/src/borrow_inference.rs`)

A new sema pass running after grad-inference (M27) and before CTMM drop-insertion.

**Algorithm (Lobster-inspired):**

For each function body, perform a single top-down AST walk:

1. **Owner assignment.** The first binding a fresh allocation reaches is the owner. A "fresh allocation" is any expression that produces a new heap object: `Tensor.gpu<f32>([...])`, kernel dispatch result, matmul result, etc.

2. **Borrow detection.** Any subsequent binding that receives the same handle (e.g. `let b = a` or passing `a` to a function that returns it unchanged) is a borrow of `a`. Mark the binding in `borrow_set: HashSet<BindingId>`.

3. **Escape detection.** A binding escapes if it is:
   - Returned from the function.
   - Passed to `tape_record_*` (i.e., it is in the grad-inference escape set from M27).
   - Stored into a struct field or list element that itself escapes (transitively).
   
   If the owner escapes, it remains RC-managed. Borrows of an escaping owner are still borrows (no extra RC).

4. **RC reduction.** For any binding in `borrow_set` (and not itself an escaping owner), do not emit `tensor_retain`/`tensor_release`. The existing `ctmm.rs` path emits `Drop` (static free) for them instead.

**Correctness invariant.** A wrong borrow = use-after-free. Before marking a binding as a borrow, verify:
- There is no path where the borrow outlives the owner's drop point.
- The function does not move the "borrow" into a container that escapes.

For V4, use a conservative criterion: a binding is a borrow only if (a) its source binding is in the same function scope AND (b) its last use occurs before the source's last use. If either condition is uncertain, treat it as an owner (conservative: always correct, sometimes suboptimal).

### 2. RC-op counters (`malus-runtime/src/lib.rs`)

Analogous to the CPU-compute counter:

```rust
static RETAIN_COUNT: AtomicI64 = AtomicI64::new(0);
static RELEASE_COUNT: AtomicI64 = AtomicI64::new(0);
static ALLOC_COUNT: AtomicI64 = AtomicI64::new(0);
```

Increment at the entry of `tensor_retain`, `tensor_release`, and `tensor_alloc_gpu` (and `tensor_alloc_zeros_gpu`, `tensor_alloc_ones_gpu`). Export `malus_rc_counts() -> (i64, i64, i64)` and `malus_rc_reset()`.

### 3. RC-op-count CI gate

```rust
#[test]
fn test_v4_m6_rc_ratio() {
    malus_rc_reset();
    run_metal_src(include_str!("../../../examples/nanogpt.ml")); // one train step
    let (retains, releases, allocs) = malus_rc_counts();
    let rc_ops = retains + releases;
    assert!(
        rc_ops as f64 <= 0.05 * allocs as f64,
        "RC overhead too high: {rc_ops} RC ops on {allocs} allocs ({:.1}%)",
        rc_ops as f64 / allocs as f64 * 100.0
    );
}
```

### 4. Benchmark

Write `bench/nanogpt_step.sh` (or a Rust bench):
1. Run 20 nanoGPT train steps, median step time (wall-clock, `std::time::Instant`).
2. Run equivalent PyTorch-MPS script (`bench/nanogpt_pytorch.py`, f32, same model config) on the same machine.
3. Compute ratio. Document in `docs/milestones/m29-benchmark-results.md` with: machine (chip, memory), malus config, PyTorch version, step times, ratio.

The Nx ratio is informational at V4 cutoff — there is no hard pass/fail threshold at this milestone since it is set empirically (see V4 plan). If ratio > 3×, investigate before declaring V4 done.

### 5. Safety validation

For each function in the test suite, verify no use-after-free via:
- Run all existing tests under `ASAN` (`RUSTFLAGS="-Z sanitizer=address" cargo test`). The borrow-inference pass is the highest-risk change in V4; ASAN must be clean.
- The `gradient_check.ml` test exercises many borrow/owner patterns through the tape path; if the escape set is wrong, gradients will be NaN or wrong, caught by the 1e-3 tolerance check.

## Out of Scope

- Per-call-site function specialization on ownership kind (Lobster's full mechanism for optimizing RC at call boundaries) — evaluate post-V4 after measuring the Nx ratio. The conservative V4 approach may already hit the target.
- Borrow inference across function boundaries (interprocedural) — V4 is intraprocedural only. Cross-function borrow is deferred; it requires a fixed-point analysis over the call graph.
- Eliminating `tensor_retain`/`tensor_release` from the runtime ABI entirely — they remain for the ~5% RC-surviving cases.
