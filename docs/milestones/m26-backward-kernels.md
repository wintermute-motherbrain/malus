# M26 — Backward Kernels

**Crates:** `malus-runtime`  
**Track:** GPU  
**Depends on:** M25

Rewrite every VJP in `tape.rs::backward` to dispatch a GPU backward kernel instead of executing Rust CPU arithmetic. The full nanoGPT train step (forward + backward + optimizer) must complete with zero CPU compute. This is the **canonical V4 milestone gate**. See ADR-0031.

## Done-When

1. `examples/nanogpt.ml` runs a full train step (forward + `backward(loss)` + AdamW update) with **`malus_cpu_compute_count() == 0`** — the canonical V4 north-star gate.
2. `gradient_check.ml` passes: numerical finite-difference gradients match `backward()` gradients within 1e-3 for all ops exercised by the nanoGPT forward pass (matmul, softmax, layernorm, gelu, embedding, cross-entropy, elementwise add/mul).
3. Training loss decreases monotonically over 10 steps with a fixed random seed (regression test for gradient correctness).
4. `cargo test --workspace` passes.

## Scope

### 1. Backward kernel `.ml` files

Author one backward kernel per forward op. Each kernel takes the saved forward inputs/outputs and the output gradient, and produces input gradient(s). File naming: `stdlib/backward/softmax_bwd.ml`, etc.

| Backward kernel | VJP formula | Notes |
|---|---|---|
| `softmax_bwd` | `dx = s ⊙ (dout − sum(dout ⊙ s, axis, keepdim=true))` | `s` = forward output (saved on tape) |
| `layernorm_bwd` | standard layernorm VJP (dx = dout/σ − mean(dout/σ) − x_norm·mean(dout/σ · x_norm)) | `x_norm` = normalized values saved on tape |
| `gelu_bwd` | `dx = dout · gelu'(x)` where `gelu'` is tanh-approx derivative | `x` = forward input |
| `cross_entropy_bwd` | `d_logits[i,j] = dout/N · (s[i,j] − 1{j==target[i]})` | `s` = softmax output, `targets` = integer index tensor |
| `embedding_bwd` | scatter-add: `dweight[indices[t]] += dout[t]` for each `t` | Atomic adds on GPU (use `atomic_fetch_add_explicit`) |
| `reduce_sum_bwd` | broadcast dout back to input shape | `keepdim` handled by reshape before broadcast |
| `reduce_mean_bwd` | `dx = dout / N` broadcast | N = size of reduced axis |
| `broadcast_add_bwd` | sum dout over broadcast dimensions | |
| `broadcast_mul_bwd` | `dx = dout · y`, `dy = dout · x` (then reduce over broadcast dims) | |
| `reshape_bwd` | reshape dout to input shape | Zero-copy reshape (same MTLBuffer) |
| `matmul_bwd` | `dA = dC @ B^T`, `dB = A^T @ dC` | Uses MPS matmul (already handled by existing tape.rs VJP code) |

### 2. Rewrite `tape.rs::backward` (`malus-runtime/src/tape.rs`)

For each `OpTag` variant, replace the Rust CPU loop body with a `kernel_dispatch_v2` call to the corresponding backward kernel ID. The `TapeNode` must carry the backward kernel ID alongside the saved forward tensors (set at record time, looked up from the `builtin_to_kernel_id` map).

Pattern:
```rust
OpTag::Softmax { axis } => {
    // old: CPU loop computing s ⊙ (dout - sum(dout⊙s))
    // new:
    kernel_dispatch_v2(
        SOFTMAX_BWD_KERNEL_ID,
        &[saved_output_handle, dout_handle],
        /* uniforms */ &SoftmaxBwdUniforms { axis, rows, cols },
    )
}
```

Remove the `gpu_barrier()` calls that currently precede each CPU read in the VJP bodies — they are there only because the CPU fns read MTLBuffer data directly. GPU-dispatched kernels do not need them (the barrier is implicit at command-buffer commit time).

### 3. Tape node changes

`TapeNode` gains a `bwd_kernel_id: u64` field. At tape-record time (in `tape_record_*` calls from codegen-cpu), look up the kernel ID from a `&HashMap<BuiltinOp, u64>` passed to the runtime at init. This map is produced by `compile_kernels` over the stdlib backward kernels.

Alternatively, embed the backward kernel ID into the `RuntimeSymbols` struct as a static map. Choose whichever avoids adding a new runtime init parameter; the `builtin_to_kernel_id` map from M25 already exists.

### 4. `gradient_check.ml` update

Extend `examples/gradient_check.ml` to cover all ops exercised by nanoGPT: softmax, layernorm, gelu, cross-entropy, embedding, batched matmul, broadcasting add/mul, sum/mean over axis, reshape.

For each op: compute the analytic gradient via `backward()`, compute the numerical gradient via finite differences (using `no_grad`), assert max-abs-diff < 1e-3.

### 5. CI tests

```rust
#[test]
fn test_v4_m3_full_step_zero_cpu_compute() {
    malus_cpu_compute_reset();
    run_metal_src(include_str!("../../../examples/nanogpt.ml")); // full train step
    assert_eq!(malus_cpu_compute_count(), 0,
        "CPU arithmetic invoked during nanoGPT train step — V4 canonical gate");
}

#[test]
fn test_gradient_check_all_ops() {
    run_metal_src(include_str!("../../../examples/gradient_check.ml"));
    // gradient_check.ml panics if any gradient is out of tolerance
}

#[test]
fn test_nanogpt_loss_decreases() {
    // run 10 steps, assert loss[9] < loss[0]
}
```

## Out of Scope

- `Variable elimination (M27).
- AdamW backward (the optimizer update uses `no_grad` — no backward kernel needed there).
- Higher-order gradients / `custom_grad`.
- Gradient checkpointing (rematerialization).
