# malus V2 Plan — Autograd

## What V2 Is For

V2 answers the question V1 proved was *askable*: **can you remove the hand-written backward pass?**

V2 adds a define-by-run gradient tape to malus. The tape records each differentiable forward op on `Variable` values; a single `backward(loss)` call walks it in reverse and accumulates gradients automatically. The done-when replaces every line of manual gradient math in `examples/xor.ml` with that one call.

V2 also closes three deferred V1 bugs: the enum-payload match-binding use-after-free, the zero-length tensor dispatch crash, and missing `break`/`continue` support.

## V2 Done-When Program

`examples/mlp_autograd.ml` runs correctly on an M-series Mac, printing decreasing loss over 10,000 training steps with final predictions rounding to `0, 1, 1, 0` — identical convergence to `examples/xor.ml` but with the manual backward pass deleted:

```malus
fn main():
    let x    = tensor.gpu<f32>([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    let tgt  = tensor.gpu<f32>([[0.0], [1.0], [1.0], [0.0]])
    let ones41 = ones(4, 1)
    let lr = 1.5

    let mut w1 = variable(tensor.gpu<f32>([[0.3, -0.2, 0.5, -0.1, 0.4, -0.3, 0.2, -0.4],
                                            [-0.1, 0.4, -0.3, 0.2, -0.4, 0.3, -0.2, 0.1]]))
    let mut b1 = variable(zeros(1, 8))
    let mut w2 = variable(tensor.gpu<f32>([[0.3], [-0.2], [0.5], [-0.1],
                                            [0.4], [-0.3], [0.2], [-0.4]]))
    let mut b2 = variable(zeros(1, 1))

    for step in range(10000):
        let z1  = variable(x) @ w1 + variable(ones41) @ b1
        let h   = sigmoid(z1)
        let z2  = h @ w2 + variable(ones41) @ b2
        let out = sigmoid(z2)
        let diff = out - variable(tgt)
        let loss = sum(diff * diff)

        zero_grad(w1, b1, w2, b2)
        backward(loss)

        with no_grad:
            w1 = variable(w1.data - lr * w1.grad)
            b1 = variable(b1.data - lr * b1.grad)
            w2 = variable(w2.data - lr * w2.grad)
            b2 = variable(b2.data - lr * b2.grad)

        if step == 0 or step == 500 or step == 9999:
            println("step {}: loss = {}", step, loss.data)
```

## Milestone Sequence

V2 is four sequential milestones. Each has a standalone done-when program.

| Milestone | Theme | Key Features |
|---|---|---|
| [M12](./m12-hardening.md) | Hardening | enum-payload retain-on-escape, zero-length dispatch guard, `break`/`continue` |
| [M13](./m13-variable-type.md) | The `Variable` Type | `Variable<f32>` type form, type-directed RC in CTMM, dormant retain/release ABI activated |
| [M14](./m14-tape-backward.md) | The Tape + `backward()` | global thread-local tape, reverse walk, VJPs for all V1 ops, `no_grad` scope |
| [M15](./m15-differentiable-stdlib.md) | Differentiable Stdlib + Capstone | `.grad`, `zero_grad`, `variable()`, V2 capstone |

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Autograd architecture | Define-by-run runtime tape | Literal micrograd north-star; one VJP per op; activates the RC machinery already wired into the ABI/IR since M9/M10. See ADR-0015. |
| Grad typing | Distinct `Variable` type | CTMM is compile-time; needs a type-directed signal to choose static `free` vs RC `release`. Plain `Tensor` keeps static `free` everywhere; only `Variable` is RC-managed. See ADR-0016. |
| Tape control | Global thread-local tape + scoped `no_grad` | micrograd/PyTorch ergonomics; no viral tape-threading through every `fn` signature; fits the runtime's existing `OnceLock<MetalContext>` global-context pattern. |
| VJP authorship | Built-in Rust VJPs; `custom_grad` deferred | Every op on the path to nanoGPT has a known analytic backward. Nothing on the critical path requires user-defined gradients. |
| CTMM/RC scope | Type-directed RC on `Variable` only | Correctness comes from the type, not a new analysis. General dataflow-liveness RC fallback stays deferred. |
| `.grad` type | Plain `Tensor<f32>` | No double-backward in V2. Keeps gradient tensors outside the tape and eligible for static Drop. |
| Tape clearing | Auto-clear after `backward` | PyTorch `retain_graph=False` default. Prevents unbounded tape growth across training steps. |
| `Variable` name | `Variable` | Most recognizable autodiff term (PyTorch, autograd literature). Capital-V type reads distinctly from lowercase program variables. See CONTEXT.md `### Autograd`. |

## What V2 Does NOT Include

Deferred to V3 or later:

- User-definable custom gradient hooks (`custom_grad`)
- Second-order / higher-order gradients (double-backward)
- Gradient checkpointing
- General dataflow-liveness RC fallback (post-V3)
- NumPy-style shape broadcasting (M16)
- Axis reductions with `keepdim` / `mean` / `var` (M16)
- `reshape` / `view`, batched/3-D matmul (M17)
- Transformer stdlib: softmax, layernorm, GELU, cross-entropy (M18)
- Embeddings, gather, index tensors, random init (M19)
- Lvalue assignment targets (`s.field = e`, `a[i] = e`) / AdamW (M20)
- MPS-accelerated matmul (M21)
- File I/O and data loading (M22)
- Full non-f32 dtype compute (f16, bf16)
