# malus V1 Plan

## What V1 Is For

V1 is a research prototyping tool for ML on Apple Silicon. The North Star question is: **could someone build a micrograd-style autograd engine on top of this?**

V1 does not include autograd, gradient tape, or an optimizer. The done-when is a hand-written forward AND backward pass for a two-layer MLP — the user writes both directions manually, which proves the language is expressive enough to eventually build a tape on top.

## V1 Done-When Program

`examples/mlp.ml` runs correctly on an M-series Mac, printing decreasing loss over 10 training steps:

```malus
struct MLP:
    w1: Tensor<f32>
    w2: Tensor<f32>

kernel relu_backward(grad_out: Tensor<f32>, x: Tensor<f32>) -> Tensor<f32>:
    let mask = x > 0.0
    return grad_out * mask

fn forward(x: Tensor<f32>, model: MLP) -> Tensor<f32>:
    let h = relu(x @ model.w1)
    return h @ model.w2

fn backward(x: Tensor<f32>, model: MLP, grad_output: Tensor<f32>) -> MLP:
    let h_pre = x @ model.w1
    let h = relu(h_pre)
    let dw2 = transpose(h) @ grad_output
    let dh = grad_output @ transpose(model.w2)
    let dh_pre = relu_backward(dh, h_pre)
    let dw1 = transpose(x) @ dh_pre
    return MLP(w1=dw1, w2=dw2)

fn main():
    let mut model = MLP(w1=ones(3, 4), w2=ones(4, 2))
    let x = ones(2, 3)
    let target = zeros(2, 2)
    let lr = 0.01

    for step in range(10):
        let out = forward(x, model)
        let diff = out - target
        let loss = sum(diff)
        println("step {}: loss = {}", step, loss)
        let grads = backward(x, model, diff)
        model = MLP(
            w1=model.w1 - lr * grads.w1,
            w2=model.w2 - lr * grads.w2
        )

    println("final output: {}", forward(x, model))
```

## Milestone Sequence

V1 is five sequential milestones. Each has a standalone done-when program that can be run independently to verify the milestone.

| Milestone | Theme | Key Features |
|---|---|---|
| [M7](./m7-kernel-thickening.md) | Kernel Thickening | multi-statement kernels, `let mut`, scalar broadcasting |
| [M8](./m8-core-stdlib.md) | Math Layer | matmul (`@`), relu/sigmoid/tanh/exp/log/sqrt/abs, transpose, zeros/ones, sum, shape metadata |
| [M9](./m9-control-flow.md) | Control Flow | if/else, for, while, CTMM RC fallback for conditional tensor lifetimes |
| [M10](./m10-structs-enums.md) | Structs + Enums | struct decl/construction/field access, data-carrying enums, match |
| [M11](./m11-mlp.md) | The MLP | fixed-length arrays, rich diagnostics, 2-layer MLP integration |

## Design Decisions

These decisions were made during the V1 planning pass. Future changes to these should produce an ADR.

| Decision | Choice | Rationale |
|---|---|---|
| CTMM for conditional paths | RC fallback (not dataflow liveness) | ADR-0002 already specifies RC as the fallback for structurally ambiguous lifetimes. Dataflow liveness analysis is a V2 optimization that reduces how often you fall back to RC. |
| Mutation | `let mut` + reassignment (`x = new_val`) | Shadowing (`let x = x + delta`) breaks in loops — the shadow is scoped to the loop body and dies each iteration. `let mut` is CTMM-friendly: reassignment = drop-old + bind-new. No aliasing risk because CTMM enforces move semantics. |
| Kernel body expressiveness | Let bindings + comparisons + ternary/select | Enough for all gradient kernels (relu_backward, sigmoid_backward, etc.). Loops inside kernels need `@threadgroup_size` and shared memory — deferred with kernel annotations. |
| Enum scope | Data-carrying variants + exhaustive match, no generics | Tag-only enums aren't expressive enough for activation dispatch; full generics would require a type system overhaul. No `Option<T>` in V1. |
| Dynamic collections | Fixed-length arrays only | Growable `Vec<T>` deferred to V1.1. The MLP done-when needs arrays for layer iteration, but size is known at compile time. CTMM can reason statically about fixed arrays. |
| Stdlib scope | Core math (~12 functions) | Once the element-wise kernel synthesis infrastructure exists (M5.1), each unary math op is a one-liner kernel — near-zero marginal cost. |

## What V1 Does NOT Include

Deferred to post-V1:

- SafeTensors / NumPy file I/O
- Terminal REPL
- Import aliasing (`as` syntax — `import ops as o`)
- Kernel annotations (`@threadgroup_size`, `@shared_memory`)
- GPU intrinsics (`thread_id()`, `simd_shuffle()`, etc.)
- `inout` kernel parameters
- GPU RNG (Philox)
- `Option<T>` and generics
- Growable `Vec<T>`
- NumPy-style shape broadcasting (e.g. `[3,1] * [1,4]` → `[3,4]`)
- Non-f32 dtypes (f16, bf16, int types)
- Loops and conditionals inside kernel bodies
- autograd / gradient tape
- nanoGPT or other full model implementations
