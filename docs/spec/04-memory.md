# 04 — Memory Model (CTMM)

## Overview

malus uses **CTMM** (Compile-Time Memory Management) as its memory model. The goal: tensor lifetimes are determined statically at compile time wherever possible; reference counting (RC) is the fallback only for tensors that genuinely escape their creation scope — chiefly tensors saved onto the autograd tape.

The programmer shares tensors freely with no annotations. The compiler infers ownership vs. borrow at each use site.

**Reference model:** the Lobster programming language's ownership model (https://aardappel.github.io/lobster/memory_management.html). See ADR-0026.

## Ownership and borrow inference

The compiler assigns a single **owner** to every tensor allocation — the first binding, struct field, or array element the allocation is assigned to. Every other use of the same allocation is a **borrow**: a reference with no refcount cost (`tensor_retain`/`tensor_release` are not emitted for borrows).

This means:

```malus
fn compute(x: Tensor<f32>) -> Tensor<f32>:
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0])   # a owns this allocation
    let b = a + x                                # a is borrowed (used but not re-owned)
    # compiler inserts: free(a) here — last use of the owner
    return b                                     # b escapes (returned); caller owns it
```

`a` is freed at compile time. `b` is owned by the caller.

## RC fallback: escape analysis

When ownership analysis determines that an allocation **escapes** its creation scope, its lifetime cannot be resolved statically and RC is required. An allocation escapes if it:

- Is saved onto the autograd tape for use in `backward()` — the tape-save extends the lifetime past the source scope.
- Is stored in a struct field or container that itself escapes (transitively).
- Is returned from a function through a path where the caller's borrow-scope is unknown.

**The autograd tape is the canonical and primary RC use case.** Tensors that are not grad-tracked — or grad-tracked but never escape to the tape — receive static-free identical to plain tensors.

```malus
# grad-tracked tensors that escape the tape use RC
let w = variable(randn(128, 64))   # variable() marks this as a grad leaf
let loss = cross_entropy(model(w, x), y)
backward(loss)   # tape holds saved tensors for backward; they get RC

# tensors that don't touch the tape get static-free regardless of grad-tracking
with no_grad:
    let logits = model(w, x)      # free(logits) after this scope
```

## Static grad-inference

Grad-tracking is a **statically-inferred property** computed by the sema pass, not a distinct type. A tensor binding is grad-tracked if:

1. It derives from a leaf created by `variable(...)`, **and**
2. It is not inside a `no_grad` scope.

This property drives:
- Which tensors emit `tape_record_*` calls (codegen-cpu).
- Which tensors are escape-analyzed for RC (ctmm).
- Which tensors carry a `.grad` slot (leaves only).

There is **one tensor type: `Tensor<dtype>`**. There is no `Variable` type. See ADR-0030.

## GPU boundary

Kernels execute asynchronously. A tensor passed to a kernel call is **in-flight** until the GPU completes. The compiler handles this as follows:

1. At the kernel call site, the tensor is marked in-flight.
2. The CTMM-determined free point is preserved.
3. Before the `free` call is emitted, the compiler inserts a **GPU barrier** (`gpu_barrier()`).
4. The barrier blocks the CPU until the Metal command buffer completes.
5. Only then is `free` emitted.

This guarantees the GPU never reads a freed buffer while preserving maximum CPU/GPU overlap.

## Kernel ownership semantics

- **Inputs** are borrowed immutably — the caller retains ownership; the compiler knows the tensor is still alive after the kernel returns.
- **Outputs** are new owned tensors — the caller receives ownership; CTMM tracks the new tensor from that point.

## Summary of free-point rules

| Situation | CTMM action |
|---|---|
| Tensor not escaped, last use in `fn` body | `free` after last use |
| Tensor passed to kernel (in-flight) | `gpu_barrier()` then `free` after last use |
| Tensor returned from function | caller owns it; analysis continues in caller |
| Tensor saved onto autograd tape | RC — `tensor_retain` on save; `tensor_release` in VJP after use |
| Tensor in grad-tracked path but not tape-escaping | static-free (same as non-grad tensor) |
| Tensor in `no_grad` scope | static-free unconditionally |
