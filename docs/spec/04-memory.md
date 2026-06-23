# 04 — Memory Model (Lobster)

## Overview

malus uses **Lobster** — Automatic Compile-Time Memory Management (CTMM) — as its memory model. Lobster uses escape analysis to insert static `free` calls for tensors at compile time. It falls back to reference counting (RC) only when a tensor's lifetime is structurally ambiguous.

The goal: the fast path (linear tensor flows through `fn` and `kernel` calls) is allocation-free at runtime. RC is a correctness fallback for cold-path patterns like model parameter storage.

## Escape analysis

The compiler performs escape analysis on every tensor binding after type checking. A tensor **escapes** if it:

- Is returned from the function
- Is stored in a struct field or heap-allocated container
- Is captured by a closure

If a tensor does not escape, its lifetime is fully determined statically. The compiler inserts a `free` call immediately after its last use.

### Example — static management

```malus
fn compute(x: Tensor<f32>) -> Tensor<f32>:
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0])   # does not escape
    let b = a + x                                # a's last use
    # compiler inserts: free(a) here
    return b                                     # b escapes (returned)
```

`a` is freed at compile time. `b` is owned by the caller.

## RC fallback

When a tensor's lifetime cannot be resolved statically, Lobster falls back to reference counting. The RC boundary is triggered when a tensor is:

1. **Stored in a heap-allocated container** — a struct field, or an element of a dynamic array
2. **Captured by a closure** — when closures are added in a future version

In practice, model weight tensors (stored in a `struct` or list) will use RC. Forward-pass computations will not. This is the intended split: cold-path (parameter storage) uses RC; hot-path (math) is static.

```malus
struct Model:
    weights: Tensor<f32>   # this tensor uses RC — lifetime is ambiguous
    bias: Tensor<f32>      # same
```

The programmer does not choose between static and RC — Lobster selects automatically based on the structural analysis.

## GPU boundary

Kernels execute asynchronously. A tensor passed to a `kernel` call is **in-flight** until the GPU completes execution. The compiler handles this as follows:

1. At the `kernel` call site, the tensor is marked **in-flight**
2. Its Lobster-determined free point is preserved
3. Before the `free` call is emitted, the compiler inserts a **GPU barrier** (`gpu_barrier()`)
4. The barrier blocks the CPU until the Metal command buffer completes
5. Only then is `free` emitted

This guarantees the GPU never reads a freed buffer, while preserving maximum CPU/GPU overlap — the CPU continues executing the `fn` body after dispatching the kernel, only blocking when it actually needs to free the tensor.

### Example — in-flight tensor

```malus
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)   # a and b are now in-flight
    print(c)            # c's last use; CPU can do other work here
    # compiler inserts: gpu_barrier()
    # compiler inserts: free(a), free(b), free(c)
```

## Kernel ownership semantics

- **Inputs** are borrowed immutably by default — the caller retains ownership, and the compiler knows the tensor is still alive after the kernel returns
- **Outputs** are new owned tensors — the caller receives ownership and Lobster tracks the new tensor from that point
- **`inout` parameters** `[v1]` — the tensor is mutated in-place; the same buffer is reused. Lobster knows no new allocation occurs and no free is needed for the input buffer

```malus
kernel scale(inout a: Tensor<f32>, factor: f32) -> None:
    a = a * factor   # mutates a in-place; no output buffer allocated
```

## Summary of free-point rules

| Situation | Lobster action |
|---|---|
| Tensor not escaped, last use in `fn` body | `free` after last use |
| Tensor passed to kernel (in-flight) | `gpu_barrier()` then `free` after last use |
| Tensor returned from function | caller owns it; analysis continues in caller |
| Tensor stored in struct field or container | RC manages lifetime |
| `inout` kernel parameter | no free inserted for the input; output is the same buffer |
