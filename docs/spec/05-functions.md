# 05 — Functions and Kernels

## Host functions (`fn`) `[MVP]`

A `fn` defines a CPU-side function. It is JIT-compiled via Cranelift and runs on the host CPU. `fn` bodies orchestrate data flow: they create tensors, call kernels, read results, and handle I/O.

```malus
fn name(param: Type, ...) -> ReturnType:
    stmt
    ...
```

- Parameters and return type are explicitly annotated
- If return type is omitted, it defaults to `None` (no return value)
- `fn` bodies may call other `fn`s and `kernel`s
- `fn` bodies may not use GPU intrinsics (`threadgroup_id()`, `shared_alloc()`, etc.)

## Device kernels (`kernel`) `[MVP]`

A `kernel` defines a GPU-side function. It is compiled to Metal Shading Language (MSL) and JIT-compiled by the Apple Metal driver at startup. Kernels execute in parallel across GPU threads.

```malus
kernel name(param: Tensor<dtype>, ...) -> Tensor<dtype>:
    stmt
    ...
```

- Return type is required and must be a tensor type
- Kernel parameters are borrowed immutably by default (see section 04)
- `inout` parameters `[v1]` allow in-place mutation

### Calling a kernel from a `fn`

```malus
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)   # dispatched asynchronously to the GPU
    print(c)
```

When a `fn` calls a `kernel`, the compiler:
1. Allocates an output buffer of the correct shape and dtype
2. Encodes the compute command into the Metal command buffer
3. Does NOT block — the GPU executes asynchronously while the `fn` continues
4. Inserts a `gpu_barrier()` before any freed in-flight tensor (see section 04)

### Kernel control flow `[MVP]`

Kernels support full control flow: `if`/`else`, `for`, `while`. There are no restrictions on branching inside kernels. SIMT divergence is the programmer's responsibility — the compiler does not warn or restrict.

## Tensor placement and transfers

Every tensor has a placement: `cpu` or `gpu`. Placement is set at creation:

```malus
let a = Tensor.gpu<f32>([1.0, 2.0])   # GPU placement
let b = Tensor.cpu<f32>([1.0, 2.0])   # CPU placement
```

When a tensor crosses the `fn`/`kernel` boundary in the wrong direction, the compiler inserts a transfer automatically. On Apple Silicon, this is implemented with `MTLResourceStorageModeShared` buffers and sync barriers — no physical copy occurs.

A CPU-placement tensor passed to a `kernel` causes the compiler to emit a sync barrier ensuring the CPU has finished writing before the GPU reads. A GPU-placement tensor read in a `fn` body after a kernel emits a barrier ensuring the GPU has finished writing.

The programmer always sees where a tensor was created (its intended placement) but never writes explicit transfer code.

## Kernel annotations `[v1]`

Annotations on the line immediately before a `kernel` declaration set its static launch configuration:

```malus
@threadgroup_size(16, 16)
@shared_memory(tile_a: f32[16][16], tile_b: f32[16][16])
kernel tiled_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    ...
```

- `@threadgroup_size(x, y, z)` — sets the Metal threadgroup dimensions. Defaults to `(min(n, maxThreadsPerThreadgroup), 1, 1)` if omitted.
- `@shared_memory(name: dtype[dim]...)` — allocates threadgroup-local shared memory. The named allocation is accessible inside the kernel body.

Annotations are compile-time only. They cannot reference runtime values.

## GPU intrinsics `[v1]`

Inside kernel bodies, the following built-in functions expose GPU-level concepts:

| Intrinsic | Returns | Description |
|---|---|---|
| `thread_id()` | `u32` | Thread index within the grid (1D) |
| `thread_id_2d()` | `(u32, u32)` | Thread index (x, y) for 2D kernels |
| `threadgroup_id()` | `u32` | Threadgroup index within the grid |
| `thread_in_group()` | `u32` | Thread index within the threadgroup |
| `shared_alloc<T>(n)` | `Buffer<T>` | Allocate `n` elements in threadgroup shared memory |
| `simd_shuffle(val, lane)` | same as `val` | Shuffle value across SIMD lanes |
| `simd_sum(val)` | same as `val` | Reduction across SIMD group |

Intrinsics are only valid inside `kernel` bodies. Using them in `fn` bodies is a compile-time error.

## `inout` parameters `[v1]`

An `inout` kernel parameter is mutated in-place rather than producing a new output buffer. This is important for performance-critical element-wise ops (like ReLU, dropout) where allocating a new buffer is wasteful.

```malus
kernel relu(inout a: Tensor<f32>):
    a = max(a, 0.0)
```

Lobster does not insert a `free` for an `inout` parameter — the same buffer is reused. The caller retains ownership; the buffer is freed according to the caller's escape analysis.

## Entry point

The entry point for script execution is `fn main()` with no parameters and no return type. It is an error for a script to have no `main`.
