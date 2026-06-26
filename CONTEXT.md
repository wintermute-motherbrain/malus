# malus

A lightweight, high-performance DSL compiled with Rust for ML workloads on Apple Silicon.

## Language

### Compilation

**Host function (`fn`)**:
A CPU-side function JIT-compiled via Cranelift. Orchestrates data flow, calls kernels, handles I/O.
_Avoid_: CPU function, host code

**Kernel (`kernel`)**:
A GPU-side function compiled to Metal Shading Language (MSL) and JIT-compiled by the Metal driver. Operates on tensors in parallel across GPU threads.
_Avoid_: Shader, compute shader, device function

**Dual-pipeline compilation**:
The model where `fn` and `kernel` are compiled through completely separate backends (Cranelift and Metal) but share a common frontend (syntax, type system, semantic analysis).
_Avoid_: Split compilation, two-stage compilation

### Memory

**CTMM** (Compile-Time Memory Management):
malus's automatic compile-time memory management system. Uses escape analysis to insert static `free` calls for tensors. Falls back to reference counting only when lifetimes are structurally ambiguous (heap-stored or closure-captured tensors). Inspired by the Lobster language's ownership model.
_Avoid_: GC, garbage collector, allocator, Lobster

**Escape analysis**:
The compiler pass that determines whether a tensor's lifetime can be fully resolved at compile time by tracking where it is created, used, and last referenced.

**Structurally ambiguous**:
A tensor lifetime that cannot be resolved at compile time — specifically, when a tensor is stored in a heap-allocated container (struct field, dynamic array) or captured by a closure.

### Tensors

**Tensor**:
The core built-in primitive type. Parameterized by dtype (`Tensor<f32>`), with dynamic shape. Not a library type — the compiler has deep knowledge of tensors for memory management and codegen.
_Avoid_: Array, ndarray, matrix

**Placement**:
Whether a tensor is logically associated with CPU or GPU. Explicit at creation, but transfers at the `fn`/`kernel` boundary are inserted automatically by the compiler.
_Avoid_: Device, location

**In-flight tensor**:
A tensor that has been passed to a kernel whose GPU work has not yet been committed. The compiler inserts a GPU barrier before any CPU-side access (free, read, or return) of an in-flight tensor.
_Avoid_: Pending input, GPU-active tensor

**Pending tensor**:
A tensor produced by a kernel whose GPU work has not yet been committed. CPU reads of a pending tensor return stale data unless preceded by a GPU barrier.
_Avoid_: Pending output, uncommitted tensor

**Pending set**:
The CTMM compile-time set of tensor bindings produced or consumed by a GPU-producing expression (`KernelCall`, tensor `BinOp`, or GPU-returning `Call`) since the last `GpuBarrier`. Any CPU-side access of a binding in the pending set triggers a barrier insertion.
_Avoid_: GPU-active set

### Kernel mechanics

**Launch configuration**:
Static properties of a kernel dispatch — threadgroup size, grid dimensions — set via annotations on the kernel function.
_Avoid_: Dispatch config, thread config

**Intrinsics**:
Built-in functions callable inside kernel bodies that expose GPU-level concepts: thread identity (`threadgroup_id()`), shared memory (`shared_alloc()`), SIMD operations (`simd_shuffle()`).
_Avoid_: Builtins, GPU functions

**`inout` parameter**:
A kernel parameter that is mutated in place rather than borrowed immutably. Avoids allocating a new output buffer for element-wise ops.
_Avoid_: Mutable parameter, write parameter

**Built-in element-wise kernel**:
An MSL kernel synthesized by `malus-codegen-gpu` for a primitive arithmetic operator (`malus_add`, `malus_sub`, `malus_mul`, `malus_div`), dispatched via `kernel_dispatch` with a sequential `kernel_id` appended after user kernels. Indistinguishable from a user kernel at runtime.
_Avoid_: Builtin kernel, intrinsic kernel, stdlib kernel
