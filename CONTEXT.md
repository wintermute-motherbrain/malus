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
malus's automatic compile-time memory management system. Uses escape analysis to insert static `free` calls for tensors. Falls back to reference counting only when lifetimes are structurally ambiguous (heap-stored tensors — struct fields, array elements). Inspired by the Lobster language's ownership model.
_Avoid_: GC, garbage collector, allocator, Lobster

**Escape analysis**:
The compiler pass that determines whether a tensor's lifetime can be fully resolved at compile time by tracking where it is created, used, and last referenced.

**Structurally ambiguous**:
A tensor lifetime that cannot be resolved at compile time — specifically, when a tensor is stored in a heap-allocated container (struct field, fixed-length array). Triggers the RC fallback.
_Avoid_: dynamic lifetime, heap lifetime

**RC fallback**:
When CTMM cannot statically determine the drop point (structurally ambiguous lifetime, or tensor flowing through a conditional branch), it falls back to reference counting. `tensor_retain` increments the refcount; `tensor_release` decrements it and frees when zero. The fast path (no RC) is preserved for all statically resolvable lifetimes. See ADR-0002.
_Avoid_: garbage collection, ARC

**Static drop**:
A CTMM drop point that is fully determined at compile time. Emitted as a `TypedStmt::Drop` node, lowered to a `tensor_free` call. No refcount overhead.
_Avoid_: deterministic free (too vague)

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

**Multi-statement kernel body** (V1):
A kernel body that contains local `let` bindings and comparison/ternary expressions before the final `return`. Enables gradient kernels like `relu_backward` that need intermediate values. Full control flow inside kernels (loops, conditionals) is post-V1.
_Avoid_: Complex kernel, multi-line kernel

**Intrinsics**:
Built-in functions callable inside kernel bodies that expose GPU-level concepts: thread identity (`threadgroup_id()`), shared memory (`shared_alloc()`), SIMD operations (`simd_shuffle()`). Post-V1.
_Avoid_: Builtins, GPU functions

**`inout` parameter**:
A kernel parameter that is mutated in place rather than borrowed immutably. Avoids allocating a new output buffer for element-wise ops. Post-V1.
_Avoid_: Mutable parameter, write parameter

**Built-in element-wise kernel**:
An MSL kernel synthesized by `malus-codegen-gpu` for a primitive arithmetic operator (`malus_add`, `malus_sub`, `malus_mul`, `malus_div`), dispatched via `kernel_dispatch` with a sequential `kernel_id` appended after user kernels. Indistinguishable from a user kernel at runtime.
_Avoid_: Builtin kernel, intrinsic kernel, stdlib kernel

**Element-space (kernel body)**:
The type regime inside a kernel body. Tensor parameters are bound as their *element* scalar type (`f32`, not `Tensor<f32>`). The body is checked as per-thread scalar math; the final `return` expression must have the return tensor's element type. Codegen emits `x[tid]` for params and bare `name` for `let`-bound locals. This means kernel-body comparison operators yield the operand's scalar dtype (a float mask), not `Bool`. `fn`-body BinOp rules and scalar-broadcast rules do not apply inside kernels.
_Avoid_: Scalar-space, thread-space, per-element computation

### Types (V1)

**Struct**:
A user-defined product type with named, typed fields. Constructed with keyword arguments: `Layer(weights=w, bias=b)`. Fields accessed with dot notation: `layer.weights`. Tensor fields trigger the RC fallback — the struct holds a retain, released when the struct is dropped.
_Avoid_: Record, dataclass

**Enum**:
A user-defined sum type with named variants. Variants may carry data (named fields). Constructed with dot notation: `Activation.Relu` (tag-only) or `Activation.Linear(w=weights)` (data-carrying). Matched with `match`.
_Avoid_: Tagged union, algebraic data type (ADT is fine internally but too jargon-heavy for user docs)

**Match**:
An exhaustive pattern-match statement over an enum value. All variants must have exactly one arm. Arms may destructure data-carrying variants and bind field values.
_Avoid_: Switch, case expression

**Fixed-length array** (V1):
A compile-time-sized sequence: `[expr1, expr2, ...]`. Type is `Array<T, N>` where `N` is known statically. Supports indexing (`arr[i]`) and `for x in arr` iteration. Tensor elements trigger the RC fallback. Growable arrays (`Vec<T>`) are post-V1.
_Avoid_: List, dynamic array, vector

**`let mut` / reassignment** (V1):
A mutable binding declared with `let mut x = expr`. Can be reassigned with `x = new_val`. CTMM treats reassignment as: drop the old value (if tensor, emit `tensor_free` or `tensor_release`), then bind the new value. Prevents the shadowing-in-loops problem where `let x = x + delta` inside a loop scopes the new `x` to the loop body.
_Avoid_: Mutable variable (fine as informal description; just not a technical term)
