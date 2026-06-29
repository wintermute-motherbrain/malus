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
When CTMM cannot statically determine the drop point — specifically when a tensor is stored in a heap-allocated container (struct field, fixed-length array element) — it falls back to reference counting. `tensor_retain` increments the refcount; `tensor_release` decrements it and frees when zero. The fast path (no RC) is preserved for all statically resolvable lifetimes, including tensors that flow through `if`/`else`/`for`/`while` bodies (handled by hierarchical scope analysis — see ADR-0014). See ADR-0002.
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

**Ready tensor**:
A tensor whose data is already materialized in the `StorageModeShared` buffer and safe to read on the CPU without a barrier. Produced by eager CPU-side stdlib ops (`tensor_matmul`, `tensor_transpose`, `tensor_sum`, `tensor_alloc_zeros_gpu`, `tensor_alloc_ones_gpu`). Counterpart to pending tensor.
_Avoid_: CPU tensor, completed tensor

**Shape metadata**:
The `shape: Vec<usize>` field on `TensorBuffer`, recording the n-dimensional extent of a tensor (invariant: `len == shape.iter().product()`). Runtime-only — absent from the type system, which is dtype-only. Validated at runtime by ops that require specific rank (e.g. `tensor_matmul` requires 2-D). See ADR-0013.
_Avoid_: Tensor shape, static shape

**Pending set**:
The CTMM compile-time set of tensor bindings produced or consumed by a GPU-producing expression (`KernelCall`, tensor `BinOp`, or GPU-returning `Call`) since the last `GpuBarrier`. Any CPU-side access of a binding in the pending set triggers a barrier insertion.
_Avoid_: GPU-active set

**Broadcasting**:
NumPy right-aligned shape-compatibility rule for tensor arithmetic. Two shapes are broadcast-compatible if, right-aligned, each dimension pair is either equal, 1 (expanded to match the other), or absent (treated as 1). Example: `[4,8] + [1,8]` → `[4,8]`. Added in M16.
_Avoid_: shape broadcasting (fine informally), implicit replication

**Axis reduction**:
A reduction op (`sum`, `mean`, `max`, `var`) applied over a specific tensor axis, optionally preserving the reduced dimension as size 1 (`keepdim=true`). Distinct from V1's whole-tensor `sum` (which always returns a `[1]` tensor). Added in M16.
_Avoid_: reduce, fold

**reshape**:
A zero-copy contiguous shape change. `reshape(t, d0, d1, ...)` returns a new `TensorBuffer` that shares the underlying `MTLBuffer` with the input (Obj-C retain on the same allocation, no data copy). Panics at runtime if the element counts differ. Equivalent to PyTorch `view` semantics in M17 (always contiguous, zero-copy). The name `view` is reserved for a future non-contiguous strided-view operation. Differentiable: VJP reshapes the output gradient back to the input shape.
_Avoid_: view (reserved), copy-reshape, data reshape

**transpose**:
A two-axis swap. `transpose(t)` reverses a 2-D tensor (equivalent to `permute(t, 1, 0)`); `transpose(t, i, j)` swaps axes `i` and `j` in a tensor of any rank. Distinct from `permute` — `torch.transpose` in PyTorch only ever swaps two axes; the malus name inherits that contract. Differentiable: VJP applies the inverse permutation.
_Avoid_: permute (different operation), flip axes

**permute**:
A full axis reorder. `permute(t, p0, p1, ..., p_{rank-1})` reorders all axes according to the permutation vector. Analogous to `torch.permute`. Requires exactly `rank` dim arguments. Differentiable: VJP applies the inverse permutation.
_Avoid_: transpose (different operation — transposes only two axes)

**Batched matmul**:
A `tensor_matmul` call where both operands are 3-D with identical leading batch dimension `B`: `(B, M, K) @ (B, K, N) → (B, M, N)`. Computed as `B` independent 2-D matmuls (eager CPU loops, M17; MPS-accelerated in M21 per ADR-0017). VJP is per-slice: `dA[b] = dC[b] @ B[b]^T`, `dB[b] = A[b]^T @ dC[b]`.
_Avoid_: batched matrix multiply (fine informally), bmm

**Index tensor**:
A `Tensor<i32>` or `Tensor<i64>` used as an index into another tensor (e.g. for embedding lookup). Not differentiable — integer tensors are never `Variable`. Added in M19 as a narrow dtype carve-out; full non-f32 float compute generality is post-V3. Both i32 and i64 are accepted by `embedding` and `cross_entropy`; i32 is the M19 done-when dtype.
_Avoid_: integer tensor (fine informally, but "index tensor" conveys the use)

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

### Autograd

**Variable**:
A grad-tracked tensor. Distinct from `Tensor` — the compiler always manages `Variable` lifetimes with reference counting (type-directed RC), while `Tensor` keeps static Drop. `Variable` values record their forward ops onto the global tape; `backward(loss)` accumulates gradients into leaf `.grad` slots. Constructed with `variable(t: Tensor<f32>)`.
_Avoid_: program "variable" (lowercase), "Tensor.requires_grad", "Tensor with grad"

**Tape**:
A global thread-local define-by-run record. Each differentiable op on `Variable` values pushes a `TapeNode` holding saved inputs and a VJP closure. `backward(loss)` walks the tape in reverse and clears it on completion. See ADR-0015.
_Avoid_: gradient graph, computation graph

**VJP (Vector-Jacobian Product)**:
The per-op backward rule used by `backward`. Given the output gradient, computes the input gradients. All VJPs for V1/V2 ops are hardcoded in `malus-runtime/src/tape.rs`.
_Avoid_: backward pass, gradient function

**`backward`**:
A builtin that walks the tape in reverse from a scalar-valued `Variable` (the loss), accumulates gradients into each leaf's `.grad` slot, releases saved tensors, and clears the tape.
_Avoid_: backpropagation (fine informally, not a technical term here)

**`.grad`**:
A field accessor on a leaf `Variable<f32>` that returns the accumulated gradient as a plain `Tensor<f32>`. Zero (or a zeros tensor) if `backward` has not yet been called. Cleared by `zero_grad`.
_Avoid_: gradient tensor, derivative

**`no_grad`**:
A scoped region (`with no_grad: body`) that suspends tape recording. `Variable` ops inside the body execute their forward computation but push no `TapeNode`. Used for the optimizer update step and for inference. `Variable` RC semantics are unchanged inside `no_grad`.
_Avoid_: detach, inference mode

**Leaf Variable**:
A `Variable` created directly by the user with `variable(t)`. Accumulates `.grad` on `backward`. Counterpart to intermediate Variables, which are produced by ops on other Variables and have no `.grad`.
_Avoid_: parameter, weight (informal descriptions, not technical terms)

**`zero_grad`**:
A variadic builtin that resets the accumulated `.grad` of each passed `Variable` to a zeros tensor of the same shape. Called at the start of each training step to clear gradients before the next `backward`.
_Avoid_: clear gradients, reset gradients

### Types (V1)

**Tuple**:
An anonymous product type with positional fields. Constructed with `(expr, expr, ...)` (minimum 2 elements). Fields accessed via positional dot notation (`x.0`, `x.1`) or destructured in a `let` binding (`let (a, b) = x`). `let mut (a, b) = x` makes all bindings mutable. Heap-allocated like `Struct`; `DropTuple` releases tensor/variable fields via `tensor_release` then frees the box. Valid as local bindings and `fn` return types. Flat-only: element types may not themselves be tuples. Tuple elements may not appear as struct fields or array elements. `match` on tuples is deferred.
_Avoid_: anonymous struct (informal description, not the canonical term)

**Struct**:
A user-defined product type with named, typed fields. Constructed with keyword arguments: `Layer(weights=w, bias=b)`. Fields accessed with dot notation: `layer.weights`. Tensor fields are moved into the struct at construction (ownership transfers; no retain is emitted). `DropStruct` releases each tensor field via `tensor_release`, freeing the tensor when the refcount hits zero.
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

### Transformer stdlib (M18)

**softmax** (M18):
Numerically stable softmax over a named required `axis=N` dimension. `softmax(t, axis=2)` returns a tensor of the same shape with exponentials normalized over axis 2. Differentiable; VJP: `dx = s ⊙ (dout − sum(dout⊙s, axis, keepdim))` where `s` is the forward output. Subset divergence from PyTorch: same contract (`torch.softmax(t, dim=N)`).
_Avoid_: normalized exponential, log-softmax (different op)

**layernorm** (M18):
Layer normalization over a named required `axis=N` dimension: `y = (x − μ) / sqrt(var(x, axis) + 1e-5)`. No learnable affine parameters (γ/β) in M18 — users compose their own `y * gamma + beta` after the call using `Variable` arithmetic. Differentiable. PyTorch subset: `torch.layer_norm` additionally accepts `weight`/`bias` tensors; the malus single-axis form is additive.
_Avoid_: batch norm (different algorithm), RMS norm (different normalization)

**gelu** (M18):
Gaussian Error Linear Unit activation, tanh approximation: `0.5 * x * (1 + tanh(c0 * (x + c1 * x^3)))`, `c0=0.7978845608`, `c1=0.044715`. Differentiable. PyTorch subset divergence: PyTorch's default `gelu` uses the exact `erf` formulation; the tanh-approx (PyTorch's `gelu(approximate='tanh')`) is the malus default. Additive post-V3.
_Avoid_: ReLU, SiLU (different activations)

**cross_entropy** (M18; tightened M19):
Cross-entropy loss: `cross_entropy(logits: Variable<f32> [N,C], targets: Tensor<i32|i64> [N]) -> Variable<f32> [1]`. Applies softmax internally (numerically stable) and computes `-mean(log(s[i, targets[i]]))`. M19 tightened targets from f32 placeholder to integer index tensor (`Tensor<i32>` or `Tensor<i64>`). Differentiable; VJP: `d_logits[i,j] = dout/N * (s[i,j] − 1{j == targets[i]})`.
_Avoid_: NLL loss (expects log-probabilities, not logits), softmax cross-entropy (redundant; the softmax is fused inside)

**embedding** (M19):
Differentiable token/vocabulary lookup. `embedding(weight: Variable<f32> [V,D], indices: Tensor<i32|i64> [T]) -> Variable<f32> [T,D]`. Copies row `indices[t]` of `weight` into `out[t]`. VJP: scatter-add — for each position `t`, `dweight[indices[t]] += dout[t]`; indices receive no gradient (integer, non-differentiable). Analogous to `torch.nn.Embedding.forward`. Name `gather` is reserved — `torch.gather(input, dim, index)` has a different contract (general gather over any axis, not row lookup).
_Avoid_: gather (reserved, different PyTorch contract), lookup table

**randn** (M19):
`randn(d0, d1, ...) -> Tensor<f32>`. Returns a tensor of the given shape filled with independent standard-normal samples. CPU-side implementation using Philox4x32-10 counter-based RNG + Box-Muller transform. Non-differentiable (returns plain `Tensor`, not `Variable`; wrap with `variable(randn(...))` to make it a leaf). No user-settable seed in M19; the stream is determined by a thread-local call counter (incremented per `randn` call). See ADR-0024.
_Avoid_: random normal (fine informally), random init

**causal_mask** (M18):
`causal_mask(T: i64) -> Tensor<f32>`. Returns a `[T, T]` additive mask: `0.0` on and below the main diagonal, `-1e9` strictly above. Added to attention logits before softmax so future positions receive ~0 attention weight. Non-differentiable (no tape entry). PyTorch equivalent: `torch.triu(torch.full((T,T),-1e9), diagonal=1)`.
_Avoid_: attention mask (overloaded), upper-triangular mask (imprecise)

### Optimization

**AdamW**:
The Adam optimizer with decoupled weight decay, implemented in malus source as a stdlib construct (`examples/stdlib/adamw.ml`). Uses per-parameter moment buffers (`m`, `v`) updated via lvalue assignment. Added in M20.
_Avoid_: Adam, SGD with weight decay (different algorithms)

**Lvalue assignment**:
An assignment whose target is an indexed element (`a[i] = e`) or a struct field (`s.field = e`) rather than a bare name. Added in M20. CTMM drops the old element/field value before binding the new one.
_Avoid_: in-place update, indexed assignment (fine informally)
