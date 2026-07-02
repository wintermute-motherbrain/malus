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
malus's automatic compile-time memory management system. Uses escape analysis and borrow-inference to insert static `free` calls for tensors. Falls back to reference counting only where ownership is genuinely structurally ambiguous — a `List<T>` (which may alias across a call boundary) or a struct field where a single owner can't be proven, never a scalar `Tensor` binding (M29: the autograd tape retains its own copy of anything it saves, so scalar-tensor RC is never needed for tape-survival — see ADR-0026 "why this supersedes ADR-0002/0016"). The compiler picks a single owner per allocation; all other uses are zero-cost borrows. Inspired by the Lobster language's ownership model (https://aardappel.github.io/lobster/memory_management.html). See ADR-0026.
_Avoid_: GC, garbage collector, allocator

**Escape analysis**:
The compiler pass that determines whether a tensor's lifetime can be fully resolved at compile time by tracking where it is created, used, and last referenced.

**Owner**:
The single binding, struct field, or array element assigned a tensor allocation. Exactly one owner per allocation. A scalar `Tensor` owner always gets a static `tensor_free`, at its last-use point, regardless of whether it's grad-tracked (M29) — `tensor_free` itself decrements a refcount (see Static drop); RC survives only for the aggregate cases described under RC fallback.
_Avoid_: primary reference, strong reference

**Borrow**:
Any use of a tensor allocation that is not the owner — a function parameter (uniform borrow ABI, M29/ADR-0026 D2), or a same-scope alias the compiler proves is outlived by its source. Borrows carry zero refcount cost — `tensor_retain`/`tensor_release` are not emitted, and the owner's own drop covers every borrow's uses. The programmer does not annotate borrows; the compiler infers them.
_Avoid_: reference, immutable reference, view

**RC fallback**:
When CTMM determines ownership is genuinely structurally ambiguous, it falls back to reference counting: a `List<T>` (always RC, ADR-0034 — may alias across a call boundary) and struct fields where a single owner can't be proven. `tensor_retain` increments the refcount when a second, independent owner needs its own reference; `tensor_release` decrements and frees when the refcount hits zero. A tensor saved onto the autograd tape is *not* an RC-fallback case — the tape retains its own copy synchronously when it records the op, independent of the binding's own (always-static) drop. In the hot-path training loop, ≤ ~5% of the compiler's own emitted `Retain`/`Release`/`RetainAgg`/`ReleaseAgg` nodes (relative to a naive non-borrow-inferred baseline) should remain — see the M29 RC-ratio gate. See ADR-0026.
_Avoid_: garbage collection, ARC

**Static drop**:
A CTMM drop point that is fully determined at compile time. Emitted as a `TypedStmt::Drop` node, lowered to a `tensor_free` call — which is itself a refcount decrement (`tensor_free` delegates to `tensor_release`, freeing at zero); "static" describes that the *decision* to drop here is compile-time-determined, not that the underlying op is somehow not a refcount operation. No RC bookkeeping overhead (no extra `Retain` to balance it).
_Avoid_: deterministic free (too vague)

**Retain site**:
A use of a tensor or `List<T>` handle that CTMM's emission pass recognizes as reusing an existing allocation rather than producing a fresh one — a same-scope alias (`let b = a`, `let t = v.data`) or an aggregate-literal field/element (`StructInit`, `ArrayLiteral`) — and therefore emits a `Retain`/`RetainAgg` for, to balance a later independent drop of the same handle. `malus-sema/src/retain_sites.rs`'s `retain_sites` is the single, exhaustive recognizer for these shapes: both the emission pass (`ctmm.rs`) and the borrow-inference demotion pass (`borrow_inference.rs`, which removes retain sites it can prove redundant under RC fallback) consult it, so the two passes cannot silently disagree about which aliases are retain-worthy. See ADR-0026.
_Avoid_: alias site, reference site

### Tensors

**Tensor**:
The core built-in primitive type. Parameterized by dtype (`Tensor<f32>`), with dynamic shape. Not a library type — the compiler has deep knowledge of tensors for memory management and codegen.
_Avoid_: Array, ndarray, matrix

**Placement**:
Whether a tensor is logically associated with CPU or GPU. Explicit at creation, but transfers at the `fn`/`kernel` boundary are inserted automatically by the compiler.
_Avoid_: Device, location

**In-flight tensor**:
A tensor that has been passed to a GPU op whose work has not yet been committed. Since M31, freeing one is memory-safe (Metal command buffers retain referenced resources) and reads are guarded by the runtime's pending tracking / auto-flush — the compiler no longer inserts barriers for it by default.
_Avoid_: Pending input, GPU-active tensor

**Pending tensor**:
A tensor produced by a GPU op (custom kernel or MPS matmul) whose work has not yet been committed. Since M31, every host-side read auto-flushes first (pending tracking), so user code never observes stale data.
_Avoid_: Pending output, uncommitted tensor

**Ready tensor**:
A tensor whose data is already materialized in the `StorageModeShared` buffer with no uncommitted GPU writes (commit-generation stamp ≤ last completed generation). Produced by host-initialized allocations (`tensor_alloc_*`, `randn`, `freeze`); since M31, MPS-backed ops (e.g. `tensor_matmul`) return *pending* tensors, and any tensor becomes ready after a flush. Counterpart to pending tensor.
_Avoid_: CPU tensor, completed tensor

**Shape metadata**:
The `shape: Vec<usize>` field on `TensorBuffer`, recording the n-dimensional extent of a tensor (invariant: `len == shape.iter().product()`). Runtime-only — absent from the type system, which is dtype-only. Validated at runtime by ops that require specific rank (e.g. `tensor_matmul` requires 2-D). Static shape inference is deferred post-V4. See ADR-0013.
_Avoid_: Tensor shape, static shape (describing a compile-time feature not yet built)

**Pending set**:
The CTMM compile-time set of tensor bindings produced or consumed by a GPU-producing expression (`KernelCall`, tensor `BinOp`, or GPU-returning `Call`) since the last `GpuBarrier`, driving static barrier insertion. Since M31 that pass is off by default (an opt-in A/B lever, `--static-barriers`); read safety is the runtime's pending tracking / auto-flush. Slated for deletion when V6's static commit-planner lands.
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
A `Tensor<i32>` or `Tensor<i64>` used as an index into another tensor (e.g. for embedding lookup). Not grad-tracked — integer tensors carry no gradient. Added in M19 as a narrow dtype carve-out; full non-f32 float compute generality is post-V4. Both i32 and i64 are accepted by `embedding` and `cross_entropy`; i32 is the M19 done-when dtype.
_Avoid_: integer tensor (fine informally, but "index tensor" conveys the use)

### Kernel mechanics

**Launch configuration**:
Static properties of a kernel dispatch — threadgroup size, grid dimensions — set via annotations on the kernel function.
_Avoid_: Dispatch config, thread config

**Explicit kernel**:
A `kernel` body that uses thread-hierarchy intrinsics, flat tensor indexing (`a[expr]`), `let shared` arrays, `barrier()`, and/or control flow (`if`/`for`/`while`). Tensor params stay as `Tensor<dtype>` (indexable pointers in MSL); output is written via the implicit `out` binding (`out[expr] = val`). Sema detects this form via `is_implicit_map_kernel` returning false. Codegen-gpu emits a full MSL kernel with scalar params packed into a `struct Uniforms_N` at `buffer(handle_count+1)`. Added in V4-M1 (M24). See ADR-0027.
_Avoid_: Full kernel, v2 kernel, general kernel

**Implicit-map kernel**:
The legacy `kernel` sugar form retained for backward compat: body consists only of `let`/`let mut` bindings and a final `return scalar_expr`. Sema rebinds each tensor param to its scalar element type; codegen-gpu emits `out[tid] = expr`. Detected by `is_implicit_map_kernel(body)`. No thread intrinsics or indexed writes (`out[i]=…`) are valid in this form.
_Avoid_: Simple kernel, elementwise kernel (too vague)

**Thread hierarchy intrinsics**:
Built-in functions callable only inside explicit kernel bodies, exposing the Metal thread model: `thread_id()` (global), `threadgroup_id()`, `thread_in_threadgroup()` (local), `threads_per_threadgroup()`, `threads_per_grid()`. All return `i32`. Map to MSL `[[thread_position_in_grid]]` etc. Only the intrinsics actually used in a kernel body have their MSL `[[attribute]]` parameters injected. Added in V4-M1 (M24). See ADR-0027.
_Avoid_: GPU intrinsics, builtins (too vague)

**Shared memory**:
Threadgroup-scoped fast memory in a kernel body, declared as `let shared x: Array<f32, N>`. Size `N` is a compile-time integer literal. Maps to MSL `threadgroup float x[N]`. Only valid inside an explicit kernel body; sema rejects it in `fn` bodies. Used for reduction intermediates (e.g. partial sums in softmax, mean/var in layernorm). Added in V4-M1 (M24).
_Avoid_: local memory, scratchpad, SharedArray (not a type in malus)

**`barrier()`**:
Explicit threadgroup barrier inside a kernel body. Emits MSL `threadgroup_barrier(mem_flags::mem_threadgroup)`. Only valid inside an explicit kernel body. Ensures all threads in a threadgroup have completed preceding memory writes to shared memory before any thread proceeds. Added in V4-M1 (M24).
_Avoid_: sync, fence, memory fence

**`inout` parameter**:
A kernel parameter that is mutated in place rather than borrowed immutably. Avoids allocating a new output buffer for element-wise ops. Post-V1.
_Avoid_: Mutable parameter, write parameter

**Dispatch uniforms**:
Scalar kernel parameters (non-tensor args such as `cols: i32`, `eps: f32`) packed into a `struct Uniforms_N` by codegen-gpu and bound at `buffer(handle_count+1)` in `kernel_dispatch_v2`. Accessed inside the MSL kernel as `u.field_name`. Field order in the struct matches declaration order in the malus kernel signature. Per-tensor `{shape, strides, ndim}` descriptors and multi-dimensional indexing are deferred to M25. See ADR-0027.
_Avoid_: kernel arguments (overloaded)

**`kernel_dispatch_v2`**:
The extended GPU dispatch function added in M23 (alongside the original `kernel_dispatch`). Unlike `kernel_dispatch`, which infers output shape from `inputs[0]` and uses `dispatchThreads`, `kernel_dispatch_v2` takes explicit grid/threadgroup configuration, an independent output shape/dtype, and a scalar uniforms blob. Uses `dispatchThreadgroups_threadsPerThreadgroup` so `grid_dims` are threadgroup counts. Required for reductions and any kernel that needs one threadgroup per row/slice. The `kernel_dispatch` symbol is retained unchanged for backward-compat with existing elementwise kernels.
_Avoid_: extended dispatch, v2 dispatch (fine informally; the C symbol name is canonical)

**CPU-compute counter**:
A process-global `AtomicI64 CPU_COMPUTE_CALLS` in `malus-runtime/src/lib.rs`, incremented (via `cpu_compute_inc()`) at the entry of every Rust function that loops over tensor element values. Exposed via `malus_cpu_compute_count() -> i64` and `malus_cpu_compute_reset()`. The V4 milestone CI gate: after a hot-path dispatch (forward pass, full step), `count == 0` proves every arithmetic op ran on the GPU. Double-counting (a CPU fn calling another CPU fn) is harmless — gates assert `== 0`, not exact counts. Orchestration ops (alloc, free, barrier, matmul-MPS, zero-copy reshape) do not increment. See ADR-0031.
_Avoid_: CPU op counter, CPU fallback counter (the term in the codebase is "cpu_compute")

**Threadgroup-per-row reduction**:
A GPU reduction pattern where each row (or slice) of a 2-D tensor maps to exactly one threadgroup, and threads within the threadgroup cooperate using shared memory and `threadgroup_barrier`. Requires `dispatchThreadgroups` semantics (not `dispatchThreads`) so the grid dimension equals the number of rows. Thread count per threadgroup must be a power of 2 to enable stride-halving tree reduction. The M23 `softmax_row` kernel is the canonical example: `grid=[rows,1,1]`, `tg=[cols,1,1]`, static `threadgroup float scratch[1024]`.
_Avoid_: row-wise reduction (fine informally), parallel reduction (too generic)

**Vendor primitive**:
A runtime op whose optimal implementation requires hardware resources not reachable from custom MSL compute kernels. Currently: `matmul` (→ MPS/AMX coprocessor on M-series). Implemented as a C-ABI Rust function in `malus-runtime`, not in the malus kernel language. Everything else is a kernel-language op. See ADR-0028.
_Avoid_: builtin op (overloaded), MPS op

**Built-in element-wise kernel**:
An MSL kernel synthesized by `malus-codegen-gpu` for a primitive arithmetic operator (`malus_add`, `malus_sub`, `malus_mul`, `malus_div`), dispatched via `kernel_dispatch` with a sequential `kernel_id` appended after user kernels. Indistinguishable from a user kernel at runtime.
_Avoid_: Builtin kernel, intrinsic kernel, stdlib kernel

**Element-space (kernel body)**:
The type regime inside a kernel body. Tensor parameters are bound as their *element* scalar type (`f32`, not `Tensor<f32>`). The body is checked as per-thread scalar math; the final `return` expression must have the return tensor's element type. Codegen emits `x[tid]` for params and bare `name` for `let`-bound locals. This means kernel-body comparison operators yield the operand's scalar dtype (a float mask), not `Bool`. `fn`-body BinOp rules and scalar-broadcast rules do not apply inside kernels.
_Avoid_: Scalar-space, thread-space, per-element computation

### Execution (V5 — M31 shipped)

**Async dispatch substrate**:
The V5 (M31) execution model: every GPU op — custom kernels and MPS matmul alike — encodes into the shared command buffer without committing; commits happen only when the host actually reads data. Phase 1 of the execution-model ladder (compile-time graph is the endgame; runtime lazy graph capture rejected). See ADR-0035.
_Avoid_: lazy evaluation, graph capture, deferred execution

**Pending tracking / auto-flush**:
The V5 runtime read-safety guarantee: each `TensorBuffer` records whether uncommitted GPU work has written it (a commit-generation stamp); any host-side read of a pending buffer flushes first. Replaces per-call-site `__flush()` workarounds and demotes CTMM static barrier insertion from correctness mechanism to optimization. See ADR-0035.
_Avoid_: barrier-before-read (the gap it fixes, not the mechanism), read fence

**Buffer pool**:
The M32 `MTLBuffer` recycling layer inside `MetalContext`: exact-byte-size free-lists, fed by `tensor_release`-at-zero and drawn from by `tensor_alloc_*`. A **pool hit** reuses a completed buffer; a **pool miss** allocates fresh. Invisible below the C ABI. See ADR-0039.
_Avoid_: buffer cache, allocator (it recycles; the OS/Metal still allocates on miss)

**Last-use generation**:
The pool's reuse gate (M32): the commit generation of the command buffer that last encoded *any* use of an `MTLBuffer` — input or output — stamped at every encode site, shared across reshape aliases via `Arc<PoolState>`. A pooled buffer is reusable only once this generation has completed. Distinct from the last-*write* generation, which gates host reads (M31): a ready-to-read buffer may still be an in-flight input to uncommitted work.
_Avoid_: last-write generation (the read-safety stamp, not the reuse gate), pending flag

**Memory budget**:
The soft ceiling on device bytes (live + pooled; `MALUS_MEM_BUDGET_MB`, default 8 GiB). On an over-budget pool miss whose bucket holds pending entries, the allocator flushes once and retries the pool — a recycling trigger, not a hard cap; a retry miss still allocates fresh. Bounds the memory of read-free loops, which never flush and so would never cycle the pool. See ADR-0039.
_Avoid_: memory cap / OOM limit (allocation never fails on it), eviction (nothing is evicted)

**Head-folding**:
Expressing multi-head attention with the existing 3-D batched matmul by folding batch and head dims: `[B,T,C] → reshape [B,T,H,hs] → permute (0,2,1,3) → reshape [B*H,T,hs]`, and unfolding after. Requires the rank-generic permute VJP (V5/M33); avoids adding 4-D matmul.
_Avoid_: 4-D matmul (not shipped), multi-head reshape (vague)

**Optimizer recursion**:
The V5 (M34) Module-composition pattern: the generic optimizer is applied per submodule, so each submodule's `parameters()` identity list receives the slot writes. `parameters()` results are never concatenated — a merged list would be a fresh snapshot and the optimizer would silently update the snapshot instead of the model (the ADR-0034 write-back hazard). See ADR-0036.
_Avoid_: parameter concat (rejected), flat parameter list (the V4 form this replaces)

**Submodule**:
A struct implementing `Module` stored inside another module (`GPT { blocks: List<Block> }`). Each submodule owns exactly one identity list of its trainable tensors, read via block-LOCAL index constants; composition is by optimizer recursion, never by merging lists. See ADR-0036.
_Avoid_: child module (PyTorch-ism; fine informally), layer (a Block is one transformer layer but "submodule" is the composition term)

**Moments**:
Per-submodule AdamW optimizer state: `struct Moments { ms: List<Tensor<f32>>, vs: List<Tensor<f32>> }`, held by the training loop in a `List<Moments>` parallel to the model's submodules (plus one for the top-level module's own tensors) — mirroring the parameter structure, per ADR-0036. The model never carries optimizer state.
_Avoid_: optimizer state on the model (rejected; diverges from PyTorch's optimizer-owns-state contract)

**Inherent method**:
A method in a trait impl whose name the trait does not declare (`fn forward` beside `fn parameters` in `impl Module for Block`, M34). Registered with the same mangling and static dispatch as trait methods; callable as `x.method(...)`. A method whose name matches a trait method must still match its signature exactly.
_Avoid_: extension method, default method (neither mechanism exists)

**Retain-on-bind (container elements)**:
The M34 rule (ADR-0040) that binding a container-element read (`let w = model.params[k]`) bumps the NEW binding's reference immediately after the bind — the container owns the element, so the binding's later drop must release its own reference, not steal the container's. Transient inline reads (`x @ model.params[WQ]` as an operand) are untouched borrows and cost nothing.
_Avoid_: inline-read rule (the pre-M34 workaround this retires as a constraint; inline reads survive only as a hot-path optimization)

**Autocast** (V5/M36, planned):
Mixed-precision training semantics: parameters/gradients/optimizer state stay f32; matmuls and forward elementwise kernels compute in bf16; reductions accumulate in f32. bf16-first because it needs no loss scaling. Surface finalized at M36 (recommendation: a `with autocast:` scope mirroring `no_grad`). See ADR-0037.
_Avoid_: half precision (imprecise — bf16, not f16), quantization (different technique)

### Benchmarking (V5)

**Warm per-step median**:
The canonical malus performance number (M30): the median wall-clock time of a full training step (batch construction through optimizer update, with a GPU flush inside the timed region), measured after skipping ≥3 warmup steps. Excludes process startup, MSL kernel compilation, data load/tokenize, and post-training generation. Matches `bench/nanogpt_pytorch.py`'s methodology (`torch.mps.synchronize()` inside the timed step, median over warm steps) so the Nx ratio compares like with like.
_Avoid_: avg/step (the M29 coarse whole-run average this replaces), steady-state throughput (a pipelined measure — the warm median deliberately serializes each step)

**Dispatch-overhead regression benchmark**:
The V5 role of the toy-config nanoGPT benchmark (`C=32, T=16, B=4`, `bench/nanogpt_step.sh`): at this scale both runtimes are dispatch-bound, not compute-bound, so the warm per-step median isolates dispatch-architecture cost. Run manually before/after each V5 substrate milestone (M31/M32) to confirm the number moves and nothing regresses it. Explicitly not a CI assert — wall-clock gates flake. Distinct from the M35 capstone benchmark (Karpathy config), which measures the claim itself.
_Avoid_: perf gate (the M35 ≤2x gate is the gate; this is a tracking benchmark), CI benchmark

### Autograd

**Grad-tracked tensor**:
A `Tensor<f32>` whose binding is statically inferred to derive from a grad leaf (created by `variable(...)`) and is not inside a `no_grad` scope. Inference is whole-program: the property propagates across function parameters/returns (a param is grad-tracked if any call site passes a grad-tracked arg; a return is grad-tracked if the return expr is) and struct fields (a field is grad-tracked if any store into it is), not just within one function body. There is no distinct `Variable` type in V4. Grad-tracking is a compiler-inferred property, computed in `malus-sema/src/grad_inference.rs`. See ADR-0030.
_Avoid_: Variable (eliminated in V4), Tensor.requires_grad

**Tape**:
A global thread-local define-by-run record. Each differentiable op on grad-tracked tensors pushes a `TapeNode` holding saved inputs and a VJP closure. `backward(loss)` walks the tape in reverse and clears it on completion. See ADR-0015.
_Avoid_: gradient graph, computation graph

**VJP (Vector-Jacobian Product)**:
The per-op backward rule used by `backward`. Given the output gradient, computes the input gradients. In V3, VJPs are Rust CPU functions in `malus-runtime/src/tape.rs`. In V4-M3, each op ships a paired GPU backward kernel dispatched from `backward()`.
_Avoid_: backward pass, gradient function

**`backward`**:
A builtin that walks the tape in reverse from a scalar-valued loss tensor, accumulates gradients into each leaf's `.grad` slot, releases saved tensors (RC), and clears the tape.
_Avoid_: backpropagation (fine informally, not a technical term here)

**`.grad`**:
A field accessor that returns the accumulated gradient as a `Tensor<f32>`. Legal on any grad-tracked tensor (gated on the same `grad_tracked` property as tape recording, not a separate leaf set); zero (or a zeros tensor) if the receiver is not a leaf or `backward` has not yet been called. Cleared by `zero_grad`. `.grad` is itself a detach point: `x.grad` is never grad-tracked, preventing double-backward. See ADR-0030.
_Avoid_: gradient tensor, derivative

**`.data`**:
A field accessor that returns the same tensor handle (identity at runtime) but is a detach point: the grad-inference pass marks `x.data` as never grad-tracked regardless of `x`. Used to read a grad-tracked tensor's value without pulling the read into the tape, e.g. an optimizer's `w.data - lr * grad`. Distinct from `no_grad` — `.data` severs one value's grad lineage at a point; `no_grad` suppresses tape recording for an entire scoped region. See ADR-0030.
_Avoid_: identity accessor (undersells that it detaches), raw tensor

**`no_grad`**:
A scoped region (`with no_grad: body`) that suppresses tape recording. Grad-tracked ops inside the body execute but push no `TapeNode`. A static scope — CTMM emits static-free for all tensors inside `no_grad` regardless of whether they derive from leaves. Used for inference and the optimizer update step.
_Avoid_: detach, inference mode

**Leaf tensor**:
A tensor created with `variable(t)`. The compiler marks it as a grad leaf; `.grad` accumulates from `backward`. Counterpart to intermediate tensors (produced by ops on leaves) which do not accumulate `.grad`.
_Avoid_: leaf Variable, parameter, weight (informal)

**`zero_grad`**:
A variadic builtin that resets the `.grad` of each passed leaf tensor to a zeros tensor of the same shape. Called at the start of each training step.
_Avoid_: clear gradients, reset gradients

### Types (V1)

**Tuple**:
An anonymous product type with positional fields. Constructed with `(expr, expr, ...)` (minimum 2 elements). Fields accessed via positional dot notation (`x.0`, `x.1`) or destructured in a `let` binding (`let (a, b) = x`). `let mut (a, b) = x` makes all bindings mutable. Heap-allocated like `Struct`; `DropTuple` releases owned fields on last-ref (refcount peek, M34) then decrements the box. Valid as local bindings and `fn` return types. Flat-only: element types may not themselves be tuples. Tuple elements may not appear as struct fields or array elements. `match` on tuples is deferred.
_Avoid_: anonymous struct (informal description, not the canonical term)

**Struct**:
A user-defined product type with named, typed fields. Constructed with keyword arguments: `Layer(weights=w, bias=b)`. Fields accessed with dot notation: `layer.weights`. A named tensor source is retained at construction (the struct's copy owns its own reference; the source binding's later drop is balanced — borrow-inference removes the pair when the construction is the source's last use). `DropStruct` releases fields only on last-ref (refcount peek, M34) — a struct box shared as a `List` element must not lose its fields while the container's copy is live — then decrements the box.
_Avoid_: Record, dataclass

**Enum**:
A user-defined sum type with named variants. Variants may carry data (named fields). Constructed with dot notation: `Activation.Relu` (tag-only) or `Activation.Linear(w=weights)` (data-carrying). Matched with `match`.
_Avoid_: Tagged union, algebraic data type (ADT is fine internally but too jargon-heavy for user docs)

**Match**:
An exhaustive pattern-match statement over an enum value. All variants must have exactly one arm. Arms may destructure data-carrying variants and bind field values.
_Avoid_: Switch, case expression

**Fixed-length array** (V1):
A compile-time-sized sequence: `[expr1, expr2, ...]`. Type is `Array<T, N>` where `N` is known statically. Supports indexing (`arr[i]`) and `for x in arr` iteration. Tensor elements use escape-analysis RC.
_Avoid_: List, dynamic array, vector

**`List<T>`** (V4):
A sequence type added in V4-M5, fixed-length at construction (`lst.push` deferred post-V4). Used as the return type of `Module.parameters()` — critically, `parameters()` returns a model's stored list **by identity** (a borrow of the model's own field), not a fresh literal, so that a generic optimizer's slot reassignment (`ps[i] = variable(...)`) is visible on the model's next use. That identity-return creates aliasing across a call boundary, which is why `List<T>` is itself a **reference-counted aggregate** — an ARC header (`RetainAgg`/`ReleaseAgg`, the same dormant-until-M28 mechanism structs/tuples/enums use) plus a length word, NOT `Array`'s headerless escape-analysis-only static drop. Tensor *elements* inside a `List` still use the ordinary tensor lifetime rules (tape-RC / static-free), unaffected by the container's own RC. When the container's refcount hits zero its elements are released recursively by element type (tensor, struct, nested `List` — M34; pre-M34 only tensor elements were released and `List<Struct>` leaked). Binding an element read (`let w = ps[i]`) is sound: the binding is retained on bind (ADR-0040). Supports indexing, `for x in list` iteration (the loop variable is a borrow of the list-owned element — never dropped), and `len(lst)`. `Dict` is post-V4. See ADR-0034.
_Avoid_: Vec, dynamic array, vector (fine informally; List is the canonical term); "escape-analysis RC same as Array" (wrong — List is container-RC, Array is not)

**Trait**:
A named protocol (`trait Name: fn method(self, ...) -> T`) that types can implement with `impl Name for Type`. Exactly one trait mechanism in V4; no inheritance. The primary built-in trait is `Module`. See ADR-0007 (fenced scope).
_Avoid_: interface, protocol (fine informally), type class

**`Module` trait**:
The V4 trait for neural network components: `trait Module: fn parameters(self) -> List<Tensor<f32>>`. A type that implements `Module` can be passed to a generic optimizer. `impl Module for GPT` provides the parameters list for the capstone.
_Avoid_: nn.Module (PyTorch name; fine as analogy, not the canonical term)

**Generics**:
Type parameters on `fn` (`fn f<T: Trait>(x: T)`). Monomorphized in sema before grad-inference/CTMM/codegen run — every downstream pass sees only concrete, mangled-name `TypedFn`s (compiler-internal detail, not observable syntax). V4 scope: one type parameter per item, one trait bound, no higher-kinded types, no associated types. **User-defined generic `struct`/`enum` are deferred post-V4** (ADR-0034) — V4 generics apply to `fn` only.
_Avoid_: templates, polymorphism (fine informally)

**`let mut` / reassignment** (V1):
A mutable binding declared with `let mut x = expr`. Can be reassigned with `x = new_val`. CTMM treats reassignment as: drop the old value (if tensor, emit `tensor_free` or `tensor_release`), then bind the new value. Prevents the shadowing-in-loops problem where `let x = x + delta` inside a loop scopes the new `x` to the loop body.
_Avoid_: Mutable variable (fine as informal description; just not a technical term)

### Transformer stdlib (M18)

**softmax** (M18):
Numerically stable softmax over a named required `axis=N` dimension. `softmax(t, axis=2)` returns a tensor of the same shape with exponentials normalized over axis 2. Differentiable; VJP: `dx = s ⊙ (dout − sum(dout⊙s, axis, keepdim))` where `s` is the forward output. Subset divergence from PyTorch: same contract (`torch.softmax(t, dim=N)`).
_Avoid_: normalized exponential, log-softmax (different op)

**layernorm** (M18):
Layer normalization over a named required `axis=N` dimension: `y = (x − μ) / sqrt(var(x, axis) + 1e-5)`. No learnable affine parameters (γ/β) in M18 — users compose their own `y * gamma + beta` after the call using grad-tracked tensor arithmetic. Differentiable. PyTorch subset: `torch.layer_norm` additionally accepts `weight`/`bias` tensors; the malus single-axis form is additive.
_Avoid_: batch norm (different algorithm), RMS norm (different normalization)

**gelu** (M18):
Gaussian Error Linear Unit activation, tanh approximation: `0.5 * x * (1 + tanh(c0 * (x + c1 * x^3)))`, `c0=0.7978845608`, `c1=0.044715`. Differentiable. PyTorch subset divergence: PyTorch's default `gelu` uses the exact `erf` formulation; the tanh-approx (PyTorch's `gelu(approximate='tanh')`) is the malus default. Additive post-V3.
_Avoid_: ReLU, SiLU (different activations)

**cross_entropy** (M18; tightened M19):
Cross-entropy loss: `cross_entropy(logits: Tensor<f32> [N,C], targets: Tensor<i32|i64> [N]) -> Tensor<f32> [1]`. Applies softmax internally (numerically stable) and computes `-mean(log(s[i, targets[i]]))`. M19 tightened targets from f32 placeholder to integer index tensor. Differentiable when logits are grad-tracked; VJP: `d_logits[i,j] = dout/N * (s[i,j] − 1{j == targets[i]})`.
_Avoid_: NLL loss (expects log-probabilities, not logits), softmax cross-entropy (redundant; the softmax is fused inside)

**embedding** (M19):
Differentiable token/vocabulary lookup. `embedding(weight: Tensor<f32> [V,D], indices: Tensor<i32|i64> [T]) -> Tensor<f32> [T,D]`. Output is grad-tracked when `weight` is grad-tracked. Copies row `indices[t]` of `weight` into `out[t]`. VJP: scatter-add — for each position `t`, `dweight[indices[t]] += dout[t]`; indices receive no gradient (integer, non-differentiable). Analogous to `torch.nn.Embedding.forward`. Name `gather` is reserved — `torch.gather(input, dim, index)` has a different contract (general gather over any axis, not row lookup).
_Avoid_: gather (reserved, different PyTorch contract), lookup table

**randn** (M19):
`randn(d0, d1, ...) -> Tensor<f32>`. Returns a tensor of the given shape filled with independent standard-normal samples. CPU-side implementation using Philox4x32-10 counter-based RNG + Box-Muller transform. Non-differentiable (returns `Tensor<f32>` that is not grad-tracked; wrap with `variable(randn(...))` to register it as a grad leaf). No user-settable seed in M19; the stream is determined by a thread-local call counter (incremented per `randn` call). See ADR-0024.
_Avoid_: random normal (fine informally), random init

**causal_mask** (M18):
`causal_mask(T: i64) -> Tensor<f32>`. Returns a `[T, T]` additive mask: `0.0` on and below the main diagonal, `-1e9` strictly above. Added to attention logits before softmax so future positions receive ~0 attention weight. Non-differentiable (no tape entry). PyTorch equivalent: `torch.triu(torch.full((T,T),-1e9), diagonal=1)`.
_Avoid_: attention mask (overloaded), upper-triangular mask (imprecise)

### Optimization

**AdamW**:
The Adam optimizer with decoupled weight decay. In V4, implemented as a generic `fn adamw<M: Module>(model: M, ...)` that loops `model.parameters()` — no hand-unrolling. Previous `examples/adamw.ml` was a hand-specialized prototype. Added in M20; made generic in V4-M5.
_Avoid_: Adam, SGD with weight decay (different algorithms)

**Lvalue assignment**:
An assignment whose target is an indexed element (`a[i] = e`) or a struct field (`s.field = e`) rather than a bare name. Added in M20. Requires the base binding to be mutable (`let mut` local or `mut` parameter). CTMM drops the old element/field value before binding the new one (release is emitted inline at codegen time; Index/Field targets do not receive a separate CTMM drop stmt).
_Avoid_: in-place update, indexed assignment (fine informally)

**`mut` parameter**:
A function parameter declared with a `mut` prefix (`fn f(mut a: Array<T, N>)`). Permits interior mutation of the aggregate (`a[i] = e`, `a.f = e`) but rejects bare rebinding (`a = new_val`). The callee receives a borrowed heap pointer — mutations are visible to the caller through the shared allocation. The callee does not free the aggregate box (it is a borrow, not a move). Added in M20; amends ADR-0014. Distinct from `let mut` (which permits both rebind and interior mutation).
_Avoid_: mutable parameter, pass-by-reference, in-out parameter

**power operator (`**`)**:
A binary scalar operator for exponentiation: `f32 ** {f32 | i32 | i64} → f32`. Right-associative, highest binary precedence. Non-integer exponents are supported because the exponent may be a runtime loop counter. Lowered to `malus_powf(f32, f32) -> f32` (wraps `f32::powf`). Added in M20 using `**` (Python parity) rather than `^` (would read as XOR to Python users). See ADR-0025.
_Avoid_: `^` (reserved/XOR), caret operator, exponentiation
