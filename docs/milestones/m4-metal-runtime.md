# M4 — Metal Runtime

**Crate:** `malus-runtime`
**Done when:** A Rust test allocates a `f32` GPU tensor via the runtime, writes values from the CPU, reads them back, and frees the buffer — with no Metal validation errors.

## Starting point

The five C ABI functions (`tensor_alloc_gpu`, `tensor_print`, `tensor_free`, `gpu_barrier`, `kernel_dispatch`) are currently **stubbed in `crates/malus-codegen-cpu/src/lib.rs`** using a `HashMap<i64, Vec<f32>>` store. M4's job is to implement these functions for real in `malus-runtime/src/metal.rs` and update `malus-codegen-cpu` to register the `malus-runtime` function pointers instead of its own stubs.

The JIT resolves these symbols by name via `JITBuilder::symbol()` in `compile_and_run`. No changes to the Cranelift IR are needed — only the symbol targets change. `compile_and_run` now accepts a `&RuntimeSymbols` struct (defined in `malus-codegen-cpu`) of five `extern "C" fn` pointers; the CLI constructs this from `malus-runtime`'s exported functions. This keeps `malus-codegen-cpu` platform-agnostic and Metal-unaware (see ADR-0008).

## Scope

### Metal device setup

The runtime lazily creates a `MetalContext { device, command_queue }` on first use via a process-global `OnceLock`. There is no explicit init API — the first call to any Metal function triggers `Device::system_default()` (panics per ADR-0006 if no Metal device is available). The CLI has no knowledge of Metal.

### Tensor handle

```rust
pub struct TensorBuffer {
    pub buffer: metal::Buffer,   // MTLBuffer with StorageModeShared
    pub dtype: Dtype,
    pub len: usize,              // number of elements
}
```

The opaque `i64` handle that Cranelift-compiled code passes around is a raw pointer to a heap-allocated `TensorBuffer`, cast to `i64`. The runtime owns it.

### C ABI (preserved from M3)

M4 preserves the M3 ABI verbatim — only the implementations change. M5 will migrate `kernel_dispatch` to `kernel_id: u64` / `usize` when the `KernelRegistry` is introduced.

```rust
#[no_mangle]
pub extern "C" fn tensor_alloc_gpu(dtype: i32, len: i64, data: *const f32) -> i64

#[no_mangle]
pub extern "C" fn tensor_free(handle: i64)

#[no_mangle]
pub extern "C" fn tensor_print(handle: i64)

#[no_mangle]
pub extern "C" fn gpu_barrier()

// Stub for M4 — replaced wholesale in M5
#[no_mangle]
pub extern "C" fn kernel_dispatch(name: *const u8, handles: *const i64, n: i32) -> i64
```

### Dtype

`malus-runtime` defines an independent `Dtype` enum (mirroring `ScalarTy`'s discriminant order) with a `from_tag(i32)` constructor. **M4 supports f32 only** — non-f32 dtypes panic with a clear "not yet implemented" message per ADR-0006. The `Dtype` enum and `from_tag` mapping exist for M5, but only `F32` is functional. A drift-detection test asserts all 11 tag mappings.

### `tensor_alloc_gpu`

- `Dtype::from_tag(dtype)` — panics on unknown tag, panics on non-F32
- Allocate `MTLBuffer` with `new_buffer(len * 4, MTLResourceOptions::StorageModeShared)`
- If `data` is non-null, `memcpy` the initial data into the buffer's `contents()` pointer
- Heap-allocate a `TensorBuffer`, return its pointer as `i64`

### `tensor_free`

- Cast `i64` back to `*mut TensorBuffer`, drop it (releases the `MTLBuffer` arc)
- No double-free detection — CTMM is the safety boundary

### `tensor_print`

- Read elements via the buffer's `contents()` pointer (safe on shared memory — no copy needed)
- Print in numpy style: `[1.0, 2.0, 3.0, 4.0]`

### `gpu_barrier`

- Create a command buffer from the queue, commit it, `waitUntilCompleted()`
- No persistent command buffer state — M5 introduces a `current_command_buffer` when `kernel_dispatch` encodes real compute passes

### `kernel_dispatch` (M4 stub)

Return a zeroed output buffer matching the first input's dtype and len (exercises the null-data alloc path). The `name` parameter is ignored. **This stub is replaced wholesale in M5, not extended** — M5's real implementation reads `kernel_id`, looks up the compiled `MTLComputePipelineState`, encodes a compute pass, and dispatches it.

## Metal bindings

Use the `metal` crate (`metal = "0.29"`) for safe Rust bindings to the Metal Objective-C API. This is macOS-only; the entire crate is gated with `#[cfg(target_os = "macos")]` and the `metal` dependency is `target.'cfg(target_os = "macos")'.dependencies`. On non-macOS, `malus-runtime` compiles to an empty crate; the CLI prints "Metal runtime requires macOS" and exits.

## Tests

- `tensor_alloc_gpu` with `[1.0f32, 2.0, 3.0, 4.0]` → handle is non-null, contents readable via raw pointer
- `tensor_print` on the above → runs without panic (content verified by raw-pointer read in round-trip test, not stdout capture)
- `tensor_free` on the handle → no double-free, no Metal validation error
- `gpu_barrier` with an empty command queue → returns without hang
- Allocate 10,000 tensors and free them all → no leak (verify with Metal validation layer)
- `kernel_dispatch` stub → returns a zeroed buffer matching the first input's len
- `Dtype::from_tag` drift test → asserts all 11 tag mappings match `ScalarTy` discriminant order

## Out of scope for M4

- MPS (Metal Performance Shaders) for stdlib ops — deferred to v1
- Multi-buffer command encoding (M5 adds this for kernel dispatch)
- CPU tensor placement (`tensor_alloc_cpu`) — deferred; MVP only uses GPU tensors
- Non-f32 dtypes — enum exists, but only F32 is functional
- Zero-length tensors — Metal's `new_buffer(0, ...)` does not handle zero-length gracefully; not needed for the golden example
