# M4 — Metal Runtime

**Crate:** `malus-runtime`
**Done when:** A Rust test allocates a `f32` GPU tensor via the runtime, writes values from the CPU, reads them back, and frees the buffer — with no Metal validation errors.

## Scope

### Metal device setup

Initialize once at process start (or on first use):
- `MTLCreateSystemDefaultDevice()` → `MTLDevice`
- `[device newCommandQueue]` → `MTLCommandQueue`

Expose via a process-global singleton (`OnceLock` or similar). The CLI creates the device; the runtime uses it.

### Tensor handle

Define the internal tensor representation:

```rust
pub struct TensorBuffer {
    pub buffer: metal::Buffer,   // MTLBuffer with StorageModeShared
    pub dtype: Dtype,
    pub len: usize,              // number of elements
}
```

The opaque `i64` handle that Cranelift-compiled code passes around is a raw pointer to a heap-allocated `TensorBuffer`, cast to `i64`. The runtime owns it.

### C ABI surface (called from Cranelift-compiled code)

```rust
#[no_mangle]
pub extern "C" fn tensor_alloc_gpu(dtype: u8, len: usize, data_ptr: *const u8) -> i64

#[no_mangle]
pub extern "C" fn tensor_free(handle: i64)

#[no_mangle]
pub extern "C" fn tensor_print(handle: i64)

#[no_mangle]
pub extern "C" fn gpu_barrier()

// Stub for M4 — real implementation added in M5
#[no_mangle]
pub extern "C" fn kernel_dispatch(kernel_id: u64, handles: *const i64, count: usize) -> i64
```

### `tensor_alloc_gpu`

- Allocate `MTLBuffer` with `newBufferWithLength:options:` using `MTLResourceStorageModeShared`
- If `data_ptr` is non-null, `memcpy` the initial data into the buffer's `contents()` pointer
- Heap-allocate a `TensorBuffer`, return its pointer as `i64`

### `tensor_free`

- Cast `i64` back to `*mut TensorBuffer`, drop it (releases the `MTLBuffer` arc)

### `tensor_print`

- Read elements via the buffer's `contents()` pointer (safe on shared memory — no copy needed)
- Print in numpy style: `[1.0, 2.0, 3.0, 4.0]`

### `gpu_barrier`

- Commit the current command buffer and wait for completion:
  `commandBuffer.commit(); commandBuffer.waitUntilCompleted()`
- Reset the command buffer for the next dispatch

### `kernel_dispatch` (M4 stub)

Return a zeroed output buffer of the same shape as the first input. The real implementation is added in M5.

## Metal bindings

Use the `metal` crate (`metal = "0.29"`) for safe Rust bindings to the Metal Objective-C API. This is macOS-only; gate the entire crate with `#[cfg(target_os = "macos")]`.

## Tests

- `tensor_alloc_gpu` with `[1.0f32, 2.0, 3.0, 4.0]` → handle is non-null, contents readable
- `tensor_print` on the above → prints `[1.0, 2.0, 3.0, 4.0]`
- `tensor_free` on the handle → no double-free, no Metal validation error
- `gpu_barrier` with an empty command queue → returns without hang
- Allocate 10,000 tensors and free them all → no leak (verify with Metal validation layer)

## Out of scope for M4

- MPS (Metal Performance Shaders) for stdlib ops — deferred to v1
- Multi-buffer command encoding (M5 adds this for kernel dispatch)
- CPU tensor placement (`tensor_alloc_cpu`) — deferred; MVP only uses GPU tensors
