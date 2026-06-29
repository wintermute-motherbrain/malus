# ADR-0023 — reshape zero-copy aliasing memory model

**Status**: Accepted (M17); overrules M17 spec text

## Context

The M17 spec described `reshape` as "share the box via `tensor_retain` + CTMM `Release`". This is incoherent: a reshaped tensor needs its own `shape: Vec<usize>` field (the whole point of reshape is to change the shape), so there is no single `TensorBuffer` box to share. The retain/release wording seemed to describe something like Python's view semantics but did not describe a valid implementation.

The question was: what is the correct memory model for reshape?

## Decision

`tensor_reshape` constructs a **new, independent `TensorBuffer`** whose `metal::Buffer` field is a clone of the input's `metal::Buffer` handle.

`metal::Buffer::clone()` is an Obj-C `retain` on the same underlying `MTLBuffer` — no data is copied. The two `TensorBuffer` structs exist independently on the heap, each with its own `shape`, `len`, and `ref_count: AtomicUsize::new(1)`.

Each is freed independently via the normal `tensor_free → tensor_release` path. The `MTLBuffer` itself is freed (via Obj-C ARC) when the last `Buffer` clone is dropped. Under M17 immutability, writes to one view cannot corrupt the other.

```rust
pub(crate) fn reshape_to(handle: i64, new_shape: &[usize]) -> i64 {
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let new_tb = TensorBuffer {
        buffer: tb.buffer.clone(),   // Obj-C retain on the same MTLBuffer
        dtype: tb.dtype,
        len: tb.len,
        shape: new_shape.to_vec(),
        ref_count: AtomicUsize::new(1),
    };
    Box::into_raw(Box::new(new_tb)) as i64
}
```

## Consequences

- Reshape is zero-copy: no GPU buffer allocation, no data movement.
- CTMM needs no special-casing. The reshaped handle is a normal tensor; `tensor_free` is inserted by CTMM exactly as for any other tensor.
- The two `TensorBuffer`s alias the same `MTLBuffer`. This is safe under M17 because tensors are immutable after construction: no op writes through one view and reads through another.
- When mutation is added (M20 lvalue assignment), this aliasing must be revisited — writes through a reshaped view would silently corrupt the original. A copy-on-write or uniqueness-check strategy will be needed at that point.
- The spec text about "share the box via `tensor_retain`" is incorrect and is overruled by this ADR.

## Alternatives rejected

- **Allocate a new MTLBuffer (true copy)**: correct but wastes GPU memory and bandwidth for a shape change that carries no semantic mutation.
- **Embed a pointer-to-parent + shape offset in `TensorBuffer` (true strided view)**: correct and powerful, but requires every op to handle the strided case and is the full `view` story deferred post-V3 per ADR-0022.
- **Share the TensorBuffer box via RC (the spec's intention, roughly)**: requires a single `TensorBuffer` to have two shapes simultaneously, which is a contradiction.
