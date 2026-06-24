# M3 — CPU Codegen

**Crate:** `malus-codegen-cpu`
**Done when:** A `fn` body from the M2 typed IR is JIT-compiled via Cranelift and executes correctly, including calling into `malus-runtime` for tensor allocation and kernel dispatch stubs.

## Scope

### Cranelift JIT pipeline

1. Translate the `TypedProgram`'s `fn` items into Cranelift IR (`cranelift-codegen` + `cranelift-jit`)
2. Declare external symbols for runtime functions (`tensor_alloc_gpu`, `kernel_dispatch`, `tensor_print`, `tensor_free`)
3. JIT-compile to native code for the host architecture (aarch64 on Apple Silicon)
4. Return a function pointer for `main` that the CLI can call

### Lowering rules for the MVP

| malus construct | Cranelift lowering |
|---|---|
| `let x = Tensor.gpu<f32>([...])` | Call `tensor_alloc_gpu(dtype, len, data_ptr)` → opaque `i64` handle |
| `let c = add(a, b)` | Call `kernel_dispatch(kernel_id, [a, b])` → `i64` handle |
| `print(c)` | Call `tensor_print(c)` |
| `FreePoint(x)` after GPU barrier | Call `tensor_free(x)` |
| `GpuBarrier` | Call `gpu_barrier()` |

Tensors are represented as opaque `i64` handles throughout Cranelift IR — the actual `MTLBuffer` pointer lives in the runtime. The codegen crate never touches Metal directly.

### ABI

All runtime calls use the C ABI. Declare as Cranelift external functions in `malus-codegen-cpu`; implement (or stub) in `malus-runtime`. For M3 the implementations can be no-ops or simple print-and-return stubs — the goal is a working Cranelift pipeline, not a working Metal stack.

```c
// Allocate a GPU tensor from a flat data array.
// dtype_tag: 0=f32, 1=f16, 2=i32, ... (match ScalarTy order)
// data: pointer to len floats (callee copies before returning)
// returns: opaque i64 handle
i64  tensor_alloc_gpu(i32 dtype_tag, i64 len, const float* data);

// Dispatch a named kernel over a list of tensor handles.
// name: null-terminated UTF-8 kernel name
// handles: array of nhandles opaque i64 tensor handles (inputs)
// returns: output tensor handle
i64  kernel_dispatch(const char* name, const i64* handles, i32 nhandles);

// Block until all in-flight GPU work completes.
void gpu_barrier(void);

// Print tensor contents to stdout.
void tensor_print(i64 handle);

// Free a tensor handle and its backing buffer.
void tensor_free(i64 handle);
```

### Script execution entry point

`malus-codegen-cpu` exposes:

```rust
pub fn compile_and_run(program: &TypedProgram) -> Result<(), MalusError>
```

This is what `malus-cli` calls for script execution.

## Dependencies

Add to `crates/malus-codegen-cpu/Cargo.toml`:

```toml
cranelift-codegen  = "0.113"
cranelift-frontend = "0.113"
cranelift-jit      = "0.113"
cranelift-native   = "0.113"
cranelift-module   = "0.113"
malus-sema         = { path = "../malus-sema" }
```

## Tests

- Compile a minimal `fn` that calls `tensor_print` with a mock runtime stub → executes without panic
- Compile `add_tensors.ml`'s `fn main` with all runtime calls stubbed → correct call sequence: alloc, alloc, dispatch, barrier, print, free, free, free
- Free points appear in the correct order (barrier before free for in-flight tensors)

## Out of scope for M3

- REPL (incremental compilation of expressions) — deferred to v1
- Optimizing passes — Cranelift's default pipeline is sufficient for MVP
- Debug info / DWARF — deferred
