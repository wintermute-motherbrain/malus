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

All runtime calls use the C ABI. Define an `extern "C"` interface in `malus-runtime` that this crate calls into. The interface is declared as Cranelift external function signatures here; the implementations live in M4.

### Script execution entry point

`malus-codegen-cpu` exposes:

```rust
pub fn compile_and_run(program: &TypedProgram) -> Result<(), MalusError>
```

This is what `malus-cli` calls for script execution.

## Dependencies

- `cranelift-codegen`
- `cranelift-jit`
- `cranelift-frontend`
- `cranelift-native` (for host ISA detection)
- `malus-sema` (typed IR)

## Tests

- Compile a minimal `fn` that calls `tensor_print` with a mock runtime stub → executes without panic
- Compile `add_tensors.malus`'s `fn main` with all runtime calls stubbed → correct call sequence: alloc, alloc, dispatch, barrier, print, free, free, free
- Free points appear in the correct order (barrier before free for in-flight tensors)

## Out of scope for M3

- REPL (incremental compilation of expressions) — deferred to v1
- Optimizing passes — Cranelift's default pipeline is sufficient for MVP
- Debug info / DWARF — deferred
