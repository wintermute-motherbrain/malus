# Explicit tensor placement, implicit transfer at fn/kernel boundary

Users specify where tensors are created (`Tensor.cpu(...)`, `Tensor.gpu(...)`), but the compiler inserts transfers automatically when a tensor crosses the `fn`/`kernel` boundary. On Apple Silicon, this is implemented with `MTLResourceStorageModeShared` buffers and sync barriers — no physical copy occurs. The placement abstraction is kept portable so the semantics remain valid if non-unified architectures are ever targeted.

## Considered Options

- **Fully implicit placement**: Hides too much for a performance-oriented language. ML researchers care about where their data lives because it affects memory pressure and access patterns, even on unified memory.
- **Fully explicit transfers**: Requires manual `copy_to_gpu()` / `copy_to_cpu()` calls at every boundary — too much boilerplate for a Python-like language and easy to get wrong.
