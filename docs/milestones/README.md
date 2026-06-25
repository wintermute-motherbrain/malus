# v0.1 MVP Milestones

Six milestones in compiler dependency order. Each milestone produces a working, testable artifact before the next begins.

| # | Status | Milestone | Crate(s) | Deliverable |
|---|--------|-----------|----------|-------------|
| [M1](./m1-syntax.md) | ✅ done | Syntax | `malus-syntax` | Parse `add_tensors.ml` to a valid AST |
| [M2](./m2-semantics.md) | ✅ done | Semantics | `malus-sema` | Type-check the AST; CTMM inserts free points |
| [M3](./m3-cpu-codegen.md) | ✅ done | CPU Codegen | `malus-codegen-cpu` | Cranelift JIT executes a simple `fn` body |
| [M4](./m4-metal-runtime.md) | ✅ done | Metal Runtime | `malus-runtime` | Allocate a GPU tensor buffer; round-trip data |
| [M5](./m5-gpu-codegen.md) | **← next** | GPU Codegen | `malus-codegen-gpu` | Generate and dispatch an element-wise MSL kernel |
| [M6](./m6-integration.md) | | Integration | `malus-cli` | `malus examples/add_tensors.ml` prints `[6, 8, 10, 12]` |

## Definition of done for v0.1

Running `malus examples/add_tensors.ml` on an M-series Mac:
- Creates two `f32` tensors on the GPU
- Dispatches a user-written `add` kernel compiled from malus source to MSL
- Prints the result tensor to stdout
- Exits cleanly with no leaks (CTMM frees both input tensors after the kernel barrier)
