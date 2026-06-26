# malus Milestones

## v0.1 MVP — Complete

Six milestones in compiler dependency order. All done.

| # | Milestone | Crate(s) | Deliverable |
|---|-----------|----------|-------------|
| [M1](./m1-syntax.md) ✅ | Syntax | `malus-syntax`, `malus-loader` | Parse `add_tensors.ml` to a valid AST |
| [M2](./m2-semantics.md) ✅ | Semantics | `malus-sema` | Type-check the AST; CTMM inserts free points |
| [M3](./m3-cpu-codegen.md) ✅ | CPU Codegen | `malus-codegen-cpu` | Cranelift JIT executes a simple `fn` body |
| [M4](./m4-metal-runtime.md) ✅ | Metal Runtime | `malus-runtime` | Allocate a GPU tensor buffer; round-trip data |
| [M5](./m5-gpu-codegen.md) ✅ | GPU Codegen | `malus-codegen-gpu` | Generate and dispatch an element-wise MSL kernel |
| [M5.1](./m5.1-builtin-elementwise-kernels.md) ✅ | Built-in element-wise kernels | `malus-codegen-gpu`, `malus-codegen-cpu` | `a + b` on tensors in `fn` bodies dispatches a built-in GPU kernel |
| [M6](./m6-integration.md) ✅ | Integration | `malus-cli` | `malus examples/add_tensors.ml` prints `[6, 8, 10, 12]` |

## V1 — In Progress

See [v1-plan.md](./v1-plan.md) for the full V1 vision, design decisions, and done-when program.

| # | Milestone | Crate(s) | Theme |
|---|-----------|----------|-------|
| [M7](./m7-kernel-thickening.md) **← next** | Kernel Thickening | `malus-syntax`, `malus-sema`, `malus-codegen-gpu`, `malus-codegen-cpu` | Multi-statement kernels, `let mut`, scalar broadcasting |
| [M8](./m8-core-stdlib.md) | Core Stdlib | `malus-runtime`, `malus-sema`, `malus-codegen-gpu`, `malus-codegen-cpu` | matmul (`@`), relu/sigmoid/tanh/exp/log/sqrt/abs, transpose, zeros/ones, sum, shape metadata |
| [M9](./m9-control-flow.md) | Control Flow | `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime` | if/else, for, while, CTMM RC fallback for conditional tensor lifetimes |
| [M10](./m10-structs-enums.md) | Structs + Enums | `malus-syntax`, `malus-sema`, `malus-codegen-cpu` | Struct and enum declarations, field access, data-carrying enums, match |
| [M11](./m11-mlp.md) | The 2-Layer MLP | All crates | Fixed-length arrays, rich diagnostics, 2-layer MLP forward+backward done-when |

## V1 Definition of Done

`malus examples/mlp.ml` runs on an M-series Mac, printing decreasing loss over 10 training steps. The program implements a two-layer MLP with a manual forward pass, backward pass (gradient computation by hand), and a gradient descent update loop. See [v1-plan.md](./v1-plan.md) for the full done-when program.

## Other Documents

- [ctmm-v1-gaps.md](./ctmm-v1-gaps.md) — known CTMM limitations and which milestones address them
- [m5.2-scalar-broadcasting.md](./m5.2-scalar-broadcasting.md) — scalar broadcasting spec (subsumed into M7)
