# malus

<p align="center">
  <img src="assets/malus_mascot.png" alt="malus mascot" width="300"/>
</p>

A lightweight, high-performance domain-specific language for machine learning workloads on Apple Silicon (M-series) hardware. `malus` uses Python-like syntax with a dual-pipeline compilation model that cleanly separates CPU host orchestration from GPU device execution.

## Key features

- **`fn` / `kernel` split** — `fn` defines a CPU host function; `kernel` defines a GPU device kernel
- **Dual backends** — CPU code is JIT-compiled via [Cranelift](https://cranelift.dev/); GPU code is compiled to Metal Shading Language (MSL) and JIT-compiled by the Apple Metal driver
- **CTMM memory model** — automatic compile-time memory management via escape analysis; reference counting fallback only when lifetimes are structurally ambiguous
- **Unified memory aware** — explicit placement semantics (`Tensor.cpu(...)` / `Tensor.gpu(...)`) with zero-copy transfers on Apple Silicon

## Example

```malus
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    print(c)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
```

## Status: pre-alpha (v0.1 MVP in progress — next: M3 CPU Codegen)

The v0.1 MVP proves the core dual-pipeline model end-to-end:

- [x] M1 — Lexer, parser, AST, pretty-printer, module loader
- [x] M2 — Type checker (`Tensor<dtype>`, scalars, `bool`, tuples), CTMM last-use analysis
- [ ] **M3** — Cranelift JIT for `fn` bodies ← next
- [ ] M4 — Metal runtime (shared buffers, sync barriers, kernel dispatch)
- [ ] M5 — MSL codegen for `kernel` bodies (element-wise ops)
- [ ] M6 — End-to-end: `malus examples/add_tensors.malus` prints result

## Project structure

```
crates/
  malus-syntax/       # lexer, parser, AST
  malus-sema/         # type checking, escape analysis, CTMM
  malus-codegen-cpu/  # Cranelift JIT for fn bodies
  malus-codegen-gpu/  # MSL generation for kernel bodies
  malus-runtime/      # Metal API bindings, tensor ops, memory management
  malus-cli/          # script runner, REPL, entry point
docs/adr/           # architecture decision records
examples/           # malus source files
CONTEXT.md          # domain glossary
```

## Building

```sh
cargo build
```

Requires: Rust 1.78+, macOS 14+ with Xcode command line tools (for Metal).

## Architecture decisions

See [`docs/adr/`](./docs/adr/) for the key decisions behind malus's design.
