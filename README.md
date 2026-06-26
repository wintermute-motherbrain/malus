# malus

<p align="center">
  <img src="assets/malus_mascot.png" alt="malus mascot" width="300"/>
</p>

A lightweight, high-performance domain-specific language for machine learning workloads on Apple Silicon (M-series) hardware. `malus` uses Python-like syntax with a dual-pipeline compilation model that cleanly separates CPU host orchestration from GPU device execution.

## Key features

- **`fn` / `kernel` split** — `fn` defines a CPU host function JIT-compiled via Cranelift; `kernel` defines a GPU device kernel compiled to Metal Shading Language (MSL)
- **Dual backends** — CPU code is JIT-compiled via [Cranelift](https://cranelift.dev/); GPU code is compiled to MSL and JIT-compiled by the Apple Metal driver
- **Built-in element-wise kernels** — `a + b` on tensors in `fn` bodies automatically synthesizes and dispatches a `malus_add` GPU kernel, indistinguishable from a user-written `kernel`
- **CTMM memory model** — automatic compile-time memory management via escape analysis; static `free`/barrier calls inserted at compile time, no GC, no RC on the fast path
- **Unified memory aware** — explicit placement semantics (`Tensor.gpu(...)`) with zero-copy transfers on Apple Silicon (`StorageModeShared`)

## Example

```malus
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    println(c)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
```

```sh
$ malus examples/add_tensors.ml
[6, 8, 10, 12]
```

Tensor arithmetic also works directly in `fn` bodies via built-in kernels:

```malus
fn add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
```

See [`examples/showcase.ml`](./examples/showcase.ml) for a full demonstration of every MVP capability.

## Status: v0.1 MVP complete

The v0.1 MVP proves the core dual-pipeline model end-to-end:

- [x] M1 — Lexer, parser, AST, pretty-printer, module loader
- [x] M2 — Type checker (`Tensor<dtype>`, scalars, `bool`, tuples), CTMM last-use analysis
- [x] M3 — Cranelift JIT for `fn` bodies
- [x] M4 — Metal runtime (shared buffers, sync barriers, kernel dispatch)
- [x] M5 — MSL codegen for `kernel` bodies (element-wise ops)
- [x] M5.1 — Built-in element-wise kernels for `fn`-body BinOp (`a + b` dispatches `malus_add`)
- [x] M6 — End-to-end: `malus examples/add_tensors.ml` prints `[6, 8, 10, 12]`

Next: M7 — v1 features (zeros/ones, scalar broadcasting, RNG).

## Project structure

```
crates/
  malus-syntax/       # lexer, parser, AST
  malus-loader/       # module resolution + flattening
  malus-sema/         # type checker, CTMM (last-use + barrier insertion)
  malus-codegen-cpu/  # Cranelift JIT for fn bodies
  malus-codegen-gpu/  # MSL generation for kernel + built-in kernels
  malus-runtime/      # Metal API bindings, tensor ops, memory management
  malus-cli/          # script runner, entry point
docs/
  adr/                # architecture decision records
  milestones/         # milestone specs (M1–M7)
  spec/               # language spec
examples/             # malus source files
CONTEXT.md            # domain glossary
```

## Building

```sh
cargo build --release
./target/release/malus examples/add_tensors.ml
```

Requires: Rust 1.78+, macOS 14+ with Xcode command line tools (Metal runtime is macOS-only; non-macOS builds compile but cannot execute GPU code).

## Architecture decisions

See [`docs/adr/`](./docs/adr/) for the key decisions behind malus's design, including dual-pipeline compilation (ADR-0001), CTMM memory model (ADR-0002), panic-only error model (ADR-0006), and built-in kernel id allocation (ADR-0010).
