# 01 — Overview

## What malus is

malus is a lightweight, high-performance domain-specific language for machine learning workloads on Apple Silicon (M-series) hardware. It provides Python-like syntax with a dual-pipeline compilation model that cleanly separates CPU host orchestration from GPU device execution.

malus is compiled, not interpreted. CPU code is JIT-compiled via Cranelift; GPU code is compiled to Metal Shading Language (MSL) and JIT-compiled by the Apple Metal driver. There is no separate build step — running `malus script.malus` compiles and executes immediately.

## Target user

ML researchers and power-users on M-series Macs who want to write custom GPU-accelerated ML code without leaving a Python-like comfort zone. The analogy is Triton for CUDA: researchers who need custom kernels but don't want to write raw Metal.

malus is not a general-purpose language. It is not designed for web servers, GUI applications, or systems programming.

## Design philosophy

**Familiarity over novelty.** Syntax and semantics follow Python and NumPy conventions wherever there is no compelling reason to diverge. ML researchers should feel at home within minutes.

**Explicit over implicit for performance-critical decisions.** Tensor placement (CPU vs GPU) is explicit at creation. Device transfers, memory barriers, and thread dispatch are handled by the compiler but their effects are visible and predictable.

**Compile-time memory management on the fast path.** CTMM (Compile-Time Memory Management) eliminates runtime allocation overhead for the common case (linear tensor flows). Reference counting is a fallback, not the default.

**The `fn`/`kernel` split is the architecture.** Every function in malus is either a CPU host function (`fn`) or a GPU device kernel (`kernel`). This distinction drives the entire compilation model.

## Scope

### v0.1 MVP
Proves the dual-pipeline model end-to-end: parse a malus script, type-check it, JIT-compile the `fn` body via Cranelift, generate MSL for the `kernel` body, dispatch it via Metal, and print the result. Tensor math only; minimal stdlib.

### v1
A usable ML research tool: structs, enums, full numpy-equivalent stdlib, autograd-ready design, rich error messages, terminal REPL, SafeTensors interop.

### Explicit non-goals
- **Autograd / differentiation** — designed to be layered on later; not in v1
- **Distributed training** — out of scope
- **Non-Apple-Silicon targets** — the language abstracts placement portably, but only Apple Silicon is an active target
- **String processing** — string literals exist for `print` and file I/O; no `String` type
- **General-purpose programming** — data structures, networking, concurrency beyond GPU dispatch are out of scope
