# Function-pointer injection for runtime symbol binding

`malus-codegen-cpu` does not depend on `malus-runtime`. Instead, `compile_and_run` accepts a `RuntimeSymbols` struct of five `extern "C" fn` pointers, injected by the caller (the CLI on macOS, mock functions in tests). This keeps the codegen crate platform-agnostic and Metal-unaware, allowing it to compile and test on non-macOS platforms.

## Considered Options

- **Direct dependency** (`codegen-cpu` → `malus-runtime`): Rejected — couples the Cranelift JIT crate to the macOS-only `metal` crate, breaking cross-platform compilation and isolating codegen tests.
- **cfg-gated dual impl** (stubs on Linux, real on macOS): Rejected — two runtime implementations to maintain; the stubs are dead code on macOS.
- **Separate `malus-runtime-abi` crate** for shared types: Rejected — over-engineered for a single struct; the struct belongs in its consumer.
- **Function-pointer injection** (chosen): Consumer owns the contract (`RuntimeSymbols` in codegen-cpu); producer (`malus-runtime`) exposes bare `extern "C" fn`s; CLI is the composition root. Zero coupling between codegen and runtime.

## Consequences

The CLI is the composition root: it constructs `RuntimeSymbols` by passing `malus_runtime`'s exported function pointers. Tests construct mock symbols from local stub functions. Adding a new backend (e.g. a CPU-only runtime or a mock for fuzzing) requires no codegen changes — only a new `RuntimeSymbols` construction site. M5's ABI migration (`kernel_id: u64` / `usize`) will require updating the `RuntimeSymbols` struct definition in codegen-cpu and all construction sites — a localized, mechanical change.
