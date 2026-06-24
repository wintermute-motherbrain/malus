# malus Language Specification

This directory contains the authoritative specification for the malus programming language. It captures all design decisions made during the initial design session and supersedes any earlier informal descriptions.

## Sections

| File | Covers |
|---|---|
| [01-overview.md](./01-overview.md) | Goals, non-goals, target user, design philosophy |
| [02-syntax.md](./02-syntax.md) | Lexical structure, INDENT/DEDENT, grammar, operators |
| [03-types.md](./03-types.md) | Type system: tensors, scalars, tuples, structs, enums, Option |
| [04-memory.md](./04-memory.md) | CTMM memory model: escape analysis, RC fallback, GPU boundary |
| [05-functions.md](./05-functions.md) | `fn` and `kernel` declarations, ownership, device placement |
| [06-stdlib.md](./06-stdlib.md) | Built-in tensor operations, dtypes, broadcasting, RNG |
| [07-execution.md](./07-execution.md) | Compilation pipeline, script execution, REPL |
| [08-interop.md](./08-interop.md) | SafeTensors, NumPy .npy, future C ABI |

## Version

This specification describes **malus v1**. Features marked `[MVP]` are implemented in v0.1. Features marked `[v1]` are planned for the first full release.
