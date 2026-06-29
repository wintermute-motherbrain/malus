# Cross-Module Types

**Status:** Post-V3 (required follow-up; user decision M20)

**Crates:** `malus-loader`, `malus-sema`, `malus-codegen-cpu`

## Motivation

Currently, struct and enum definitions cannot be imported across module boundaries. The `exported_names` filter in `malus-loader/src/lib.rs:193-198` only includes `Fn` and `Kernel` items — `Struct` and `Enum` items are silently ignored. As a result, a type defined in one module cannot be used as a parameter type or return type in another.

This forces AdamW and similar stdlib constructs to be inlined into every file that uses them (see `examples/adamw.ml`). M22's nanoGPT capstone may also need to import struct-defined layer or config types.

## Gap Details

**`malus-loader/src/lib.rs:193-198`** — `exported_names` match arm:
```rust
// Current: only Fn/Kernel are exported
Item::Fn(f) => Some(f.name.clone()),
Item::Kernel(k) => Some(k.name.clone()),
_ => None,   // <-- Struct/Enum silently dropped
```

**Additional gaps (not yet prototyped):**
1. `exported_names` must include `Item::Struct` and `Item::Enum`.
2. Struct/Enum definitions from imported modules must be merged into the consuming module's flattened `Program` before sema (the loader currently only flattens `Fn`/`Kernel` items).
3. `malus-sema` must resolve imported type names in struct field types, function parameter types, and return types.
4. Name collision handling: two imported modules defining the same struct name.
5. Cross-module enums in `match` arms.

## Fix Sketch

1. **Loader**: include `Struct`/`Enum` in `exported_names`; flatten imported type defs into the consuming `Program` before returning `LoadedProgram`.
2. **Sema**: no changes needed if the loader flattens correctly — sema already processes `Item::Struct`/`Item::Enum` from the program's item list.
3. **Codegen**: no changes — codegen never sees raw AST types (only typed IR).
4. **Name collision**: simplest first pass — last-wins (deterministic import order); escalate to error in a follow-up.

## Done-When

`examples/stdlib/adamw.ml` defines `struct AdamW` and `fn adamw_step`, and a separate `examples/train.ml` imports and uses them without needing to inline the definitions.
