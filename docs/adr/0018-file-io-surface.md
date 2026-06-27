# Minimal file I/O surface for data loading

Qualifies ADR-0006 (panic-only error model, no I/O).

## Decision

In M22, add a single `read_file(path: str) -> str` builtin and three string-walking primitives (`str_len`, `str_char_at`, `str_from_char`). These are the only I/O primitives in V3. Everything else — tokenization, vocab construction, batch iteration — is written in malus using existing language features.

## Why this is surprising

ADR-0006 and the V1 "What it does NOT include" list explicitly exclude file I/O. Every V1 and V2 example embeds its data as inline tensor literals. The V3 capstone (tiny Shakespeare) requires real text data that cannot reasonably be embedded inline (~1 MB). A `read_file` builtin is the minimum necessary to make the capstone self-contained.

The scope fence is strict: `read_file` panics on any error (consistent with ADR-0006's panic-only model — no `Result` type). There is no write, no seek, no directory listing, no network access, no binary file format. SafeTensors checkpoint I/O is explicitly excluded.

## Why not embed the data as generated literals

The dataset is ~1 MB of text. Embedding it as a literal would produce a source file that is larger than the runtime, makes every compilation load 1 MB of parser output, and defeats the point of demonstrating that malus can train on external data.

## Consequences

- `RuntimeSymbols` gains `malus_read_file` (and the three string primitives).
- `ResolvedTy::Str` is used for string return types; codegen represents strings as `(ptr: i64, len: i64)` in Cranelift SSA. String types are not first-class in the broader language — they appear only as return values of `read_file` and as arguments to `println` (which already handles `Str`).
- The leaking read (`std::fs::read_to_string`, then leaked) is intentional: the buffer lives for the program's duration, consistent with the single-pass JIT execution model where `fn main()` runs once.
- String escape / UTF-8 handling is best-effort: `str_char_at` returns Unicode scalar values; multi-byte char iteration is user responsibility.
- Post-V3, a proper string type with ownership, a `File` type with seek/close, and SafeTensors I/O would be separate milestones.
