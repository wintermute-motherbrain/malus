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

- `RuntimeSymbols` gains `malus_read_file` and the three string primitives (`malus_str_len`, `malus_str_char_at`, `malus_str_from_char`), plus `malus_str_box` for lowering `Lit::Str` to a StrBox handle.
- `ResolvedTy::Str` is a new type; codegen represents strings as an `i64` handle to a heap-allocated `StrBox { ptr, len }` (uniform with tensor/buffer handles). `make_drop_stmt_for_ty` returns `None` for `Str` — string handles are intentionally leaked for the whole-program lifetime.
- The leaking read (`std::fs::read_to_string`, then leaked) is intentional: the buffer lives for the program's duration, consistent with the single-pass JIT execution model where `fn main()` runs once.
- String escape / UTF-8 handling is best-effort: `str_char_at` returns Unicode scalar values; multi-byte char iteration is user responsibility.
- **Also added in M22** (extending this ADR's I/O scope): `Buffer<i32>` mutable staging container (`malus_buffer_i32`, `malus_buffer_get_i32`, `malus_buffer_set_i32`, `malus_buffer_free`, `malus_buffer_freeze_i32`); `freeze(buf) -> Tensor<i32>` (non-destructive copy); `tensor.data[i]` flat element read (`malus_tensor_get_f32`); `rand_uniform() -> f32` via Philox (ADR-0024 counter domain 1); `rand_int(n: i64) -> i64` uniform integer [0, n).
- Post-V3, a proper string type with ownership, a `File` type with seek/close, and SafeTensors I/O would be separate milestones.
