# M12: sema-reject escaping non-tensor struct/enum payloads; unify heap-box RC at M13

When a `match` arm binds a struct or enum payload (e.g. `Some(pt: Point)` or `Some(inner: Inner)`), the bound value is an alias into the heap box allocated for the matched enum/struct. The heap box has no reference count — `DropEnum`/`DropStruct` simply calls `free` on it. If `pt` or `inner` escape the arm (returned, assigned out, passed to a call), they alias freed memory once the matched value is dropped.

## Decision

**In M12**, make it a hard compile error (`SemaError::NonTensorPayloadEscapes`) for a match-arm binding whose resolved type is `Struct` or `Enum` to appear in any position that would outlive the arm:

- `Return { expr }` containing the bare binding name
- `Assign { expr }` RHS containing the bare binding name
- A `Call` argument containing the bare binding name

`FieldAccess` and `Index` on the bound name (e.g. `pt.x`, `inner[0]`) are explicitly allowed — they read *through* the alias without moving it out.

**In M13** (the `Variable` type milestone), add a reference count (`AtomicUsize`) to every struct/enum heap box and a shared `retain`/`release` ABI for aggregates, retiring the sema rejection. Until M13, the compile error is the correct behavior.

Tensor payloads (`Some(val: Tensor<f32>)`) are exempted from the error — tensors already have refcounts and the retain-on-escape path is handled by CTMM (`annotate_match_arms` prepends `Retain` for tensor payloads; the Retain + Assign pattern correctly balances `DropEnum`).

## Why this is surprising

A reader seeing a struct-valued match binding fail to compile would expect this to work — other languages (Rust, Swift) handle exactly this pattern. The error is a sequencing artifact: tensor boxes have had refcounts since M10 (needed for `DropStruct`/`DropEnum` tensor field release), but aggregate boxes were never given one because no M10–M11 program ever needed to escape an aggregate payload. The refcount gap only surfaced in M12 when the escape path was explicitly enumerated.

## Why reject at sema rather than silently copying

Silent structural copy would change semantics for programs that want shared mutable state (post-V2). A hard compile error preserves semantic clarity and gives a precise note pointing to M13, rather than compiling silently and producing bugs.

## Considered alternatives

- **Add aggregate RC in M12:** Would require adding `AtomicUsize` to every struct/enum heap box and updating all `DropStruct`/`DropEnum` sites, the codegen `StructInit`/`EnumInit` call sites, and match-arm binding lowering. That is the M13 scope — conflating it with M12 hardening would dilute both milestones. The M12 compile error costs nothing at runtime and keeps M12 focused.
- **Silent structural copy:** Semantically wrong for non-trivially-copyable types (anything containing a tensor). Would silently double-free tensor fields.
- **Unique ownership with explicit move:** Would require a `move` keyword in match arms, a larger syntax surface, and a borrow checker — post-V3 scope.
