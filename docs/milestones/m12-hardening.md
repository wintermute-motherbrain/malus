# M12 — Hardening

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`, `malus-runtime`.

Close three deferred V1 correctness gaps before the autograd work begins: the enum-payload match-binding use-after-free, the zero-length dispatch crash in Metal, and missing `break`/`continue` support.

## Done-When

`examples/hardening.ml` compiles and runs without crashes or leaks:

```malus
enum Wrapper:
    Some(val: Tensor<f32>)
    Empty

fn make() -> Wrapper:
    return Wrapper.Some(val=ones(2, 2))

fn main():
    # break/continue
    let mut acc = 0
    for i in range(10):
        if i == 7:
            break
        if i == 3:
            continue
        acc = acc + i
    println("acc: {}", acc)

    # zero-length tensor (must not crash)
    let empty = zeros(0)
    tensor_print(empty)

    # enum-payload escape (retain-on-escape must fire)
    let w = make()
    let escaped: Tensor<f32>
    match w:
        Some(val):
            escaped = val
        Empty:
            escaped = zeros(1)
    tensor_print(escaped)
```

Expected output:
```
acc: 18
[]
[[1.0, 1.0], [1.0, 1.0]]
```

CTMM emits `tensor_retain` when a match-bound tensor payload escapes its arm; the existing leak-check regression tests (`malus-codegen-cpu/src/tests.rs:801` et al.) remain green.

## Scope

### 1. `break` / `continue`

**AST (`malus-syntax/src/ast.rs`):** Add `StmtKind::Break` and `StmtKind::Continue` (no payload).

**Lexer (`malus-syntax/src/lexer.rs`):** Add `Token::Break` and `Token::Continue` in `scan_ident_or_keyword`.

**Parser (`malus-syntax/src/parser.rs`):** Parse `break` and `continue` as bare statements inside `parse_stmt`. Emit a parse error if used outside a loop body (check a `in_loop` flag threaded through `parse_block`).

**Typed IR (`malus-sema/src/typed_ir.rs`):** Add `TypedStmt::Break` and `TypedStmt::Continue`.

**Sema (`malus-sema/src/check.rs`):** Thread a `loop_depth: usize` counter; type-check `StmtKind::Break`/`Continue` by asserting `loop_depth > 0`, else emit `SemaError::BreakOutsideLoop` / `ContinueOutsideLoop`.

**CTMM (`malus-sema/src/ctmm.rs`):** `break`/`continue` are early-exit points from a loop body. Extend `inject_early_return_unwinds` (currently handles `Return` inside control flow at `:463`) with analogous `inject_break_continue_unwinds`: collect all bindings live in the loop body scope and emit `Drop`/`DropStruct`/`DropEnum`/`DropArray` for them before each `Break`/`Continue` node.

**Codegen-cpu (`malus-codegen-cpu/src/lib.rs`):** Thread the loop's exit block handle (`break_block`) and continue block handle (`continue_block`) through the loop-lowering context. `TypedStmt::Break` → unconditional `ins().jump(break_block, &[])`. `TypedStmt::Continue` → unconditional `ins().jump(continue_block, &[])`. Applies to `For`, `ForIn`, and `While` lowering.

### 2. Zero-Length Dispatch Guard

**Runtime (`malus-runtime/src/metal.rs`):** In `kernel_dispatch` (currently at `:322–377`), after computing `out_len`, add an early-return guard:

```rust
if out_len == 0 {
    return tensor_alloc_zeros_gpu(/* shape of first input */);
}
```

This skips the compute pass entirely and returns a zero-length ready tensor rather than encoding a `dispatchThreads` call with `grid_size = MTLSize::new(0,1,1)`, which Metal aborts.

### 3. Enum-Payload Retain-on-Escape

**CTMM (`malus-sema/src/ctmm.rs`):** In `collect_idents_in_stmt` for `Match` (currently at `:791–794`), track match-arm payload bindings as potential owners. When an arm payload binding is used outside its arm body (detected during `find_last_uses` by seeing it appear as a last-use site outside the `Match` node), emit `TypedStmt::Retain { name }` at the start of the arm, balanced by a `TypedStmt::Release { name }` at the binding's last-use site.

This extends the RC-on-store logic already used for struct construction into the match-arm extraction path. The VJP: if the payload binding's last use is *inside* the arm body, no retain/release is emitted (the enum's `DropEnum` handles the payload release as before).

**Codegen-cpu (`malus-codegen-cpu/src/lib.rs`):** `TypedStmt::Retain` → call `tensor_retain(handle)`. `TypedStmt::Release` → call `tensor_release(handle)`. Both already exist as JIT symbols (`:65–66`); this wires them to the new typed-IR nodes in the match path alongside the struct path (`:1672–1675`).

## Out of Scope

- Full non-f32 dtype support (still panics; post-V3)
- Cross-module struct/enum export (post-V3)
- CTMM barrier coalescing (still conservative; post-V3)
- `ScalarBroadcast` typed IR node (inline scalar-broadcast BinOps still work; post-V3)
