# CTMM barrier insertion uses a GPU-pending set, not in-flight-before-drop

M5 replaces CTMM's original barrier-insertion logic (which only inserted barriers before frees of in-flight tensor inputs) with a linear-scan GPU-pending set. A tensor binding is "GPU-pending" if it has been produced or consumed by a `KernelCall` since the last `GpuBarrier`. Any CPU-side access of a pending tensor — free, read (print/println), or return — triggers a barrier before that statement.

## Why the old logic was insufficient

The original CTMM tracked a flat set of in-flight input bindings and inserted barriers only at drop sites (last-use indices). If a kernel's output was read on the CPU before any in-flight input was dropped — e.g., `let c = add(a, b); println(c); use(a)` — the barrier would land too late (at `a`'s drop, after `println(c)`), producing stale reads. The golden example worked by accident: the inputs' last use was the kernel call itself, so the barrier landed immediately after, before the output was read.

## Decision

Replace the flat in-flight set and grouped-by-index barrier insertion with a two-phase approach: (1) drop insertion (unchanged — find last uses, insert `Drop` statements), then (2) a linear scan maintaining a `HashSet<String>` of pending bindings. `KernelCall` adds input and output names to the set. `GpuBarrier` clears it. Any non-kernel statement referencing a pending binding triggers a barrier and clears the set. `Return` of a `KernelCall` always triggers a barrier (the output is handed to the caller).

## Considered Options

- **Defer (ship M5 with known limitation)**: Rejected — the gap manifests in natural patterns like "print results then reuse inputs." An MVP that only works for one exact example is a demo, not an MVP.
- **Make `tensor_print` implicitly flush**: Rejected — only fixes `print`, not arbitrary CPU reads. Papers over the real issue and becomes dead code once proper barrier insertion lands.
- **Coexist (old + new logic)**: Rejected — produces redundant barriers requiring deduplication. The new pending-set logic is strictly more general; a `Drop` of a pending tensor is a CPU read and is already covered.

## Consequences

Barriers may be inserted between chained kernel calls when a drop of an in-flight input lands between them, causing two command buffers instead of one. This is correct but suboptimal — a future optimization can defer drops past the last kernel call in a chain. The distinction between "in-flight tensor" (GPU reading) and "pending tensor" (GPU writing) is preserved in the glossary for future precision-barrier optimizations, even though the current implementation treats both uniformly.
