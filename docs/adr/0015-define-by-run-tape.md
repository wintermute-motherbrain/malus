# Define-by-run runtime tape for autograd

Supersedes the "dataflow liveness RC fallback" framing in ADR-0002 and the "V2 optimization" deferral in ADR-0014.

## Decision

Implement autograd as a define-by-run (dynamic) runtime tape living in `malus-runtime`. Forward ops on `Variable` values push nodes onto a global thread-local tape; `backward(loss)` walks it in reverse calling per-op VJP rules. VJPs for all V1 ops are hardcoded in Rust; user-defined gradients are deferred.

## Why this is surprising

The M9/M10 planning docs (ADR-0014, `docs/milestones/ctmm-v1-gaps.md`) framed the deferred RC fallback as a future CTMM analysis — a compiler-level answer to ambiguous tensor lifetimes in conditional paths. V2 does not build that analysis. Instead, the RC problem is dissolved by the distinct `Variable` type (ADR-0016): RC is type-directed, not analysis-directed, so the old "dataflow liveness RC fallback" is no longer needed for the autograd use case. A future reader seeing no general RC fallback in the compiler should check ADR-0016 before assuming it is an oversight.

## Considered alternatives

**Compile-time source-to-source autodiff (Enzyme/Zygote-style).** The compiler differentiates a forward `fn`'s typed IR into an adjoint backward `fn` at compile time. Maximally "compiled-language" flavored, zero runtime tape overhead. Rejected because differentiating through control flow, managing activation storage, and handling non-differentiable ops is a research-grade undertaking that would stall the milestone cadence for multiple quarters.

**Library-level autograd in malus.** Add closures and growable structures so a user implements micrograd as a malus library. Purest reading of "could someone build micrograd on this?" but scalar-valued graphs can't reach transformer throughput and front-loads large language work (closures, heap graphs, `Vec<T>`) unrelated to autograd.

## Consequences

- `malus-runtime` grows a `tape.rs` module with `TapeNode`, `TAPE: thread_local`, `backward`, `tape_push/clear/pause/resume`, per-op VJP closures, and leaf-gradient accumulation.
- `RuntimeSymbols` gains new fn pointers for tape-recording op wrappers.
- `malus-codegen-cpu` emits tape-recording calls for `Variable`-typed ops instead of the raw `tensor_*` calls.
- The global-tape model means `backward` is not re-entrant and multiple concurrent tapes are not supported. This matches the V3 target (single-model training loop) and is consistent with the existing `OnceLock<MetalContext>` global runtime design.
- Tape auto-clears after `backward` (`retain_graph=False` default). Retained-graph / double-backward is post-V3.
