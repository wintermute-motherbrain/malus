# M21 — MPS Migration

**Crates:** `malus-runtime`.  **Status:** ✅ done (two-commit approach).

## What shipped

**Commit 1 — objc2-metal port (zero behavior change):** Replaced the deprecated
`metal-rs 0.29` / `objc 0.2` / `foreign-types 0.5` stack with `objc2 0.6` +
`objc2-metal 0.3` + `objc2-foundation 0.3`. All Metal types are now
`Retained<ProtocolObject<dyn MTL…>>`; the `msg_send![…, retain]` dance in
`kernel_dispatch` was eliminated; `buffer.contents()` returns `NonNull<c_void>` and
requires `.as_ptr()` before casting. 379 existing tests stayed green.

**Commit 2 — MPS matmul:** Replaced the CPU triple-loop `tensor_matmul` with
`MPSMatrixMultiplication` from `objc2-metal-performance-shaders 0.3`. Three shape
regimes: 2-D, 3-D batched (loop per batch slice), and 3-D⊗2-D broadcast. The
implementation is **eager**: calls `gpu_barrier()` first, encodes all MPS ops into a
fresh command buffer, commits, and waits. Returns a ready tensor; CTMM and codegen-cpu
are unchanged. The old CPU impl is kept as `tensor_matmul_cpu` for differential tests.

## Scope changes from the original spec

The original spec said pending tensors and included axis reductions and optionally
transpose. After grilling, the scope was narrowed:

- **Eager not pending** — the 10× speedup comes from AMX compute, not command-buffer
  batching. Chained-GPU batching is marginal for malus because the softmax/layernorm/
  gelu ops between matmuls are CPU-eager and would force barriers anyway.
- **matmul only** — axis reductions and transpose stay CPU loops; they are not on the
  critical matmul path.

## Done-when

`examples/mps_bench.ml` compiles, produces correct results, and demonstrates speedup:

```malus
fn main():
    let a = randn(512, 512)
    let b = randn(512, 512)
    let c = a @ b
    tensor_print(c)

    let a3 = randn(8, 512, 512)
    let b3 = randn(8, 512, 512)
    let c3 = a3 @ b3
    tensor_print(c3)

    println("MPS matmul: OK")
```

MPS results match the CPU reference within 1e-3 (verified in `tests.rs` for all three
regimes). The speedup test (`#[test] #[ignore]`) asserts ≥10× on M-series.

## Out of scope (post-V3)

- MPS for softmax, layernorm, GELU, cross-entropy
- MPS for axis reductions (`sum`, `mean`, `max`, `var`) and transpose
- Pending-tensor MPS (command-buffer batching across chained ops)
- Non-f32 MPS dispatch
