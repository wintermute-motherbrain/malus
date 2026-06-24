# M7 — v1 Features

This milestone covers everything deferred from the v0.1 MVP. It is intentionally a collection of parallel workstreams rather than a single sequential milestone — each section below can be worked independently after M6 passes.

**Done when:** `malus examples/train_step.malus` — a forward pass through a two-layer MLP using custom kernels, structs for model config, and SafeTensors weight loading — runs correctly on an M-series Mac.

---

## M7a — Type system: structs and enums

**Crate:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`

Implement user-defined product and sum types.

### Structs

```malus
struct LinearConfig:
    in_features: i32
    out_features: i32
    bias: bool
```

- Named fields with explicit types
- Constructed with keyword arguments: `LinearConfig(in_features=128, out_features=64, bias=true)`
- Field access: `cfg.in_features`
- Fields are immutable after construction
- Structs containing `Tensor` fields trigger Lobster RC (by design — model parameter storage)

### Enums

```malus
enum Activation:
    ReLU
    GELU
    SiLU(beta: f32)
```

- Variants may carry data (named fields)
- Variant construction: `Activation.ReLU`, `Activation.SiLU(beta=1.0)`
- Pattern matching via `if`/`else` with variant checks in v1; `match` expression is future

### Option\<T\>

`Option<T>` is defined in the stdlib as a generic enum. Requires generics (see M7f). Provides `Some(value)` and `None` variants for explicit absence.

---

## M7b — Full stdlib

**Crate:** `malus-runtime`, `malus-codegen-gpu`

Implement the numpy-equivalent core defined in [spec 06](../spec/06-stdlib.md).

### Priority order

1. **Mathematical functions** — `exp`, `log`, `sqrt`, `abs`, `relu`, `sigmoid`, `tanh`, `max`, `min`, `clip`
2. **Reductions** — `sum`, `mean`, `max`, `min` with optional `dim` argument — MPS-dispatched
3. **Matrix operations** — `a @ b` (MPS `MPSMatrixMultiplication`), `transpose`
4. **Shape manipulation** — `reshape`, `flatten`, `squeeze`, `unsqueeze`, `concat`, `stack`
5. **Indexing and slicing** — full NumPy bracket syntax including boolean indexing
6. **Casting** — `a.to<dtype>()`
7. **Shape inspection** — `a.shape`, `a.ndim`, `a.len`, `a.dtype`

### MPS integration

`matmul` and reductions delegate to `MPSNDArray` operations. The runtime must:
- Initialize `MPSNDArrayDescriptor` from the tensor's shape and dtype
- Dispatch `MPSNDArrayMatrixMultiplication` or reduction ops
- Wrap the output `MTLBuffer` in a malus tensor handle

---

## M7c — Lobster RC fallback

**Crate:** `malus-sema`, `malus-runtime`

Implement reference counting for tensors that escape static analysis (struct fields, dynamic arrays).

- In `malus-sema`: detect structurally ambiguous lifetimes during escape analysis; annotate them as RC-managed rather than statically-freed
- In `malus-runtime`: add atomic refcount to `TensorBuffer`; implement `tensor_retain(handle)` and `tensor_release(handle)` in the C ABI
- In `malus-codegen-cpu`: emit `retain`/`release` calls at struct field assignments and drops

---

## M7d — Rich error diagnostics

**Crate:** `malus-sema`, `malus-cli`

Upgrade error reporting to Rust/Elm-style rich diagnostics:

```
error: dtype mismatch in binary op
  --> script.malus:5:13
   |
 5 |     let c = a + b
   |             ^~~~^ left is Tensor<f32>, right is Tensor<f16>
   |
   = help: cast with b.to<f32>()
```

- Source file, line, column with the offending span underlined
- Plain-English error messages with concrete types/shapes
- `help:` suggestions for common mistakes (dtype mismatch, shape mismatch, wrong argument count)
- Multiple errors reported per pass where possible (error recovery in the type checker)

Runtime panics print concrete shapes, dtypes, and values:

```
panic: shape mismatch in matmul
  left:  [32, 64]
  right: [128, 64]
  dim 1 of left (64) must equal dim 0 of right (128)
```

---

## M7e — GPU RNG (Philox)

**Crate:** `malus-codegen-gpu`, `malus-runtime`

Implement `rand` and `randn` as built-in kernels using the Philox counter-based PRNG.

- Each GPU thread derives its random stream from `seed + thread_id` — no shared state, SIMT-safe
- `rand(shape, seed)` → uniform `[0, 1)` as `Tensor<f32>`
- `randn(shape, seed)` → standard normal via Box-Muller transform
- Deterministic given the same seed; reproducibility is a first-class concern

The Philox kernel is implemented as a built-in MSL kernel in `malus-codegen-gpu`, not user-written malus code.

---

## M7f — Kernel annotations and intrinsics

**Crate:** `malus-syntax`, `malus-sema`, `malus-codegen-gpu`

Expose fine-grained GPU control for power users.

### Annotations (launch configuration)

```malus
@threadgroup_size(16, 16)
@shared_memory(tile_a: f32[16][16])
kernel tiled_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    ...
```

- `@threadgroup_size` sets the `MTLSize` for threadgroup dispatch
- `@shared_memory` allocates named threadgroup-local memory accessible in the kernel body
- Validated at compile time against Metal limits (max threads per threadgroup, max shared memory size)

### Intrinsics

Built-in functions inside kernel bodies:

```malus
thread_id()          # u32 — 1D thread index in grid
thread_id_2d()       # (u32, u32) — 2D thread index
threadgroup_id()     # u32 — threadgroup index
thread_in_group()    # u32 — thread index within threadgroup
simd_shuffle(v, l)   # shuffle value across SIMD lanes
simd_sum(v)          # SIMD-group reduction
```

Using intrinsics in `fn` bodies is a compile-time error.

### `inout` parameters

```malus
kernel relu(inout a: Tensor<f32>):
    a = max(a, 0.0)
```

- `inout` tensors are mutated in-place; no output buffer is allocated
- Lobster does not insert a free for the `inout` input

---

## M7g — Import aliasing (`as` syntax)

**Crate:** `malus-syntax`, `malus-loader`, `malus-sema`

Add `as` aliasing to resolve naming conflicts between imported modules or names.

```malus
import ops as o          # qualified: o.add(...)
from ops import add as my_add   # unqualified: my_add(...)
```

**Context:** v0.1 detects naming conflicts (two imports exporting the same name) as errors and requires the user to use qualified imports instead. Aliasing is the ergonomic solution.

**What's needed:**
- Lexer: add `as` keyword
- Parser/AST: extend `import_stmt` and `from_import_stmt` grammar with optional `as ident`
- Loader: store alias in `LoadedProgram.module_aliases` under the alias name instead of the module name
- Sema: resolve qualified calls using the alias

---

## M7h — Terminal REPL

**Crate:** `malus-cli`

Interactive session with the same JIT pipeline as script execution.

- `malus` (no arguments) drops into REPL
- Block mode: lines ending in `:` collect until a blank line
- Persistent GPU context and tensor bindings across inputs
- Errors do not terminate the session
- `quit` or `Ctrl+D` exits

---

## M7h — SafeTensors and NumPy interop

**Crate:** `malus-runtime`

File I/O for tensor exchange:

- `load("file.safetensors")` → mapping of name → `Tensor<dtype>`, GPU placement
- `save(tensors, "file.safetensors")` → write named tensors
- `load_npy("file.npy")` → `Tensor<dtype>` (read-only)
- `load_npz("file.npz")` → mapping of name → `Tensor<dtype>` (read-only)

Dependencies: `safetensors` Rust crate, custom `.npy` header parser (the format is simple enough to avoid a heavy dependency).

---

## v1 definition of done

A representative end-to-end script runs correctly:

```malus
struct MLPConfig:
    hidden: i32
    output: i32

fn mlp_forward(x: Tensor<f32>, w1: Tensor<f32>, w2: Tensor<f32>, cfg: MLPConfig) -> Tensor<f32>:
    let h = relu(x @ w1)
    return h @ w2

fn main():
    let weights = load("weights.safetensors")
    let cfg = MLPConfig(hidden=128, output=10)
    let x = randn((32, 784), seed=42)
    let logits = mlp_forward(x, weights["w1"], weights["w2"], cfg)
    print(logits.shape)
```
