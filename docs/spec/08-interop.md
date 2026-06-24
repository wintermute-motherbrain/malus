# 08 — Interoperability

## Philosophy

malus is not an island, but its interop surface is deliberately narrow in v1. ML researchers need to load pretrained weights and exchange tensors with their existing Python tooling. They do not need malus to be embeddable in Python or callable from arbitrary C code in v1.

## SafeTensors `[v1]`

[SafeTensors](https://github.com/huggingface/safetensors) is the primary tensor exchange format. It is memory-mapped, pickle-free, and supports all dtypes malus targets. It is the emerging standard in the Hugging Face ecosystem for model weight distribution.

```malus
let weights = load("model.safetensors")     # returns a struct-like mapping
let w0 = weights["layer0.weight"]           # Tensor<f32>, GPU placement
save(tensors, "output.safetensors")         # write a set of named tensors
```

Loading places tensors on GPU by default (leveraging shared memory — the file is mmap'd and the buffer is handed to Metal directly where possible).

### Format details

- Header is JSON: maps string keys to dtype, shape, and byte offsets
- Data region is raw bytes, contiguous per tensor
- malus parses the header in Rust (in `malus-runtime`), validates dtypes against malus's supported set, and allocates `MTLBuffer` for each tensor

## NumPy `.npy` / `.npz` `[v1]`

NumPy's `.npy` (single array) and `.npz` (zip archive of arrays) formats are the secondary interop target, primarily for dataset loading.

```malus
let data = load_npy("data.npy")       # returns Tensor<dtype> — dtype from file header
let arrays = load_npz("dataset.npz")  # returns a mapping of name → Tensor
```

NumPy format support is read-only in v1. Writing `.npy` is a future addition. `f64` arrays in NumPy files are rejected with a clear error (malus does not support `f64`).

## PyTorch `.pt` files

Not supported. PyTorch `.pt` files use Python pickle, which requires a Python interpreter to parse safely. Use SafeTensors for weight exchange with PyTorch models (Hugging Face's `safetensors` library converts `.pt` to `.safetensors` in one line).

## Python runtime interop `[future]`

A Python C extension (`import malus`) that allows calling malus kernels from Python and receiving results as NumPy arrays is a high-value future addition. It is not in v1.

The JIT pipeline is designed with a future C ABI in mind: `fn` and `kernel` functions can be exposed as `extern "C"` symbols. The runtime's tensor handle (`i64`) is designed to be castable to a pointer that a C extension can wrap. This is an intentional design constraint, not an afterthought.

## No string type

malus has no `String` type. String *literals* are valid in source code but only as arguments to specific stdlib functions (`print`, `load`, `save`). They are not first-class values and cannot be assigned to `let` bindings or passed to user-defined functions.

This is intentional: malus is an ML DSL, not a general-purpose language. Adding a heap-allocated string type would require integrating it with CTMM's escape analysis and RC for minimal practical benefit.
