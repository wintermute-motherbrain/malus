# 06 — Standard Library

## Implementation note

Stdlib functions fall into two categories:
- **CPU-implemented** — called directly from Cranelift-compiled `fn` bodies
- **MPS-dispatched** — delegated to Apple's Metal Performance Shaders for optimized GPU execution

User-written `kernel` functions always go through malus's MSL codegen pipeline. MPS is only used for stdlib ops — it is never exposed to user code.

## Tensor creation `[MVP]`

```malus
zeros(shape: ...)  -> Tensor<f32>          # all zeros, GPU placement
ones(shape: ...)   -> Tensor<f32>          # all ones, GPU placement
zeros<dtype>(shape: ...) -> Tensor<dtype>  # [v1] with explicit dtype
ones<dtype>(shape: ...)  -> Tensor<dtype>  # [v1] with explicit dtype
```

Shape is specified as a sequence of integer arguments: `zeros(3, 4)` → shape `[3, 4]`.

## Element-wise operations `[MVP]`

These are dispatched as MSL compute kernels (either via MPS or malus's own codegen):

```malus
a + b    # add
a - b    # subtract
a * b    # multiply (element-wise)
a / b    # divide
```

All standard Python/NumPy broadcasting rules apply (see Broadcasting below).

## Mathematical functions `[v1]`

```malus
exp(a)      sqrt(a)     log(a)      log2(a)
abs(a)      sin(a)      cos(a)      tanh(a)
relu(a)     sigmoid(a)
max(a, b)   min(a, b)   clip(a, lo, hi)
```

All operate element-wise on tensors. Scalars are broadcast.

## Matrix operations `[v1]`

```malus
a @ b              # matrix multiply — MPS-dispatched
transpose(a)       # reverse last two dimensions
transpose(a, 0, 2) # swap specified dimensions
```

`a @ b` requires both tensors to have a float dtype. It delegates to `MPSMatrixMultiplication` for optimized execution on Apple Silicon.

## Reductions `[v1]`

```malus
sum(a)            # sum all elements → scalar
sum(a, dim=0)     # sum along axis 0 → tensor
mean(a)
mean(a, dim=1)
max(a)
max(a, dim=0)
min(a)
min(a, dim=0)
```

## Shape manipulation `[v1]`

```malus
reshape(a, 4, 8)        # new shape must have same total elements
flatten(a)              # reshape to 1D
squeeze(a, dim=0)       # remove a size-1 dimension
unsqueeze(a, dim=0)     # insert a size-1 dimension
transpose(a)
concat([a, b, c], dim=0)
stack([a, b, c], dim=0) # new axis
```

## Indexing and slicing `[v1]`

Full NumPy-style bracket syntax:

```malus
a[0]          # first element along axis 0
a[1:3]        # slice axis 0 from index 1 to 3 (exclusive)
a[:, 0]       # all rows, column 0
a[mask]       # boolean indexing — mask is Tensor<bool>
a[-1]         # last element
a[::2]        # every other element
```

Slices return a view where possible. Boolean indexing always returns a copy.

## Broadcasting `[v1]`

malus follows NumPy broadcasting rules exactly:

1. Right-align shapes
2. Dimensions are compatible if they are equal or one of them is 1
3. The output shape is the element-wise maximum of the input shapes

```malus
# [3, 1] + [1, 4] → [3, 4]
# [5, 3, 1] + [3, 4] → [5, 3, 4]
```

Incompatible shapes produce a runtime panic with both shapes printed.

## Random number generation `[v1]`

malus uses a Philox counter-based PRNG — the same algorithm used by PyTorch and JAX. Philox generates reproducible, parallel-safe random streams: each GPU thread derives its stream from `seed + thread_id`, with no shared state between threads.

```malus
rand(shape, seed: i64)    -> Tensor<f32>   # uniform [0, 1)
randn(shape, seed: i64)   -> Tensor<f32>   # standard normal
rand<dtype>(shape, seed)  -> Tensor<dtype> # with explicit dtype
```

`seed` is required and explicit. There is no global random state. For reproducible experiments, always pass a fixed seed. For variation across runs, pass a runtime-generated value.

## Type casting `[v1]`

```malus
a.to<f16>()    # cast tensor to a different dtype
a.to<i8>()     # lossy casts are allowed — no implicit conversion
```

## Printing `[MVP]`

```malus
print(a)              # prints tensor in numpy style: [1.0, 2.0, 3.0]
print("label:", a)    # string literal followed by tensor (string literal only, not a type)
```

## Shape inspection `[v1]`

```malus
a.shape     # returns a tuple of i64
a.ndim      # number of dimensions
a.dtype     # returns the dtype as a string literal (for printing only)
a.len       # total number of elements
```
