# 03 — Type System

## Overview

malus uses static typing with type inference inside function bodies. Function and kernel signatures require explicit type annotations. `let` bindings infer their type from the right-hand side.

## Built-in types

### Tensor `[MVP]`

`Tensor<dtype>` is malus's core built-in primitive. It is not a library type — the compiler has deep knowledge of tensors for memory management (Lobster) and code generation (MSL).

- **dtype** is static and known at compile time
- **shape** is dynamic and validated at runtime
- Every tensor has a **placement** (CPU or GPU), tracked by the compiler

```malus
let a: Tensor<f32> = Tensor.gpu<f32>([1.0, 2.0, 3.0])
let b = Tensor.cpu<i32>([1, 2, 3])   # placement inferred from constructor
```

Supported dtypes:

| dtype | Description |
|---|---|
| `f32` | 32-bit float |
| `f16` | 16-bit float |
| `bf16` | bfloat16 (M-series native) |
| `i8` | 8-bit signed integer |
| `i16` | 16-bit signed integer |
| `i32` | 32-bit signed integer |
| `i64` | 64-bit signed integer |
| `u8` | 8-bit unsigned integer |
| `u16` | 16-bit unsigned integer |
| `u32` | 32-bit unsigned integer |
| `u64` | 64-bit unsigned integer |

`f64` is intentionally omitted: Apple's GPU architecture does not natively support double-precision, and adding a CPU-only `f64` tensor would break the clean `fn`/`kernel` model.

### Scalar types `[MVP]`

Scalar types correspond to tensor dtypes but represent single values:

`f32`, `f16`, `bf16`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`

Scalars and tensors of matching dtype interoperate in arithmetic expressions (scalar is broadcast across the tensor).

### Bool `[MVP]`

`bool` — the values `true` and `false`. Used for conditionals and boolean tensor masks.

### Tuple `[MVP]`

A fixed-length, heterogeneous product type. Types are structural (no named tuple type).

```malus
let pair: (Tensor<f32>, i32) = (my_tensor, 42)
let (t, n) = pair   # destructuring
```

Tuples are the primary way to return multiple values from a function.

### Struct `[v1]`

User-defined named product types. Fields are immutable after construction.

```malus
struct AdamConfig:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32

let cfg = AdamConfig(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
let lr = cfg.lr
```

Structs are nominal types. Two structs with identical fields are distinct types.

Structs cannot contain tensors directly in v1 — a struct field of type `Tensor<f32>` triggers the CTMM RC fallback (see section 04). This is correct and expected; model weight containers will naturally use RC.

### Enum `[v1]`

User-defined tagged unions (algebraic data types). Variants may carry data.

```malus
enum Optimizer:
    SGD(lr: f32, momentum: f32)
    Adam(AdamConfig)
    RMSProp(lr: f32, alpha: f32)

let opt = Optimizer.Adam(AdamConfig(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8))
```

Pattern matching on enums uses `if`/`else` with variant checks in v1. A `match` expression is a future addition.

### Option `[v1]`

`Option<T>` is the canonical enum for nullable values. It is defined in the stdlib as:

```malus
enum Option<T>:
    Some(T)
    None
```

`Option` is the recommended way to handle recoverable absence — malus has no exceptions and no null. This composes with the panic-only error model: functions that might not have a result return `Option<T>`; functions that must succeed or die panic.

Note: `Option` requires generics, which land in v1 alongside enums. In v0.1 MVP, there is no `Option`.

## Type inference

Inside a function body, `let` bindings infer their type:

```malus
fn example():
    let x = 1.0           # inferred: f32
    let a = Tensor.gpu<f32>([1.0, 2.0])  # inferred: Tensor<f32>, placement Gpu
    let b = a + a         # inferred: Tensor<f32>
```

Function and kernel signatures are always fully annotated — inference does not cross function boundaries.

## Type rules for operators

| Expression | Constraint | Result type |
|---|---|---|
| `a + b` (and `-`, `*`, `/`) | `a` and `b` same dtype tensor or scalar | same dtype tensor |
| `a @ b` | `a: Tensor<T>`, `b: Tensor<T>`, both float dtype | `Tensor<T>` |
| scalar `op` tensor | scalar broadcast | same dtype tensor |
| `a == b` | same type | `bool` or `Tensor<bool>` |
| `a[i]` | `a: Tensor<T>`, `i: i32` or `Tensor<bool>` | `Tensor<T>` |

dtype mixing in binary ops is a compile-time error. Explicit casting is required:

```malus
let c = a + b.to<f32>()   # cast b from f16 to f32 before adding
```

## Generics `[future]`

Generics (type parameters on `fn`, `kernel`, `struct`, `enum`) are deferred beyond v1. They are required for `Module` trait abstractions and generic optimizers. The type system is designed so generics can be added without breaking existing code.
