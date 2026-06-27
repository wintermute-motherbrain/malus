# M18 — Transformer Stdlib

**Crates:** `malus-sema`, `malus-codegen-cpu`, `malus-runtime`.

Add the differentiable ops every transformer block needs: `softmax`, `layernorm`, `gelu`, `cross_entropy`, and causal masking. All include VJPs. The done-when is a full transformer block (self-attention + MLP + LayerNorm residuals) that forward-backward gradient-checks.

## Done-When

`examples/transformer_block.ml` compiles and gradient-checks:

```malus
fn attention(q: Variable<f32>, k: Variable<f32>, v: Variable<f32>,
             mask: Tensor<f32>) -> Variable<f32>:
    let scale = 1.0 / 2.8284
    let scores = (q @ transpose(k, 0, 2, 1)) * scale + variable(mask)
    let attn = softmax(scores, axis=2)
    return attn @ v

fn mlp_block(x: Variable<f32>, w1: Variable<f32>, w2: Variable<f32>) -> Variable<f32>:
    return gelu(x @ w1) @ w2

fn main():
    let B = 2
    let T = 4
    let C = 8

    let x  = variable(ones(B, T, C))
    let wq = variable(ones(C, C))
    let wk = variable(ones(C, C))
    let wv = variable(ones(C, C))
    let w1 = variable(ones(C, C))
    let w2 = variable(ones(C, C))
    let mask = causal_mask(T)

    let q = x @ wq
    let k = x @ wk
    let v = x @ wv
    let attn_out = attention(q, k, v, mask)
    let mlp_out  = mlp_block(attn_out, w1, w2)
    let normed   = layernorm(mlp_out, axis=2)
    let logits   = reshape(normed, B * T, C)
    let targets  = zeros_int(B * T)
    let loss     = cross_entropy(logits, targets)

    backward(loss)
    tensor_print(wq.grad)
    println("transformer block: OK")
```

All parameter gradients are non-zero and match finite differences to 1e-3.

## Scope

### 1. `softmax(t, axis=N)`

**Builtins:** Register `softmax(t: Variable<f32>, axis: i32) -> Variable<f32>`.

**Runtime:** `tensor_softmax_axis` — compute `exp(x - max(x, axis, keepdim=true)) / sum(exp(...), axis, keepdim=true)` in an eager CPU loop (numerically stable max-subtract). Output shape = input shape.

**VJP:** Standard Jacobian-vector product: `dx = softmax_out * (dout - sum(dout * softmax_out, axis, keepdim=true))`. Saved tensor for backward: the softmax output.

### 2. `layernorm(t, axis=N)`

**Builtins:** Register `layernorm(t: Variable<f32>, axis: i32) -> Variable<f32>`. No learnable scale/bias in M18 (those are parameters the user provides multiplied elementwise after the call).

**Runtime:** `tensor_layernorm_axis` — `(x - mean(x, axis)) / sqrt(var(x, axis) + 1e-5)` in an eager CPU loop.

**VJP:** The layernorm VJP in terms of saved `mean`, `var`, and `norm` output. Saved tensors: `mean`, `var`, `norm_out`.

### 3. `gelu(t)`

**Builtins:** Register `gelu(t: Variable<f32>) -> Variable<f32>`. Uses the tanh approximation: `0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))`.

**Runtime:** `tensor_gelu` — eager CPU elementwise. This is an element-wise op so it can also be synthesized as an MSL kernel via the existing builtin-kernel infrastructure (codegen-gpu); use whichever path is simpler to implement.

**VJP:** Analytically differentiated version of the tanh-GELU approximation. Saved tensor: input `x`.

### 4. `cross_entropy(logits, targets)`

**Builtins:** Register `cross_entropy(logits: Variable<f32>, targets: Tensor<i32>) -> Variable<f32>`. `logits` is `[N, C]` (float); `targets` is `[N]` (integer index tensor from M19 — declare the signature now, defer i32 tensor support to M19, use a float index placeholder in M18 if i32 tensors aren't available yet). Output is a scalar `[1]` loss.

**Runtime:** `tensor_cross_entropy(logits_handle, targets_handle) -> i64` — `softmax(logits)` then `-log(softmax[i, target[i]])` averaged over N. Eager CPU.

**VJP:** `d_logits[i, j] = (softmax[i,j] - 1_{j == target[i]}) / N`. Saved tensors: `softmax_out`, `targets`.

### 5. `causal_mask(T)`

**Builtins:** Register `causal_mask(T: i32) -> Tensor<f32>`. Returns a `[T, T]` upper-triangular mask of `-inf` (large negative float) and `0.0`. Not differentiable (pure constant).

**Runtime:** `tensor_causal_mask(T: usize) -> i64` — allocates a `[T, T]` buffer and fills the upper triangle (above the diagonal) with `-1e9` and lower-or-equal with `0.0`.

## Out of Scope

- Learnable LayerNorm scale and bias as builtin arguments (they're user-managed `Variable` params multiplied/added after the call)
- Flash attention / online softmax (post-V3)
- `cross_entropy` with `ignore_index` or label smoothing (post-V3)
- `gelu` exact form (uses tanh approx as in GPT-2)
- RMS norm (post-V3)
