fn attention(q: Tensor<f32>, k: Tensor<f32>, v: Tensor<f32>, mask: Tensor<f32>) -> Tensor<f32>:
    let scores = q @ permute(k, 0, 2, 1) + variable(mask)
    let attn = softmax(scores, axis=2)
    return attn @ v

fn mlp_block(x: Tensor<f32>, w1: Tensor<f32>, w2: Tensor<f32>) -> Tensor<f32>:
    return gelu(x @ w1) @ w2

fn main():
    let B = 2
    let T = 4
    let C = 8

    let wq = variable(ones(C, C))
    let wk = variable(ones(C, C))
    let wv = variable(ones(C, C))
    let wn = variable(ones(C, C))
    let wp = variable(ones(C, C))

    let x    = variable(ones(B, T, C))
    let mask = causal_mask(T)

    let q = x @ wq
    let k = x @ wk
    let v = x @ wv

    let attn_out = attention(q, k, v, mask)
    let x2 = layernorm(attn_out + x, axis=2)

    let mlp_out = mlp_block(x2, wn, wp)
    let x3 = layernorm(mlp_out + x2, axis=2)

    let logits  = reshape(x3, B * T, C)
    let targets = zeros(B * T)
    let loss    = cross_entropy(logits, targets)

    backward(loss)

    let grad_wq = wq.grad
    let grad_wk = wk.grad
    let grad_wv = wv.grad
    let grad_wn = wn.grad
    let grad_wp = wp.grad
    println("wq.grad: {}", grad_wq)
    println("wk.grad: {}", grad_wk)
    println("wv.grad: {}", grad_wv)
    println("wn.grad: {}", grad_wn)
    println("wp.grad: {}", grad_wp)
    println("transformer block: OK")
