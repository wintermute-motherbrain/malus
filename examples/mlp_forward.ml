fn forward(x: Tensor<f32>, w1: Tensor<f32>, w2: Tensor<f32>) -> Tensor<f32>:
    let h = relu(x @ w1)
    return h @ w2

fn main():
    let x = ones(2, 3)
    let w1 = ones(3, 4)
    let w2 = ones(4, 2)
    let out = forward(x, w1, w2)
    println("forward output: {}", out)
    let s = sum(out)
    println("sum: {}", s)
    let wt = transpose(w1)
    println("transpose done")
    println("exp result: {}", exp(x))
