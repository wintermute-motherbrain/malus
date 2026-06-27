fn main():
    let x = Tensor.gpu<f32>([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    let w = Tensor.gpu<f32>([[1.0, 0.0], [0.0, 1.0]])
    let out = x @ w
    print(out)
    let n = out.len
    println("elements: {}", n)
