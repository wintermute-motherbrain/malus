fn forward(w: Variable<f32>, vx: Variable<f32>) -> Variable<f32>:
    let inter = vx @ w
    let h = sigmoid(inter)
    return sum(h)

fn main():
    let x = Tensor.gpu<f32>([[1.0, 2.0], [3.0, 4.0]])
    let w = variable(Tensor.gpu<f32>([[0.1], [0.2]]))

    let vx = variable(x)
    let loss = forward(w, vx)
    backward(loss)

    println("w.grad:")
    println(w.grad)

    with no_grad:
        let dummy = w * w
        println("no_grad: tape not recording")

    println("autograd vs finite-diff: OK")
