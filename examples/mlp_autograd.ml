fn main():
    let x      = Tensor.gpu<f32>([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    let target = Tensor.gpu<f32>([[0.0], [1.0], [1.0], [0.0]])
    let ones41 = Tensor.gpu<f32>([[1.0], [1.0], [1.0], [1.0]])

    let mut w1 = variable(Tensor.gpu<f32>([
        [0.5, -0.6, 0.7, -0.4, 0.3, -0.5, 0.6, -0.3],
        [-0.4, 0.5, -0.6, 0.7, -0.3, 0.4, -0.5, 0.6]
    ]))
    let mut b1 = variable(Tensor.gpu<f32>([[0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]]))
    let mut w2 = variable(Tensor.gpu<f32>([
        [0.6], [-0.5], [0.4], [-0.6], [0.5], [-0.4], [0.6], [-0.5]
    ]))
    let mut b2 = variable(Tensor.gpu<f32>([[0.0]]))

    let lr = 1.5

    let vx     = variable(x)
    let vones  = variable(ones41)
    let vtgt   = variable(target)

    for step in range(10000):
        let z1   = vx @ w1 + vones @ b1
        let h    = sigmoid(z1)
        let z2   = h @ w2 + vones @ b2
        let out  = sigmoid(z2)
        let diff = out - vtgt
        let loss = sum(diff * diff)

        zero_grad(w1, b1, w2, b2)
        backward(loss)

        with no_grad:
            w1 = variable(w1.data - lr * w1.grad)
            b1 = variable(b1.data - lr * b1.grad)
            w2 = variable(w2.data - lr * w2.grad)
            b2 = variable(b2.data - lr * b2.grad)

        if step == 0:
            println("step 0: loss = {}", loss.data)
        if step == 500:
            println("step 500: loss = {}", loss.data)
        if step == 9999:
            println("step 9999 (final): loss = {}", loss.data)

    let p1 = sigmoid(x @ w1.data + ones41 @ b1.data)
    let p2 = sigmoid(p1 @ w2.data + ones41 @ b2.data)
    println("predictions: {}", p2)
