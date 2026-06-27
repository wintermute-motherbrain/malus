kernel sigmoid_backward(grad_out: Tensor<f32>, sig_z: Tensor<f32>) -> Tensor<f32>:
    return grad_out * sig_z * (1.0 - sig_z)

fn main():
    let x = Tensor.gpu<f32>([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    let target = Tensor.gpu<f32>([[0.0], [1.0], [1.0], [0.0]])
    let ones41 = Tensor.gpu<f32>([[1.0], [1.0], [1.0], [1.0]])

    let mut w1 = Tensor.gpu<f32>([
        [0.5, -0.6, 0.7, -0.4, 0.3, -0.5, 0.6, -0.3],
        [-0.4, 0.5, -0.6, 0.7, -0.3, 0.4, -0.5, 0.6]
    ])
    let mut b1 = Tensor.gpu<f32>([[0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]])
    let mut w2 = Tensor.gpu<f32>([
        [0.6], [-0.5], [0.4], [-0.6], [0.5], [-0.4], [0.6], [-0.5]
    ])
    let mut b2 = Tensor.gpu<f32>([[0.0]])

    let lr = 1.5

    for step in range(10000):
        let z1 = x @ w1 + ones41 @ b1
        let h = sigmoid(z1)
        let z2 = h @ w2 + ones41 @ b2
        let out = sigmoid(z2)

        let diff = out - target
        let loss = sum(diff * diff)

        if step == 0:
            println("step 0: loss = {}", loss)
        if step == 500:
            println("step 500: loss = {}", loss)
        if step == 1000:
            println("step 1000: loss = {}", loss)
        if step == 2000:
            println("step 2000: loss = {}", loss)
        if step == 5000:
            println("step 5000: loss = {}", loss)
        if step == 9999:
            println("step 9999 (final): loss = {}", loss)
            println("predictions: {}", out)

        let dout = 2.0 * diff
        let dz2 = sigmoid_backward(dout, out)
        let dw2 = transpose(h) @ dz2
        let db2 = transpose(ones41) @ dz2
        let dh = dz2 @ transpose(w2)
        let dz1 = sigmoid_backward(dh, h)
        let dw1 = transpose(x) @ dz1
        let db1 = transpose(ones41) @ dz1

        w1 = w1 - lr * dw1
        b1 = b1 - lr * db1
        w2 = w2 - lr * dw2
        b2 = b2 - lr * db2
