kernel relu_backward(grad_out: Tensor<f32>, x: Tensor<f32>) -> Tensor<f32>:
    let mask = x > 0.0
    return grad_out * mask

fn main():
    let mut acc = Tensor.gpu<f32>([0.0, 0.0, 0.0])
    let delta = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    acc = acc + delta
    acc = acc + delta
    println("accumulated: {}", acc)
    let scaled = acc * 0.5
    println("scaled: {}", scaled)
    let grad_out = Tensor.gpu<f32>([1.0, 1.0, 1.0])
    let x = Tensor.gpu<f32>([0.5, -0.5, 0.0])
    let grad_in = relu_backward(grad_out, x)
    println("relu_backward: {}", grad_in)
