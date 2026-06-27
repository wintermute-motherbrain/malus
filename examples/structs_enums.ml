struct Layer:
    weights: Tensor<f32>
    bias: Tensor<f32>

enum Activation:
    Relu
    Sigmoid

enum Scale:
    Identity
    Factor(s: f32)

fn activate(x: Tensor<f32>, act: Activation) -> Tensor<f32>:
    match act:
        Relu:
            return relu(x)
        Sigmoid:
            return sigmoid(x)

fn scale_output(x: Tensor<f32>, s: Scale) -> Tensor<f32>:
    match s:
        Identity:
            return x
        Factor(v):
            return x * v

fn linear(x: Tensor<f32>, layer: Layer, act: Activation, s: Scale) -> Tensor<f32>:
    let h = x @ layer.weights + layer.bias
    let a = activate(h, act)
    return scale_output(a, s)

fn main():
    let l = Layer(weights=ones(3, 4), bias=zeros(1, 4))
    let x = ones(2, 3)
    let out = linear(x, l, Activation.Relu, Scale.Factor(s=2.0))
    println("scaled linear+relu output: {}", out)
