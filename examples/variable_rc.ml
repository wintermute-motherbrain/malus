fn wrap(t: Tensor<f32>) -> Variable<f32>:
    return variable(t)

fn identity(v: Variable<f32>) -> Variable<f32>:
    return v

fn main():
    let a = variable(ones(2, 2))
    let b = identity(a)
    let c = variable(zeros(3, 3))
    tensor_print(b.data)
    tensor_print(c.data)
