fn make_var(t: Tensor<f32>) -> Variable<f32>:
    let v = variable(t)
    return v

fn main():
    let t = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let v = variable(t)
    let d = v.data
    println("{}", d)
    let v2 = make_var(t)
    println("{}", v2.data)
