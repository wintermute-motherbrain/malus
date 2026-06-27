fn main():
    let scores = [0.9, 0.4, 0.7, 0.2, 0.8]
    let idx = 2
    let v = scores[idx]
    println("scores[{}] = {}", idx, v)

    let mut total = 0.0
    for s in scores:
        total = total + s
    println("sum = {}", total)

    let ts = [Tensor.gpu<f32>([1.0, 2.0]), Tensor.gpu<f32>([3.0, 4.0])]
    print(ts[0])
    print(ts[1])
