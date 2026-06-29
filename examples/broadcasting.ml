fn main():
    # Broadcasting: (1,8) + (4,8) → (4,8).  Bias broadcast for a linear layer.
    let b = variable(ones(1, 8))
    let m = variable(ones(4, 8))

    let out = m + b
    let loss = sum(mean(out, axis=0))
    backward(loss)

    println("b.grad:")
    println(b.grad)
    println("m.grad:")
    println(m.grad)
    println("broadcasting + axis reductions: OK")
