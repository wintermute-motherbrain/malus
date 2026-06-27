fn main():
    let x = ones(2, 3)
    let w = ones(3, 2)
    for i in range(5):
        let out = x @ w
        let s = sum(out)
        println("step {}: sum = {}", i, s)
        if i > 2:
            println("  past halfway")
    println("done")
