fn fib(a: f32) -> f32:
    if a < 2.0:
        return a
    else:
        return fib(a - 1.0) + fib(a - 2.0)

fn main():
    println("fib({}) = {}", 35.0, fib(35.0))
