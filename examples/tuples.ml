fn divmod(a: f32, b: f32) -> (f32, f32):
    return (a / b, a - b * (a / b))

fn swap(x: f32, y: f32) -> (f32, f32):
    return (y, x)

fn main():
    let t = (25.0, 50.0)
    let x = t.0
    let y = t.1
    println("x={}, y={}", x, y)

    let (a, b) = swap(1.0, 2.0)
    println("swap: a={}, b={}", a, b)

    let (q, r) = divmod(17.0, 5.0)
    println("17 / 5 = {} rem {}", q, r)
