fn madd(a: f32, b: f32, c: f32) -> f32:
    return (a + b) * c

fn main():
    println("({} + {}) * {} = {}", 2.0, 3.0, 4.0, madd(2.0, 3.0, 4.0))
