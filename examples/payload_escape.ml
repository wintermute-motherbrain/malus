struct Point:
    x: f32
    y: f32

enum Container:
    WithPoint(pt: Point)
    Empty

fn make_point(v: f32) -> Point:
    return Point(x=v, y=v)

fn main():
    let c = Container.WithPoint(pt=make_point(3.14))
    let mut escaped = make_point(0.0)
    match c:
        WithPoint(pt):
            escaped = pt
        Empty:
            escaped = make_point(0.0)
    println(zeros(1))
