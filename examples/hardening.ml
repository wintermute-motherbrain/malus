enum MaybeWeight:
    Some(val: Tensor<f32>)
    Empty

fn make_weight() -> MaybeWeight:
    return MaybeWeight.Some(val=Tensor.gpu<f32>([[1.0, 1.0], [1.0, 1.0]]))

fn main():
    # break / continue: accumulate 0+1+2 + (skip 3) + 4+5+6 = 18, stop at 7
    let mut acc = 0
    for i in range(10):
        if i == 7:
            break
        if i == 3:
            continue
        acc = acc + i
    println("acc: {}", acc)

    # zero-length tensor: must not crash on allocation or dispatch
    let empty = zeros(0)
    print(empty)
    println("")

    # enum-payload escape: tensor retained across DropEnum, then freed with escaped
    let w = make_weight()
    let mut escaped = zeros(1)
    match w:
        Some(val):
            escaped = val
        Empty:
            escaped = zeros(1)
    print(escaped)
    println("")
