# Demonstrates both import styles.
#
# Qualified import: ops.add(...) — the module name is used as a prefix.
import ops

# Selective import: scale(...) — brought into scope without a prefix.
from ops import scale

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let b = Tensor.gpu<f32>([4.0, 5.0, 6.0])

    let c = ops.add(a, b)
    let d = scale(c, 2.0)
    return d
