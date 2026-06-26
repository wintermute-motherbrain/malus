# malus MVP showcase
#
# Demonstrates every capability of the v0.1 compiler:
#   - fn-body tensor BinOps dispatching built-in GPU kernels (M5.1)
#   - User-defined kernels compiled to MSL (M5)
#   - Chained BinOps (multiple kernel dispatches in one expression)
#   - Mixed user kernels + built-in BinOps in the same program
#   - fn-to-fn calls with tensor returns
#   - Scalar arithmetic on the CPU (Cranelift JIT)
#   - Format-string printing with tensor and scalar arguments
#   - CTMM automatic memory management (no manual free/barrier calls)


# ── User-defined kernels (compiled to Metal Shading Language) ─────────────────

kernel fma(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) -> Tensor<f32>:
    return a + b * c

kernel neg(a: Tensor<f32>) -> Tensor<f32>:
    return -a


# ── Host functions (JIT-compiled via Cranelift) ───────────────────────────────

fn pairwise(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn net(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) -> Tensor<f32>:
    let fused = fma(a, b, c)
    return fused + a

fn average(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b

fn scalars(x: f32, y: f32) -> f32:
    return (x + y) * 2.0

fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([4.0, 3.0, 2.0, 1.0])
    let c = Tensor.gpu<f32>([2.0, 2.0, 2.0, 2.0])

    println("=== inputs ===")
    println("a: {}", a)
    println("b: {}", b)
    println("c: {}", c)

    println("")
    println("=== built-in element-wise ops (M5.1) ===")
    println("a + b: {}", a + b)
    println("a - b: {}", a - b)
    println("a * b: {}", a * b)
    println("a / b: {}", a / b)

    println("")
    println("=== chained BinOps (two dispatches) ===")
    println("a + b * c: {}", a + b * c)

    println("")
    println("=== user-defined kernels (M5) ===")
    println("fma(a, b, c): {}", fma(a, b, c))
    println("neg(a):       {}", neg(a))

    println("")
    println("=== fn-to-fn calls ===")
    println("pairwise(a, b): {}", pairwise(a, b))

    println("")
    println("=== mixed user kernel + built-in BinOp ===")
    println("fma(a,b,c) + a: {}", net(a, b, c))

    println("")
    println("=== scalar arithmetic (CPU / Cranelift) ===")
    println("(1.0 + 2.0) * 2.0 = {}", scalars(1.0, 2.0))
