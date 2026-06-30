# GELU elementwise kernel using the tanh approximation.
# gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# One thread per element; pure elementwise, no shared memory.
#
# Grid: [N, 1, 1] threads (one per element).

kernel gelu(input: Tensor<f32>) -> Tensor<f32>:
    let i = thread_id()
    let x = input[i]
    let x3 = x * x * x
    let inner = 0.7978845608028654 * (x + 0.044715 * x3)
    out[i] = 0.5 * x * (1.0 + tanh(inner))
