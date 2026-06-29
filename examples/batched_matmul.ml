fn main():
    # Q: reshape flat (8,8) -> (2,4,8) — simulates (batch, heads, head_dim)
    let q_flat = ones(8, 8)
    let q_var = variable(q_flat)
    let q = reshape(q_var, 2, 4, 8)

    # K: reshape flat (8,8) -> (2,4,8), then permute last two dims -> (2,8,4) = K^T
    let k_flat = ones(8, 8)
    let k_var = variable(k_flat)
    let k_3d = reshape(k_var, 2, 4, 8)
    let k = permute(k_3d, 0, 2, 1)

    # batched matmul: (2,4,8) @ (2,8,4) -> (2,4,4)  (scores = Q @ K^T)
    let scores = q @ k

    let loss = sum(scores)
    backward(loss)

    println(q_var.grad)
    println(k_var.grad)
    println("shapes + batched matmul: OK")
