fn main():
    # Random init (Philox4x32-10, fixed seed)
    let w_emb = variable(randn(256, 16))
    let w_pos = variable(randn(8, 16))

    # Token indices (integer tensor)
    let tokens = Tensor.cpu<i32>([3, 1, 4, 1, 5, 9, 2, 6])

    # Embedding lookup with gradient
    let tok_emb = embedding(w_emb, tokens)
    let pos_emb = embedding(w_pos, Tensor.cpu<i32>([0, 1, 2, 3, 4, 5, 6, 7]))
    let x = tok_emb + pos_emb
    let loss = sum(x)
    backward(loss)
    print(w_emb.grad)
    println("embeddings: OK")
