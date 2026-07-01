struct AdamW:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32
    wd: f32

struct Block:
    ln1_w: Tensor<f32>
    wq: Tensor<f32>
    wk: Tensor<f32>
    wv: Tensor<f32>
    wo: Tensor<f32>
    ln2_w: Tensor<f32>
    w1: Tensor<f32>
    w2: Tensor<f32>

struct GPT:
    wte: Tensor<f32>
    wpe: Tensor<f32>
    ln_f: Tensor<f32>
    lm_head: Tensor<f32>

struct BlockState:
    ms_ln1: Tensor<f32>
    vs_ln1: Tensor<f32>
    ms_wq: Tensor<f32>
    vs_wq: Tensor<f32>
    ms_wk: Tensor<f32>
    vs_wk: Tensor<f32>
    ms_wv: Tensor<f32>
    vs_wv: Tensor<f32>
    ms_wo: Tensor<f32>
    vs_wo: Tensor<f32>
    ms_ln2: Tensor<f32>
    vs_ln2: Tensor<f32>
    ms_w1: Tensor<f32>
    vs_w1: Tensor<f32>
    ms_w2: Tensor<f32>
    vs_w2: Tensor<f32>

struct GPTState:
    ms_wte: Tensor<f32>
    vs_wte: Tensor<f32>
    ms_wpe: Tensor<f32>
    vs_wpe: Tensor<f32>
    ms_ln_f: Tensor<f32>
    vs_ln_f: Tensor<f32>
    ms_lm_head: Tensor<f32>
    vs_lm_head: Tensor<f32>

fn forward(gpt: GPT, blk: Block, toks: Tensor<i32>, B: i64, T: i64, C: i64) -> Tensor<f32>:
    let mut pos_buf = buffer_i32(B * T)
    for b in range(B):
        for t in range(T):
            pos_buf[b * T + t] = t
    let pos_toks = freeze(pos_buf)
    let te = embedding(gpt.wte, toks)
    let pe = embedding(gpt.wpe, pos_toks)
    let x = te + pe
    let xn1 = layernorm(x, axis=1) * blk.ln1_w
    let Q = reshape(xn1 @ blk.wq, B, T, C)
    let K = reshape(xn1 @ blk.wk, B, T, C)
    let V = reshape(xn1 @ blk.wv, B, T, C)
    let Kt = permute(K, 0, 2, 1)
    let scale = variable(ones(1, 1) * 0.17678)
    let scores = (Q @ Kt) * scale
    let mask = causal_mask(T)
    let vmask = variable(mask)
    let masked = scores + vmask
    let attn = softmax(masked, axis=2)
    let att_out = reshape(attn @ V, B * T, C)
    let proj = att_out @ blk.wo
    let x2 = x + proj
    let xn2 = layernorm(x2, axis=1) * blk.ln2_w
    let hidden = gelu(xn2 @ blk.w1)
    let mlp_out = hidden @ blk.w2
    let x3 = x2 + mlp_out
    let xf = layernorm(x3, axis=1) * gpt.ln_f
    return xf @ gpt.lm_head

fn adamw_block(opt: AdamW, mut blk: Block, mut st: BlockState, t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    let g_ln1 = blk.ln1_w.grad + opt.wd * blk.ln1_w.data
    st.ms_ln1 = opt.beta1 * st.ms_ln1 + (1.0 - opt.beta1) * g_ln1
    st.vs_ln1 = opt.beta2 * st.vs_ln1 + (1.0 - opt.beta2) * g_ln1 * g_ln1
    let m_hat_ln1 = st.ms_ln1 / bc1
    let v_hat_ln1 = st.vs_ln1 / bc2
    blk.ln1_w = variable(blk.ln1_w.data - opt.lr * m_hat_ln1 / (sqrt(v_hat_ln1) + opt.eps))
    let g_wq = blk.wq.grad + opt.wd * blk.wq.data
    st.ms_wq = opt.beta1 * st.ms_wq + (1.0 - opt.beta1) * g_wq
    st.vs_wq = opt.beta2 * st.vs_wq + (1.0 - opt.beta2) * g_wq * g_wq
    let m_hat_wq = st.ms_wq / bc1
    let v_hat_wq = st.vs_wq / bc2
    blk.wq = variable(blk.wq.data - opt.lr * m_hat_wq / (sqrt(v_hat_wq) + opt.eps))
    let g_wk = blk.wk.grad + opt.wd * blk.wk.data
    st.ms_wk = opt.beta1 * st.ms_wk + (1.0 - opt.beta1) * g_wk
    st.vs_wk = opt.beta2 * st.vs_wk + (1.0 - opt.beta2) * g_wk * g_wk
    let m_hat_wk = st.ms_wk / bc1
    let v_hat_wk = st.vs_wk / bc2
    blk.wk = variable(blk.wk.data - opt.lr * m_hat_wk / (sqrt(v_hat_wk) + opt.eps))
    let g_wv = blk.wv.grad + opt.wd * blk.wv.data
    st.ms_wv = opt.beta1 * st.ms_wv + (1.0 - opt.beta1) * g_wv
    st.vs_wv = opt.beta2 * st.vs_wv + (1.0 - opt.beta2) * g_wv * g_wv
    let m_hat_wv = st.ms_wv / bc1
    let v_hat_wv = st.vs_wv / bc2
    blk.wv = variable(blk.wv.data - opt.lr * m_hat_wv / (sqrt(v_hat_wv) + opt.eps))
    let g_wo = blk.wo.grad + opt.wd * blk.wo.data
    st.ms_wo = opt.beta1 * st.ms_wo + (1.0 - opt.beta1) * g_wo
    st.vs_wo = opt.beta2 * st.vs_wo + (1.0 - opt.beta2) * g_wo * g_wo
    let m_hat_wo = st.ms_wo / bc1
    let v_hat_wo = st.vs_wo / bc2
    blk.wo = variable(blk.wo.data - opt.lr * m_hat_wo / (sqrt(v_hat_wo) + opt.eps))
    let g_ln2 = blk.ln2_w.grad + opt.wd * blk.ln2_w.data
    st.ms_ln2 = opt.beta1 * st.ms_ln2 + (1.0 - opt.beta1) * g_ln2
    st.vs_ln2 = opt.beta2 * st.vs_ln2 + (1.0 - opt.beta2) * g_ln2 * g_ln2
    let m_hat_ln2 = st.ms_ln2 / bc1
    let v_hat_ln2 = st.vs_ln2 / bc2
    blk.ln2_w = variable(blk.ln2_w.data - opt.lr * m_hat_ln2 / (sqrt(v_hat_ln2) + opt.eps))
    let g_w1 = blk.w1.grad + opt.wd * blk.w1.data
    st.ms_w1 = opt.beta1 * st.ms_w1 + (1.0 - opt.beta1) * g_w1
    st.vs_w1 = opt.beta2 * st.vs_w1 + (1.0 - opt.beta2) * g_w1 * g_w1
    let m_hat_w1 = st.ms_w1 / bc1
    let v_hat_w1 = st.vs_w1 / bc2
    blk.w1 = variable(blk.w1.data - opt.lr * m_hat_w1 / (sqrt(v_hat_w1) + opt.eps))
    let g_w2 = blk.w2.grad + opt.wd * blk.w2.data
    st.ms_w2 = opt.beta1 * st.ms_w2 + (1.0 - opt.beta1) * g_w2
    st.vs_w2 = opt.beta2 * st.vs_w2 + (1.0 - opt.beta2) * g_w2 * g_w2
    let m_hat_w2 = st.ms_w2 / bc1
    let v_hat_w2 = st.vs_w2 / bc2
    blk.w2 = variable(blk.w2.data - opt.lr * m_hat_w2 / (sqrt(v_hat_w2) + opt.eps))

fn adamw_gpt_params(opt: AdamW, mut gpt: GPT, mut st: GPTState, t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    let g_wte = gpt.wte.grad + opt.wd * gpt.wte.data
    st.ms_wte = opt.beta1 * st.ms_wte + (1.0 - opt.beta1) * g_wte
    st.vs_wte = opt.beta2 * st.vs_wte + (1.0 - opt.beta2) * g_wte * g_wte
    let m_hat_wte = st.ms_wte / bc1
    let v_hat_wte = st.vs_wte / bc2
    gpt.wte = variable(gpt.wte.data - opt.lr * m_hat_wte / (sqrt(v_hat_wte) + opt.eps))
    let g_wpe = gpt.wpe.grad + opt.wd * gpt.wpe.data
    st.ms_wpe = opt.beta1 * st.ms_wpe + (1.0 - opt.beta1) * g_wpe
    st.vs_wpe = opt.beta2 * st.vs_wpe + (1.0 - opt.beta2) * g_wpe * g_wpe
    let m_hat_wpe = st.ms_wpe / bc1
    let v_hat_wpe = st.vs_wpe / bc2
    gpt.wpe = variable(gpt.wpe.data - opt.lr * m_hat_wpe / (sqrt(v_hat_wpe) + opt.eps))
    let g_ln_f = gpt.ln_f.grad + opt.wd * gpt.ln_f.data
    st.ms_ln_f = opt.beta1 * st.ms_ln_f + (1.0 - opt.beta1) * g_ln_f
    st.vs_ln_f = opt.beta2 * st.vs_ln_f + (1.0 - opt.beta2) * g_ln_f * g_ln_f
    let m_hat_ln_f = st.ms_ln_f / bc1
    let v_hat_ln_f = st.vs_ln_f / bc2
    gpt.ln_f = variable(gpt.ln_f.data - opt.lr * m_hat_ln_f / (sqrt(v_hat_ln_f) + opt.eps))
    let g_lm = gpt.lm_head.grad + opt.wd * gpt.lm_head.data
    st.ms_lm_head = opt.beta1 * st.ms_lm_head + (1.0 - opt.beta1) * g_lm
    st.vs_lm_head = opt.beta2 * st.vs_lm_head + (1.0 - opt.beta2) * g_lm * g_lm
    let m_hat_lm = st.ms_lm_head / bc1
    let v_hat_lm = st.vs_lm_head / bc2
    gpt.lm_head = variable(gpt.lm_head.data - opt.lr * m_hat_lm / (sqrt(v_hat_lm) + opt.eps))

fn generate(gpt: GPT, blk: Block, seed: Buffer<i32>, n_gen: i64, vocab: i64, T: i64, C: i64):
    let mut ctx = buffer_i32(T)
    for i in range(T):
        ctx[i] = seed[i]
    let mut step = 0
    let mut next_tok = 0
    while step < n_gen:
        with no_grad:
            let mut copy = buffer_i32(T)
            for i in range(T):
                copy[i] = ctx[i]
            let ctx_toks = freeze(copy)
            let logits = forward(gpt, blk, ctx_toks, 1, T, C)
            let probs = softmax(logits, axis=1)
            let offset = (T - 1) * vocab
            let thresh = rand_uniform()
            let mut cum = 0.0
            let mut j = 0
            while j < vocab:
                let p = probs.data[offset + j]
                cum = cum + p
                if cum > thresh:
                    next_tok = j
                    break
                j = j + 1
        print(str_from_char(next_tok))
        for i in range(T - 1):
            ctx[i] = ctx[i + 1]
        ctx[T - 1] = next_tok
        step = step + 1
    println("")

fn main():
    let C = 32
    let T = 16
    let B = 4
    let V = 128
    let C4 = 128
    let max_steps = 300

    let text = read_file("data/tiny_shakespeare.txt")
    let data_len = str_len(text)
    let mut data_buf = buffer_i32(data_len)
    for i in range(data_len):
        data_buf[i] = str_char_at(text, i)

    let init_scale = ones(1, 1) * 0.02
    let ln1_w = variable(ones(C))
    let wq = variable(randn(C, C) * init_scale)
    let wk = variable(randn(C, C) * init_scale)
    let wv = variable(randn(C, C) * init_scale)
    let wo = variable(randn(C, C) * init_scale)
    let ln2_w = variable(ones(C))
    let w1 = variable(randn(C, C4) * init_scale)
    let w2 = variable(randn(C4, C) * init_scale)
    let blk = Block(
        ln1_w=ln1_w,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        ln2_w=ln2_w,
        w1=w1,
        w2=w2,
    )
    let wte = variable(randn(V, C) * init_scale)
    let wpe = variable(randn(T, C) * init_scale)
    let ln_f = variable(ones(C))
    let lm_head = variable(randn(C, V) * init_scale)
    let gpt = GPT(wte=wte, wpe=wpe, ln_f=ln_f, lm_head=lm_head)

    let blk_st = BlockState(
        ms_ln1=zeros(C),    vs_ln1=zeros(C),
        ms_wq=zeros(C, C),  vs_wq=zeros(C, C),
        ms_wk=zeros(C, C),  vs_wk=zeros(C, C),
        ms_wv=zeros(C, C),  vs_wv=zeros(C, C),
        ms_wo=zeros(C, C),  vs_wo=zeros(C, C),
        ms_ln2=zeros(C),    vs_ln2=zeros(C),
        ms_w1=zeros(C, C4), vs_w1=zeros(C, C4),
        ms_w2=zeros(C4, C), vs_w2=zeros(C4, C),
    )
    let gpt_st = GPTState(
        ms_wte=zeros(V, C),  vs_wte=zeros(V, C),
        ms_wpe=zeros(T, C),  vs_wpe=zeros(T, C),
        ms_ln_f=zeros(C),    vs_ln_f=zeros(C),
        ms_lm_head=zeros(C, V), vs_lm_head=zeros(C, V),
    )
    let opt = AdamW(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)

    let mut print_at = 1
    for step in range(1, max_steps + 1):
        let mut x_buf = buffer_i32(B * T)
        let mut y_buf = buffer_i32(B * T)
        for b in range(B):
            let start = rand_int(data_len - T - 1)
            for t in range(T):
                x_buf[b * T + t] = data_buf[start + t]
                y_buf[b * T + t] = data_buf[start + t + 1]
        let x_toks = freeze(x_buf)
        let y_toks = freeze(y_buf)
        zero_grad(gpt.wte, gpt.wpe, gpt.ln_f, gpt.lm_head,
                  blk.ln1_w, blk.wq, blk.wk, blk.wv,
                  blk.wo, blk.ln2_w, blk.w1, blk.w2)
        let logits = forward(gpt, blk, x_toks, B, T, C)
        let loss = cross_entropy(logits, y_toks)
        backward(loss)
        adamw_block(opt, blk, blk_st, step)
        adamw_gpt_params(opt, gpt, gpt_st, step)
        if step == print_at:
            println("step {}: loss = {}", step, loss.data)
            print_at = print_at + 20

    println("\n--- sample ---")
    let mut seed_buf = buffer_i32(T)
    for i in range(T):
        seed_buf[i] = str_char_at(text, i)
    generate(gpt, blk, seed_buf, 200, V, T, C)
