struct AdamW:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32
    wd: f32

struct GPT:
    params: List<Tensor<f32>>

trait Module:
    fn parameters(self) -> List<Tensor<f32>>

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

# Index constants into GPT.params (single flat list, single-block model — V4/M28
# fence: named submodule nesting deferred, ADR-0034).
fn forward(model: GPT, toks: Tensor<i32>, B: i64, T: i64, C: i64) -> Tensor<f32>:
    let LN1_W = 0
    let WQ = 1
    let WK = 2
    let WV = 3
    let WO = 4
    let LN2_W = 5
    let W1 = 6
    let W2 = 7
    let WTE = 8
    let WPE = 9
    let LN_F = 10
    let LM_HEAD = 11
    # NOTE: every `model.params[IDX]` below is read inline, never bound to a
    # persistent `let` name. Binding one (`let wq = model.params[WQ]`) would
    # make CTMM treat `wq` as an owned local and RC-release it at its last use
    # inside this call — decrementing (and, on a later `forward()` call with
    # the same model, potentially freeing) the tensor `model.params` itself
    # still owns. Inline reads are consumed transiently as operands and are
    # never tracked as bindings, exactly like the pre-M28 `blk.wq` pattern.
    let mut pos_buf = buffer_i32(B * T)
    for b in range(B):
        for t in range(T):
            pos_buf[b * T + t] = t
    let pos_toks = freeze(pos_buf)
    let te = embedding(model.params[WTE], toks)
    let pe = embedding(model.params[WPE], pos_toks)
    let x = te + pe
    let xn1 = layernorm(x, axis=1) * model.params[LN1_W]
    # M33: true multi-head attention via head-folding (ADR-0029 composed form):
    # [B*T,C] → reshape [B,T,H,hs] → permute (0,2,1,3) → reshape [B*H,T,hs],
    # so the existing 3-D batched matmul treats B*H as the batch dim.
    # The zero-copy reshape after each permute is valid because permute
    # MATERIALIZES a fresh contiguous buffer (ADR-0023's trust-the-caller
    # contract) — this is the one place that invariant is load-bearing.
    let H = 4
    let hs = C / H
    let Q = reshape(permute(reshape(xn1 @ model.params[WQ], B, T, H, hs), 0, 2, 1, 3), B * H, T, hs)
    let K = reshape(permute(reshape(xn1 @ model.params[WK], B, T, H, hs), 0, 2, 1, 3), B * H, T, hs)
    let V = reshape(permute(reshape(xn1 @ model.params[WV], B, T, H, hs), 0, 2, 1, 3), B * H, T, hs)
    let Kt = permute(K, 0, 2, 1)
    # 1/sqrt(hs) = 1/sqrt(8)
    let scale = variable(ones(1, 1) * 0.35355)
    let scores = (Q @ Kt) * scale
    let mask = causal_mask(T)
    let vmask = variable(mask)
    let masked = scores + vmask
    let attn = softmax(masked, axis=2)
    # Unfold: [B*H,T,hs] → [B,H,T,hs] → permute (0,2,1,3) → [B,T,H,hs] → [B*T,C].
    let att_out = reshape(permute(reshape(attn @ V, B, H, T, hs), 0, 2, 1, 3), B * T, C)
    let proj = att_out @ model.params[WO]
    let x2 = x + proj
    let xn2 = layernorm(x2, axis=1) * model.params[LN2_W]
    let hidden = gelu(xn2 @ model.params[W1])
    let mlp_out = hidden @ model.params[W2]
    let x3 = x2 + mlp_out
    let xf = layernorm(x3, axis=1) * model.params[LN_F]
    return xf @ model.params[LM_HEAD]

# The single generic optimizer (M28 done-when #1/#2): every hand-unrolled
# per-parameter update from V3's `adamw_block`/`adamw_gpt_params` collapses
# into one loop over `model.parameters()`. `.grad`/`.data` arithmetic lives
# ONLY in this function — the no-unroll lint fails if it appears anywhere else.
fn adamw<M: Module>(model: M, mut ms: List<Tensor<f32>>, mut vs: List<Tensor<f32>>,
                     opt: AdamW, t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    let mut ps = model.parameters()
    for i in range(len(ps)):
        let g = ps[i].grad + opt.wd * ps[i].data
        ms[i] = opt.beta1 * ms[i] + (1.0 - opt.beta1) * g
        vs[i] = opt.beta2 * vs[i] + (1.0 - opt.beta2) * g * g
        let m_hat = ms[i] / bc1
        let v_hat = vs[i] / bc2
        ps[i] = variable(ps[i].data - opt.lr * m_hat / (sqrt(v_hat) + opt.eps))

# AdamW moment buffers (m, v), one zeros tensor per parameter matching its
# shape. Called once each for `ms` and `vs` in `main` — independent allocations.
fn zeros_moments(C: i64, C4: i64, V: i64, T: i64) -> List<Tensor<f32>>:
    return [zeros(C), zeros(C, C), zeros(C, C), zeros(C, C), zeros(C, C),
            zeros(C), zeros(C, C4), zeros(C4, C),
            zeros(V, C), zeros(T, C), zeros(C), zeros(C, V)]

fn generate(model: GPT, seed: Buffer<i32>, n_gen: i64, vocab: i64, T: i64, C: i64):
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
            let logits = forward(model, ctx_toks, 1, T, C)
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
    let gpt = GPT(params=[
        variable(ones(C)),                        # ln1_w
        variable(randn(C, C) * init_scale),        # wq
        variable(randn(C, C) * init_scale),        # wk
        variable(randn(C, C) * init_scale),        # wv
        variable(randn(C, C) * init_scale),        # wo
        variable(ones(C)),                         # ln2_w
        variable(randn(C, C4) * init_scale),       # w1
        variable(randn(C4, C) * init_scale),       # w2
        variable(randn(V, C) * init_scale),        # wte
        variable(randn(T, C) * init_scale),        # wpe
        variable(ones(C)),                         # ln_f
        variable(randn(C, V) * init_scale),        # lm_head
    ])

    let mut ms = zeros_moments(C, C4, V, T)
    let mut vs = zeros_moments(C, C4, V, T)
    let opt = AdamW(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)

    let mut print_at = 1
    for step in range(1, max_steps + 1):
        # M30: no-ops unless run with `malus --bench`; step_end flushes GPU
        # work inside the timed region (matches PyTorch's per-step
        # torch.mps.synchronize() — see ADR-0038).
        bench_step_begin()
        let mut x_buf = buffer_i32(B * T)
        let mut y_buf = buffer_i32(B * T)
        for b in range(B):
            let start = rand_int(data_len - T - 1)
            for t in range(T):
                x_buf[b * T + t] = data_buf[start + t]
                y_buf[b * T + t] = data_buf[start + t + 1]
        let x_toks = freeze(x_buf)
        let y_toks = freeze(y_buf)
        let ps = gpt.parameters()
        zero_grad(ps[0], ps[1], ps[2], ps[3], ps[4], ps[5],
                  ps[6], ps[7], ps[8], ps[9], ps[10], ps[11])
        let logits = forward(gpt, x_toks, B, T, C)
        let loss = cross_entropy(logits, y_toks)
        backward(loss)
        adamw(gpt, ms, vs, opt, step)
        bench_step_end()
        if step == print_at:
            println("step {}: loss = {}", step, loss.data)
            print_at = print_at + 20

    println("\n--- sample ---")
    let mut seed_buf = buffer_i32(T)
    for i in range(T):
        seed_buf[i] = str_char_at(text, i)
    generate(gpt, seed_buf, 200, V, T, C)
