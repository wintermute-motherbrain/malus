# M34 done-when #5: nanoGPT written the way a PyTorch user would write it —
# named submodules (GPT { blocks: List<Block> }), method-form forward, one
# generic optimizer applied per submodule (ADR-0036 optimizer recursion).
# Smoke scale: 2 blocks, C=32, H=4 head-folded attention (the M33 MHA shape).
# The benchmark harness stays examples/nanogpt.ml (flat, single-block) until
# M35 rewrites the capstone at the Karpathy config.

struct AdamW:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32
    wd: f32

struct Block:
    params: List<Tensor<f32>>

struct GPT:
    blocks: List<Block>
    params: List<Tensor<f32>>

# Per-submodule AdamW moment state, mirroring that submodule's identity list.
struct Moments:
    ms: List<Tensor<f32>>
    vs: List<Tensor<f32>>

trait Module:
    fn parameters(self) -> List<Tensor<f32>>

impl Module for Block:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

    # Inherent method (M34): lives beside the trait method without widening
    # the Module trait's contract. Block-LOCAL index constants — the
    # cross-layer `params[l*12 + WQ]` arithmetic M34 exists to kill never
    # appears; each block indexes only its own 8-tensor identity list.
    fn forward(self, x: Tensor<f32>, B: i64, T: i64, C: i64) -> Tensor<f32>:
        let LN1_W = 0
        let WQ = 1
        let WK = 2
        let WV = 3
        let WO = 4
        let LN2_W = 5
        let W1 = 6
        let W2 = 7
        let xn1 = layernorm(x, axis=1) * self.params[LN1_W]
        # M33 head-folded multi-head attention (ADR-0029 composed form).
        let H = 4
        let hs = C / H
        let Q = reshape(permute(reshape(xn1 @ self.params[WQ], B, T, H, hs), 0, 2, 1, 3), B * H, T, hs)
        let K = reshape(permute(reshape(xn1 @ self.params[WK], B, T, H, hs), 0, 2, 1, 3), B * H, T, hs)
        let V = reshape(permute(reshape(xn1 @ self.params[WV], B, T, H, hs), 0, 2, 1, 3), B * H, T, hs)
        let Kt = permute(K, 0, 2, 1)
        # 1/sqrt(hs) = 1/sqrt(8)
        let scale = variable(ones(1, 1) * 0.35355)
        let scores = (Q @ Kt) * scale
        let mask = causal_mask(T)
        let vmask = variable(mask)
        let masked = scores + vmask
        let attn = softmax(masked, axis=2)
        let att_out = reshape(permute(reshape(attn @ V, B, H, T, hs), 0, 2, 1, 3), B * T, C)
        let proj = att_out @ self.params[WO]
        let x2 = x + proj
        let xn2 = layernorm(x2, axis=1) * self.params[LN2_W]
        let hidden = gelu(xn2 @ self.params[W1])
        let mlp_out = hidden @ self.params[W2]
        return x2 + mlp_out

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

    # GPT's own identity list holds only the non-block tensors.
    fn forward(self, toks: Tensor<i32>, B: i64, T: i64, C: i64) -> Tensor<f32>:
        let WTE = 0
        let WPE = 1
        let LN_F = 2
        let LM_HEAD = 3
        let mut pos_buf = buffer_i32(B * T)
        for b in range(B):
            for t in range(T):
                pos_buf[b * T + t] = t
        let pos_toks = freeze(pos_buf)
        let te = embedding(self.params[WTE], toks)
        let pe = embedding(self.params[WPE], pos_toks)
        let mut x = te + pe
        for blk in self.blocks:
            x = blk.forward(x, B, T, C)
        let xf = layernorm(x, axis=1) * self.params[LN_F]
        return xf @ self.params[LM_HEAD]

# The single generic optimizer, unchanged from the flat form (M28 done-when).
# Composition happens at the CALL SITE by recursion — once per Block, once for
# GPT's own tensors — so each submodule's identity list receives the slot
# writes (ADR-0036: parameters() is never concatenated).
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

fn make_block(C: i64, C4: i64) -> Block:
    let init_scale = ones(1, 1) * 0.02
    return Block(params=[
        variable(ones(C)),                         # ln1_w
        variable(randn(C, C) * init_scale),        # wq
        variable(randn(C, C) * init_scale),        # wk
        variable(randn(C, C) * init_scale),        # wv
        variable(randn(C, C) * init_scale),        # wo
        variable(ones(C)),                         # ln2_w
        variable(randn(C, C4) * init_scale),       # w1
        variable(randn(C4, C) * init_scale),       # w2
    ])

fn block_moments(C: i64, C4: i64) -> Moments:
    return Moments(
        ms=[zeros(C), zeros(C, C), zeros(C, C), zeros(C, C), zeros(C, C),
            zeros(C), zeros(C, C4), zeros(C4, C)],
        vs=[zeros(C), zeros(C, C), zeros(C, C), zeros(C, C), zeros(C, C),
            zeros(C), zeros(C, C4), zeros(C4, C)])

fn make_moments(C: i64, C4: i64) -> List<Moments>:
    return [block_moments(C, C4), block_moments(C, C4)]

fn top_moments(C: i64, V: i64, T: i64) -> Moments:
    return Moments(
        ms=[zeros(V, C), zeros(T, C), zeros(C), zeros(C, V)],
        vs=[zeros(V, C), zeros(T, C), zeros(C), zeros(C, V)])

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
    let gpt = GPT(
        blocks=[make_block(C, C4), make_block(C, C4)],
        params=[
            variable(randn(V, C) * init_scale),    # wte
            variable(randn(T, C) * init_scale),    # wpe
            variable(ones(C)),                     # ln_f
            variable(randn(C, V) * init_scale),    # lm_head
        ])

    let moments = make_moments(C, C4)
    let topm = top_moments(C, V, T)
    let opt = AdamW(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)

    let mut print_at = 1
    for step in range(1, max_steps + 1):
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
        for blk in gpt.blocks:
            let bp = blk.parameters()
            zero_grad(bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6], bp[7])
        let tp = gpt.parameters()
        zero_grad(tp[0], tp[1], tp[2], tp[3])
        let logits = gpt.forward(x_toks, B, T, C)
        let loss = cross_entropy(logits, y_toks)
        backward(loss)
        for i in range(len(gpt.blocks)):
            adamw(gpt.blocks[i], moments[i].ms, moments[i].vs, opt, step)
        adamw(gpt, topm.ms, topm.vs, opt, step)
        bench_step_end()
        if step == print_at:
            println("step {}: loss = {}", step, loss.data)
            print_at = print_at + 20
