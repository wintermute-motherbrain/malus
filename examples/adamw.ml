struct AdamW:
    lr: f32
    beta1: f32
    beta2: f32
    eps: f32
    wd: f32

fn adamw_step(opt: AdamW, mut params: Array<Variable<f32>, 4>,
              mut ms: Array<Tensor<f32>, 4>, mut vs: Array<Tensor<f32>, 4>,
              t: i64):
    let bc1 = 1.0 - opt.beta1 ** t
    let bc2 = 1.0 - opt.beta2 ** t
    for i in range(4):
        let g = params[i].grad + opt.wd * params[i].data
        ms[i] = opt.beta1 * ms[i] + (1.0 - opt.beta1) * g
        vs[i] = opt.beta2 * vs[i] + (1.0 - opt.beta2) * g * g
        let m_hat = ms[i] / bc1
        let v_hat = vs[i] / bc2
        params[i] = variable(params[i].data - opt.lr * m_hat / (sqrt(v_hat) + opt.eps))

fn main():
    let x      = ones(8, 4)
    let target = ones(8, 1)

    let mut params = [variable(randn(4, 1)), variable(zeros(1, 1)),
                      variable(zeros(1, 1)), variable(zeros(1, 1))]
    let mut ms = [zeros(4, 1), zeros(1, 1), zeros(1, 1), zeros(1, 1)]
    let mut vs = [zeros(4, 1), zeros(1, 1), zeros(1, 1), zeros(1, 1)]

    let opt = AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)

    for t in range(1, 201):
        let pred = variable(x) @ params[0] + variable(ones(8, 1)) @ params[1]
        let loss = sum((pred - variable(target)) * (pred - variable(target)))
        zero_grad(params[0], params[1])
        backward(loss)
        adamw_step(opt, params, ms, vs, t)
        if t == 1 or t == 200:
            println("step {}: loss = {}", t, loss.data)
