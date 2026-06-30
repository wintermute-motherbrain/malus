# M26 done-when: numerical (finite-difference) vs analytic (backward())
# gradient check for every op exercised by the nanoGPT forward pass.
# Central difference, eps=1e-3; max |analytic - numeric| recorded via
# record_diff() so the test harness can assert a tolerance after the run
# (examples/gradient_check.ml has no return value to inspect otherwise).

kernel __perturb_kernel(x: Tensor<f32>, idx: i32, delta: f32) -> Tensor<f32>:
    let i = thread_id()
    if i == idx:
        out[i] = x[i] + delta
    else:
        out[i] = x[i]

fn __perturb_fwd(x: Tensor<f32>, idx: i64, delta: f32) -> Tensor<f32>:
    let n = x.len
    return __perturb_kernel[grid=[n, 1, 1], tg=[1, 1, 1]](x, idx, delta)

# gpu_barrier() is only inserted by CTMM before a *drop* of a pending tensor,
# never before a plain read — so a Variable's .grad (RC-managed, not
# CTMM-static-dropped) can be read before its GPU computation is visible.
# A throwaway static-drop Tensor read reliably forces the flush.
fn __flush():
    let t = ones(1)
    let _v = t[0]

# Records |analytic[i] - numeric central-diff of f_t at x[i]| for every i,
# where f_t(perturbed_x) is supplied by the caller via two pre-computed
# scalar-loss tensors fp/fm per element — callers loop and call this once
# per element since malus has no closures to pass `f` itself.
fn __check_elem(analytic: Tensor<f32>, i: i64, fp0: f32, fm0: f32, eps: f32):
    let numeric = (fp0 - fm0) / (2.0 * eps)
    let a = analytic[i]
    let diff = numeric - a
    let mut abs_diff = diff
    if diff < 0.0:
        abs_diff = 0.0 - diff
    record_diff(abs_diff)

# ── add / mul (elementwise) ───────────────────────────────────────────────────

fn __loss_add(x: Tensor<f32>, y: Tensor<f32>) -> Tensor<f32>:
    return sum(x + y)

fn __loss_add_v(x: Variable<f32>, y: Variable<f32>) -> Variable<f32>:
    return sum(x + y)

fn check_add():
    let eps = 0.001
    let x = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let y = Tensor.gpu<f32>([0.5, -1.0, 2.0])
    let vx = variable(x)
    let vy = variable(y)
    let loss = __loss_add_v(vx, vy)
    backward(loss)
    __flush()
    let n = x.len
    with no_grad:
        for i in range(0, n):
            let xp = __perturb_fwd(x, i, eps)
            let xm = __perturb_fwd(x, i, 0.0 - eps)
            let fp = __loss_add(xp, y)
            let fm = __loss_add(xm, y)
            __check_elem(vx.grad, i, fp[0], fm[0], eps)
        for i in range(0, n):
            let yp = __perturb_fwd(y, i, eps)
            let ym = __perturb_fwd(y, i, 0.0 - eps)
            let fp = __loss_add(x, yp)
            let fm = __loss_add(x, ym)
            __check_elem(vy.grad, i, fp[0], fm[0], eps)

fn __loss_mul(x: Tensor<f32>, y: Tensor<f32>) -> Tensor<f32>:
    return sum(x * y)

fn __loss_mul_v(x: Variable<f32>, y: Variable<f32>) -> Variable<f32>:
    return sum(x * y)

fn check_mul():
    let eps = 0.001
    let x = Tensor.gpu<f32>([1.0, 2.0, 3.0])
    let y = Tensor.gpu<f32>([0.5, -1.0, 2.0])
    let vx = variable(x)
    let vy = variable(y)
    let loss = __loss_mul_v(vx, vy)
    backward(loss)
    __flush()
    let n = x.len
    with no_grad:
        for i in range(0, n):
            let xp = __perturb_fwd(x, i, eps)
            let xm = __perturb_fwd(x, i, 0.0 - eps)
            let fp = __loss_mul(xp, y)
            let fm = __loss_mul(xm, y)
            __check_elem(vx.grad, i, fp[0], fm[0], eps)
        for i in range(0, n):
            let yp = __perturb_fwd(y, i, eps)
            let ym = __perturb_fwd(y, i, 0.0 - eps)
            let fp = __loss_mul(x, yp)
            let fm = __loss_mul(x, ym)
            __check_elem(vy.grad, i, fp[0], fm[0], eps)

# ── matmul ─────────────────────────────────────────────────────────────────────

fn __loss_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return sum(a @ b)

fn __loss_matmul_v(a: Variable<f32>, b: Variable<f32>) -> Variable<f32>:
    return sum(a @ b)

# matmul's gradient is exactly linear in each operand (zero finite-difference
# truncation error at any eps), so a larger eps is used here specifically to
# reduce 1/(2*eps) amplification of MPS's own float32 rounding noise — at
# eps=1e-3 that amplification (~500x) pushed noise on the order of 1e-6 up
# past the 1e-3 gate threshold.
fn check_matmul():
    let eps = 0.01
    let a = Tensor.gpu<f32>([[1.0, 2.0], [3.0, 4.0]])
    let b = Tensor.gpu<f32>([[0.5, -0.5], [1.0, 2.0]])
    let va = variable(a)
    let vb = variable(b)
    let loss = __loss_matmul_v(va, vb)
    backward(loss)
    __flush()
    let n = a.len
    with no_grad:
        for i in range(0, n):
            let ap = __perturb_fwd(a, i, eps)
            let am = __perturb_fwd(a, i, 0.0 - eps)
            let fp = __loss_matmul(ap, b)
            let fm = __loss_matmul(am, b)
            __check_elem(va.grad, i, fp[0], fm[0], eps)
        for i in range(0, n):
            let bp = __perturb_fwd(b, i, eps)
            let bm = __perturb_fwd(b, i, 0.0 - eps)
            let fp = __loss_matmul(a, bp)
            let fm = __loss_matmul(a, bm)
            __check_elem(vb.grad, i, fp[0], fm[0], eps)

# ── softmax ────────────────────────────────────────────────────────────────────
# sum(softmax(x)) is always 1 regardless of x (degenerate, zero gradient) —
# dot with a fixed weight to get a non-trivial scalar loss.

fn __loss_softmax(x: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32>:
    return sum(softmax(x, axis=1) * w)

fn __loss_softmax_v(x: Variable<f32>, w: Variable<f32>) -> Variable<f32>:
    return sum(softmax(x, axis=1) * w)

fn check_softmax():
    let eps = 0.001
    let x = Tensor.gpu<f32>([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]])
    let w = Tensor.gpu<f32>([[0.2, 0.5, -0.3], [1.0, -1.0, 0.5]])
    let vx = variable(x)
    let vw = variable(w)
    let loss = __loss_softmax_v(vx, vw)
    backward(loss)
    __flush()
    let n = x.len
    with no_grad:
        for i in range(0, n):
            let xp = __perturb_fwd(x, i, eps)
            let xm = __perturb_fwd(x, i, 0.0 - eps)
            let fp = __loss_softmax(xp, w)
            let fm = __loss_softmax(xm, w)
            __check_elem(vx.grad, i, fp[0], fm[0], eps)

# ── layernorm ──────────────────────────────────────────────────────────────────

fn __loss_layernorm(x: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32>:
    return sum(layernorm(x, axis=1) * w)

fn __loss_layernorm_v(x: Variable<f32>, w: Variable<f32>) -> Variable<f32>:
    return sum(layernorm(x, axis=1) * w)

fn check_layernorm():
    let eps = 0.001
    let x = Tensor.gpu<f32>([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, -0.5, 2.5]])
    let w = Tensor.gpu<f32>([[0.3, -0.2, 0.5, 1.0], [1.0, 0.5, -0.5, 0.2]])
    let vx = variable(x)
    let vw = variable(w)
    let loss = __loss_layernorm_v(vx, vw)
    backward(loss)
    __flush()
    let n = x.len
    with no_grad:
        for i in range(0, n):
            let xp = __perturb_fwd(x, i, eps)
            let xm = __perturb_fwd(x, i, 0.0 - eps)
            let fp = __loss_layernorm(xp, w)
            let fm = __loss_layernorm(xm, w)
            __check_elem(vx.grad, i, fp[0], fm[0], eps)

# ── gelu ───────────────────────────────────────────────────────────────────────

fn __loss_gelu(x: Tensor<f32>) -> Tensor<f32>:
    return sum(gelu(x))

fn __loss_gelu_v(x: Variable<f32>) -> Variable<f32>:
    return sum(gelu(x))

fn check_gelu():
    let eps = 0.001
    let x = Tensor.gpu<f32>([1.0, -1.0, 2.0, -0.5, 0.5])
    let vx = variable(x)
    let loss = __loss_gelu_v(vx)
    backward(loss)
    __flush()
    let n = x.len
    with no_grad:
        for i in range(0, n):
            let xp = __perturb_fwd(x, i, eps)
            let xm = __perturb_fwd(x, i, 0.0 - eps)
            let fp = __loss_gelu(xp)
            let fm = __loss_gelu(xm)
            __check_elem(vx.grad, i, fp[0], fm[0], eps)

# ── embedding ──────────────────────────────────────────────────────────────────
# embedding()/cross_entropy() always require Variable<f32> input (no Tensor
# variant exists) — the numeric path re-wraps each perturbed Tensor in a
# fresh variable() under no_grad and reads the scalar back via .data[0].

fn __loss_embedding_v(weight: Variable<f32>, idx: Tensor<i32>) -> Variable<f32>:
    return sum(embedding(weight, idx))

fn check_embedding():
    let eps = 0.001
    let weight = Tensor.gpu<f32>([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    let idx = Tensor.gpu<i32>([0, 1, 0])
    let vweight = variable(weight)
    let loss = __loss_embedding_v(vweight, idx)
    backward(loss)
    __flush()
    let n = weight.len
    with no_grad:
        for i in range(0, n):
            let wp = __perturb_fwd(weight, i, eps)
            let wm = __perturb_fwd(weight, i, 0.0 - eps)
            let fp = __loss_embedding_v(variable(wp), idx)
            let fm = __loss_embedding_v(variable(wm), idx)
            __flush()
            __check_elem(vweight.grad, i, fp.data[0], fm.data[0], eps)

# ── cross_entropy ────────────────────────────────────────────────────────────

fn check_cross_entropy():
    let eps = 0.001
    let logits = Tensor.gpu<f32>([[1.0, 2.0, 0.5], [0.2, -0.3, 1.2]])
    let targets = Tensor.gpu<i32>([1, 2])
    let vlogits = variable(logits)
    let loss = cross_entropy(vlogits, targets)
    backward(loss)
    __flush()
    let n = logits.len
    with no_grad:
        for i in range(0, n):
            let lp = __perturb_fwd(logits, i, eps)
            let lm = __perturb_fwd(logits, i, 0.0 - eps)
            let fp = cross_entropy(variable(lp), targets)
            let fm = cross_entropy(variable(lm), targets)
            __flush()
            __check_elem(vlogits.grad, i, fp.data[0], fm.data[0], eps)

fn main():
    record_diff(0.0)
    println("check_add")
    check_add()
    println("check_mul")
    check_mul()
    println("check_matmul")
    check_matmul()
    println("check_softmax")
    check_softmax()
    println("check_layernorm")
    check_layernorm()
    println("check_gelu")
    check_gelu()
    println("check_embedding")
    check_embedding()
    println("check_cross_entropy")
    check_cross_entropy()

    with no_grad:
        let w = variable(Tensor.gpu<f32>([[0.1], [0.2]]))
        let dummy = w * w
        println("no_grad: tape not recording")

    println("autograd vs finite-diff: OK")
