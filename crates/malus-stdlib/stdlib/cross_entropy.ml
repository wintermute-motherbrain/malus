# Cross-entropy loss: loss = -mean(log(softmax(logits)[i, targets[i]])).
# Returns (loss: Tensor<f32>, probs: Tensor<f32>).
#   loss  — scalar [1], the mean negative log-likelihood
#   probs — [N, C] softmax probabilities (saved by tape for VJP)
#
# Implementation:
#   probs = softmax(logits, last_axis)
#   per-token nll[i] = -log(probs[i*C + targets[i]])
#   loss = mean(nll)
#
# Uniforms: vocab: i32
# Grid: [n_tokens, 1, 1], tg: [1, 1, 1]
kernel __cross_entropy_nll_kernel(probs: Tensor<f32>, targets: Tensor<i32>, vocab: i32) -> Tensor<f32>:
    let i = thread_id()
    let tgt = targets[i]
    out[i] = -log(probs[i * vocab + tgt])

fn __cross_entropy_fwd(logits: Tensor<f32>, targets: Tensor<i32>) -> (Tensor<f32>, Tensor<f32>):
    let n_tokens = logits.shape[0]
    let vocab = logits.shape[1]
    let probs = __softmax_fwd(logits, 1)
    # out= must be explicit here: with no out=, kernel_dispatch_v2 infers the
    # output shape from input[0] (probs, shape [n_tokens, vocab]) rather than
    # the kernel's actual [n_tokens] output, corrupting the downstream mean
    # reduction (which then sees a 2-D tensor and reduces the wrong axis).
    let nll = __cross_entropy_nll_kernel[grid=[n_tokens, 1, 1], tg=[1, 1, 1], out=[n_tokens, 0, 0]](probs, targets, vocab)
    let loss = __reduce_mean_fwd(nll, 0, 1)
    return (loss, probs)
