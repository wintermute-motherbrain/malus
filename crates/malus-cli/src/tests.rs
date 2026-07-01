use crate::lint::check_no_unroll;
use malus_syntax::{parse, FileId};

fn parse_src(src: &str) -> malus_syntax::ast::Program {
    parse(FileId(0), src).expect("fixture should parse")
}

/// M28 CI gate (docs/milestones/m28-module-trait.md §6): the actual capstone
/// file must pass the no-unroll lint.
#[test]
fn test_nanogpt_passes_no_unroll_lint() {
    let src = include_str!("../../../examples/nanogpt.ml");
    let program = parse_src(src);
    let violations = check_no_unroll(&program);
    assert!(violations.is_empty(), "no-unroll lint violations in examples/nanogpt.ml: {:#?}", violations);
}

/// A deliberately hand-unrolled optimizer (V3 shape: a plain helper fn that
/// reads `.grad` directly, no generics at all) must fail the lint — both
/// check 1 (no fn generic over `Module`) and check 2 (`.grad` read outside
/// the optimizer, since there is no recognized optimizer at all).
#[test]
fn test_no_unroll_lint_catches_hand_unrolled_optimizer() {
    let src = r#"
struct AdamW:
    lr: f32

struct Block:
    wq: Tensor<f32>

fn adamw_block(opt: AdamW, mut blk: Block, t: i64):
    let g = blk.wq.grad + 0.0 * blk.wq.data
    blk.wq = variable(blk.wq.data - opt.lr * g)

fn main():
    let wq = variable(Tensor.gpu<f32>([1.0]))
    let blk = Block(wq=wq)
    let opt = AdamW(lr=0.01)
    adamw_block(opt, blk, 1)
"#;
    let violations = check_no_unroll(&parse_src(src));
    assert!(!violations.is_empty(), "expected the hand-unrolled fixture to fail the lint");
}

/// A well-formed single generic optimizer with one `.parameters()` call and
/// no `.grad` reads elsewhere passes cleanly.
#[test]
fn test_no_unroll_lint_passes_minimal_generic_optimizer() {
    let src = r#"
struct AdamW:
    lr: f32

struct GPT:
    params: List<Tensor<f32>>

trait Module:
    fn parameters(self) -> List<Tensor<f32>>

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

fn adamw<M: Module>(model: M, opt: AdamW):
    let mut ps = model.parameters()
    for i in range(len(ps)):
        ps[i] = variable(ps[i].data - opt.lr * ps[i].grad)

fn main():
    let gpt = GPT(params=[variable(Tensor.gpu<f32>([1.0]))])
    let opt = AdamW(lr=0.01)
    adamw(gpt, opt)
"#;
    let violations = check_no_unroll(&parse_src(src));
    assert!(violations.is_empty(), "expected the minimal generic optimizer to pass, got: {:#?}", violations);
}

/// Two fns generic over `Module` — check 1 must catch it even when neither
/// reads `.grad` outside itself.
#[test]
fn test_no_unroll_lint_catches_two_module_generic_fns() {
    let src = r#"
trait Module:
    fn parameters(self) -> List<Tensor<f32>>

struct GPT:
    params: List<Tensor<f32>>

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

fn adamw<M: Module>(model: M):
    let ps = model.parameters()
    print(len(ps))

fn sgd<M: Module>(model: M):
    let ps = model.parameters()
    print(len(ps))

fn main():
    let gpt = GPT(params=[Tensor.gpu<f32>([1.0])])
    adamw(gpt)
    sgd(gpt)
"#;
    let violations = check_no_unroll(&parse_src(src));
    assert!(
        violations.iter().any(|v| v.0.contains("exactly one")),
        "expected a violation for two Module-generic fns, got: {:#?}", violations
    );
}

/// A `.grad` read inside `main` (outside the optimizer) must fail check 2,
/// even though a valid single generic optimizer also exists — this is the
/// property that distinguishes M28's retargeted lint from the original
/// spec's narrower "no `.grad` arith in main's loop" check (D6): it catches
/// hand-unrolling ANYWHERE, not just in `main`.
#[test]
fn test_no_unroll_lint_catches_grad_read_in_main() {
    let src = r#"
struct AdamW:
    lr: f32

struct GPT:
    params: List<Tensor<f32>>

trait Module:
    fn parameters(self) -> List<Tensor<f32>>

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

fn adamw<M: Module>(model: M, opt: AdamW):
    let mut ps = model.parameters()
    for i in range(len(ps)):
        ps[i] = variable(ps[i].data - opt.lr * ps[i].grad)

fn main():
    let gpt = GPT(params=[variable(Tensor.gpu<f32>([1.0]))])
    let opt = AdamW(lr=0.01)
    adamw(gpt, opt)
    let sneaky = gpt.params[0].grad
    print(sneaky)
"#;
    let violations = check_no_unroll(&parse_src(src));
    assert!(
        violations.iter().any(|v| v.0.contains("main") && v.0.contains(".grad")),
        "expected a violation for `.grad` read in main, got: {:#?}", violations
    );
}
