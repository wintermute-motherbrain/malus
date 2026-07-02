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

/// M34 CI gate (done-when #4): the named-submodule modular example passes the
/// lint as-is — still exactly one Module-generic fn, one `.parameters()` call
/// site inside it (reached once per submodule at runtime via the recursion),
/// `.grad` confined to it.
#[test]
fn test_nanogpt_modular_passes_no_unroll_lint() {
    let src = include_str!("../../../examples/nanogpt_modular.ml");
    let program = parse_src(src);
    let violations = check_no_unroll(&program);
    assert!(
        violations.is_empty(),
        "no-unroll lint violations in examples/nanogpt_modular.ml: {violations:#?}"
    );
}

/// M34 negative test (done-when #4): a hand-unrolled RECURSIVE form — a
/// per-Block helper that reads `.grad` itself instead of routing through the
/// one generic optimizer — must still fail, even though a legitimate generic
/// optimizer also exists in the program.
#[test]
fn test_no_unroll_lint_catches_hand_unrolled_submodule_helper() {
    let src = r#"
struct AdamW:
    lr: f32

struct Block:
    params: List<Tensor<f32>>

struct GPT:
    blocks: List<Block>
    params: List<Tensor<f32>>

trait Module:
    fn parameters(self) -> List<Tensor<f32>>

impl Module for Block:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params

fn adamw<M: Module>(model: M, opt: AdamW):
    let mut ps = model.parameters()
    for i in range(len(ps)):
        ps[i] = variable(ps[i].data - opt.lr * ps[i].grad)

fn update_block_by_hand(mut blk: Block, opt: AdamW):
    let mut ps = blk.parameters()
    for i in range(len(ps)):
        ps[i] = variable(ps[i].data - opt.lr * ps[i].grad)

fn main():
    let gpt = GPT(
        blocks=[Block(params=[variable(Tensor.gpu<f32>([1.0]))])],
        params=[variable(Tensor.gpu<f32>([2.0]))]
    )
    let opt = AdamW(lr=0.01)
    for i in range(len(gpt.blocks)):
        update_block_by_hand(gpt.blocks[i], opt)
    adamw(gpt, opt)
"#;
    let violations = check_no_unroll(&parse_src(src));
    assert!(
        !violations.is_empty(),
        "expected the hand-unrolled per-Block helper to fail the lint"
    );
}

/// M34 (scope 2): measure — NOT gate — the RC-op ratio of the modular
/// example's user fns. Per the M34 planning decision the ≤5% M29 gate applies
/// to the existing corpus only (`test_v4_m29_rc_ratio_gate` in malus-sema);
/// the modular form's residual RC sites are the documented structural cost of
/// List<T>/element aliasing (ADR-0034/ADR-0040), not lost borrow-inference.
/// The generous ceiling here only catches a runaway regression; the measured
/// number is recorded in the M34 addendum of m29-benchmark-results.md.
#[test]
fn test_m34_modular_example_rc_ratio_reported() {
    use malus_sema::TypedStmt;

    let src = include_str!("../../../examples/nanogpt_modular.ml");
    let mut user_program = parse_src(src);
    let user_fn_names: Vec<String> = user_program
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            malus_syntax::ast::ItemKind::Fn { name, .. } => Some(name.clone()),
            malus_syntax::ast::ItemKind::Impl { for_type, methods, .. } => {
                let _ = (for_type, methods);
                None
            }
            _ => None,
        })
        .collect();
    let mut stdlib = malus_stdlib::stdlib_items();
    stdlib.extend(user_program.items.drain(..));
    let program = malus_syntax::ast::Program { items: stdlib };
    let aliases = std::collections::HashMap::new();
    let typed = malus_sema::check(&program, &aliases).expect("type check failed");

    fn count<F: Fn(&TypedStmt) -> bool>(stmts: &[TypedStmt], pred: &F) -> usize {
        let mut n = 0;
        for s in stmts {
            if pred(s) {
                n += 1;
            }
            match s {
                TypedStmt::If { then_body, else_body, .. } => {
                    n += count(then_body, pred);
                    if let Some(eb) = else_body {
                        n += count(eb, pred);
                    }
                }
                TypedStmt::For { body, .. }
                | TypedStmt::While { body, .. }
                | TypedStmt::ForIn { body, .. }
                | TypedStmt::NoGrad { body } => n += count(body, pred),
                TypedStmt::Match { arms, .. } => {
                    for arm in arms {
                        n += count(&arm.body, pred);
                    }
                }
                _ => {}
            }
        }
        n
    }

    let is_user_fn = |name: &str| {
        user_fn_names.iter().any(|u| u == name)
            || name.starts_with("Block__")
            || name.starts_with("GPT__")
            || name.starts_with("adamw__")
    };
    let user_fns: Vec<_> = typed.fns.iter().filter(|f| is_user_fn(&f.name)).collect();
    assert!(!user_fns.is_empty(), "no user fns found to measure");

    let rc_ops: usize = user_fns
        .iter()
        .map(|f| {
            count(&f.body, &|s: &TypedStmt| {
                matches!(
                    s,
                    TypedStmt::Retain { .. }
                        | TypedStmt::Release { .. }
                        | TypedStmt::RetainAgg { .. }
                        | TypedStmt::ReleaseAgg { .. }
                )
            })
        })
        .sum();
    let baseline: usize = user_fns
        .iter()
        .map(|f| {
            count(&f.body, &|s: &TypedStmt| {
                matches!(s, TypedStmt::Let { expr, .. } if expr.ty.is_tensor())
            })
        })
        .sum();
    assert!(baseline > 0);
    let ratio = rc_ops as f64 / baseline as f64;
    println!(
        "M34 modular RC ratio: {rc_ops} RC ops / {baseline} tensor bindings = {:.1}%",
        ratio * 100.0
    );
    assert!(
        ratio <= 1.0,
        "modular RC ratio blew past the runaway ceiling: {:.1}%",
        ratio * 100.0
    );
}
