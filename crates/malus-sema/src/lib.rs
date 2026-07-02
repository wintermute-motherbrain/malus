#[cfg(test)]
mod tests;

mod builtins;
mod check;
mod env;
mod error;
mod ctmm;
mod borrow_inference;
mod grad_inference;
mod retain_sites;
mod ty;
mod typed_ir;

pub use check::check as check_program;
pub use error::SemaError;
pub use ty::ResolvedTy;
pub use typed_ir::{
    TypedAssignTarget, TypedExpr, TypedExprKind, TypedFn, TypedKernel, TypedKernelParam,
    TypedMatchArm, TypedParam, TypedProgram, TypedStmt,
};

use std::collections::{HashMap, HashSet};
use malus_syntax::ast::Program;

/// Compilation options for `check_with_options`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckOptions {
    /// M31 (ADR-0035): re-enable CTMM's static `GpuBarrier` insertion.
    /// Off by default — read safety is the runtime's per-buffer pending
    /// tracking + auto-flush; every static barrier is a full commit+wait,
    /// so leaving the pass on re-creates sync-per-drop. Kept as an opt-in
    /// A/B lever until V6's static commit-planner replaces it.
    pub insert_static_barriers: bool,
}

/// Type-check and run CTMM last-use analysis on a loaded program.
///
/// Returns a fully annotated `TypedProgram` on success, or all errors found.
pub fn check(
    program: &Program,
    module_aliases: &HashMap<String, HashSet<String>>,
) -> Result<TypedProgram, Vec<SemaError>> {
    check_with_options(program, module_aliases, CheckOptions::default())
}

pub fn check_with_options(
    program: &Program,
    module_aliases: &HashMap<String, HashSet<String>>,
    options: CheckOptions,
) -> Result<TypedProgram, Vec<SemaError>> {
    let mut typed = check_program(program, module_aliases)?;
    grad_inference::infer(&mut typed);
    ctmm::set_static_barriers(options.insert_static_barriers);
    ctmm::annotate_fns(&mut typed);
    Ok(typed)
}
