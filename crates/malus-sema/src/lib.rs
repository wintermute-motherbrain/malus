#[cfg(test)]
mod tests;

mod builtins;
mod check;
mod env;
mod error;
mod ctmm;
mod borrow_inference;
mod grad_inference;
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

/// Type-check and run CTMM last-use analysis on a loaded program.
///
/// Returns a fully annotated `TypedProgram` on success, or all errors found.
pub fn check(
    program: &Program,
    module_aliases: &HashMap<String, HashSet<String>>,
) -> Result<TypedProgram, Vec<SemaError>> {
    let mut typed = check_program(program, module_aliases)?;
    grad_inference::infer(&mut typed);
    ctmm::annotate_fns(&mut typed);
    Ok(typed)
}
