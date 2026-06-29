use std::collections::{HashMap, HashSet};
use malus_syntax::ast::Placement;
use malus_syntax::Span;
use crate::builtins::BuiltinSig;
use crate::ty::ResolvedTy;

// ── Nominal type definitions ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StructDef {
    pub fields: Vec<(String, ResolvedTy)>,
    pub defined_at: Span,
}

#[derive(Debug, Clone)]
pub struct VariantSig {
    pub name: String,
    pub fields: Vec<(String, ResolvedTy)>,
}

#[derive(Debug, Clone)]
pub struct EnumDef {
    pub variants: Vec<VariantSig>,
    pub defined_at: Span,
}

// ── Signatures ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParamSig {
    pub name: String,
    pub ty: ResolvedTy,
    /// True if the parameter was declared `mut`: interior mutation (`p[i]=e`,
    /// `p.f=e`) is permitted; bare rebind (`p = e`) is still rejected.
    pub is_mut: bool,
}

#[derive(Debug, Clone)]
pub struct KernelParamSig {
    #[allow(dead_code)]
    pub inout: bool,
    pub name: String,
    pub ty: ResolvedTy,
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub params: Vec<ParamSig>,
    pub return_ty: ResolvedTy,
    pub defined_at: Span,
}

#[derive(Debug, Clone)]
pub struct KernelSig {
    pub params: Vec<KernelParamSig>,
    pub return_ty: ResolvedTy,
    pub defined_at: Span,
}

// ── Callee resolution result ──────────────────────────────────────────────────

pub enum Callee<'a> {
    Fn(&'a FnSig),
    Kernel(&'a KernelSig),
    Builtin(&'a BuiltinSig),
}

// ── Environment ───────────────────────────────────────────────────────────────

pub struct Env {
    /// Local variable bindings: name → (type, optional placement).
    bindings: Vec<HashMap<String, (ResolvedTy, Option<Placement>)>>,
    /// Names bound with `let mut` or as `mut` params — checked at Assign sites.
    mutable_names: HashSet<String>,
    /// Names that are specifically `mut` parameters (subset of `mutable_names`).
    /// These allow interior mutation (`p[i]=e`) but reject bare rebind (`p=e`).
    mut_param_names: HashSet<String>,
    pub functions: HashMap<String, FnSig>,
    pub kernels: HashMap<String, KernelSig>,
    pub builtins: HashMap<String, BuiltinSig>,
    /// Qualified import aliases: module name → set of exported names.
    pub module_aliases: HashMap<String, HashSet<String>>,
    /// User-defined struct types.
    pub structs: HashMap<String, StructDef>,
    /// User-defined enum types.
    pub enums: HashMap<String, EnumDef>,
}

impl Env {
    pub fn new(
        builtins: HashMap<String, BuiltinSig>,
        module_aliases: HashMap<String, HashSet<String>>,
    ) -> Self {
        Env {
            bindings: vec![HashMap::new()],
            mutable_names: HashSet::new(),
            mut_param_names: HashSet::new(),
            functions: HashMap::new(),
            kernels: HashMap::new(),
            builtins,
            module_aliases,
            structs: HashMap::new(),
            enums: HashMap::new(),
        }
    }

    // ── Scope management ──────────────────────────────────────────────────────

    pub fn push_scope(&mut self) {
        self.bindings.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.bindings.pop();
    }

    pub fn bind(&mut self, name: String, ty: ResolvedTy, placement: Option<Placement>) {
        if let Some(scope) = self.bindings.last_mut() {
            scope.insert(name, (ty, placement));
        }
    }

    pub fn bind_mutable(&mut self, name: String, ty: ResolvedTy, placement: Option<Placement>) {
        self.mutable_names.insert(name.clone());
        self.bind(name, ty, placement);
    }

    /// Bind a `mut` parameter: interior mutation permitted, bare rebind rejected.
    pub fn bind_mut_param(&mut self, name: String, ty: ResolvedTy, placement: Option<Placement>) {
        self.mutable_names.insert(name.clone());
        self.mut_param_names.insert(name.clone());
        self.bind(name, ty, placement);
    }

    pub fn is_mutable(&self, name: &str) -> bool {
        self.mutable_names.contains(name)
    }

    /// True iff `name` was bound as a `mut` parameter (not a `let mut` local).
    pub fn is_mut_param(&self, name: &str) -> bool {
        self.mut_param_names.contains(name)
    }

    pub fn lookup_binding(&self, name: &str) -> Option<&(ResolvedTy, Option<Placement>)> {
        for scope in self.bindings.iter().rev() {
            if let Some(b) = scope.get(name) {
                return Some(b);
            }
        }
        None
    }

    // ── Callee resolution ─────────────────────────────────────────────────────

    pub fn resolve_callee(&self, name: &str) -> Option<Callee<'_>> {
        if let Some(sig) = self.functions.get(name) {
            return Some(Callee::Fn(sig));
        }
        if let Some(sig) = self.kernels.get(name) {
            return Some(Callee::Kernel(sig));
        }
        if let Some(sig) = self.builtins.get(name) {
            return Some(Callee::Builtin(sig));
        }
        None
    }

    /// Resolve a qualified call like `ops.add` — looks up the module alias then
    /// returns the callee for the bare name.
    pub fn resolve_qualified(&self, module: &str, name: &str) -> Option<Callee<'_>> {
        let exports = self.module_aliases.get(module)?;
        if exports.contains(name) {
            self.resolve_callee(name)
        } else {
            None
        }
    }
}
