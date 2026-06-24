use malus_syntax::ast::ScalarTy;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedTy {
    Tensor { dtype: ScalarTy },
    Scalar(ScalarTy),
    Bool,
    Tuple(Vec<ResolvedTy>),
    Unit,
}

impl fmt::Display for ResolvedTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolvedTy::Tensor { dtype } => write!(f, "Tensor<{}>", scalar_ty_name(dtype)),
            ResolvedTy::Scalar(s) => write!(f, "{}", scalar_ty_name(s)),
            ResolvedTy::Bool => write!(f, "bool"),
            ResolvedTy::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            ResolvedTy::Unit => write!(f, "None"),
        }
    }
}

impl ResolvedTy {
    pub fn is_tensor(&self) -> bool {
        matches!(self, ResolvedTy::Tensor { .. })
    }

    pub fn tensor_dtype(&self) -> Option<&ScalarTy> {
        match self {
            ResolvedTy::Tensor { dtype } => Some(dtype),
            _ => None,
        }
    }
}

pub fn scalar_ty_name(s: &ScalarTy) -> &'static str {
    match s {
        ScalarTy::F32 => "f32",
        ScalarTy::F16 => "f16",
        ScalarTy::Bf16 => "bf16",
        ScalarTy::I8 => "i8",
        ScalarTy::I16 => "i16",
        ScalarTy::I32 => "i32",
        ScalarTy::I64 => "i64",
        ScalarTy::U8 => "u8",
        ScalarTy::U16 => "u16",
        ScalarTy::U32 => "u32",
        ScalarTy::U64 => "u64",
    }
}

pub fn is_float_scalar(s: &ScalarTy) -> bool {
    matches!(s, ScalarTy::F32 | ScalarTy::F16 | ScalarTy::Bf16)
}
