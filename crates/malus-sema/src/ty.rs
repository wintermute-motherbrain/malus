use malus_syntax::ast::ScalarTy;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedTy {
    Tensor { dtype: ScalarTy },
    Variable { dtype: ScalarTy },
    Scalar(ScalarTy),
    Bool,
    Tuple(Vec<ResolvedTy>),
    Unit,
    /// A runtime string value — opaque i64 handle to a heap-allocated
    /// `StrBox { ptr, len }`.  Whole-program lifetime (leaked); no Drop.
    /// Used for `read_file`, `str_len`, `str_char_at`, `str_from_char`.
    Str,
    /// User-defined product type. Nominal: name determines identity.
    Struct {
        name: String,
        fields: Vec<(String, ResolvedTy)>,
    },
    /// User-defined sum type. Nominal: name determines identity.
    /// `variants` is `(variant_name, [(field_name, field_ty)])`.
    Enum {
        name: String,
        variants: Vec<(String, Vec<(String, ResolvedTy)>)>,
    },
    /// Fixed-length homogeneous array `Array<T, N>`.
    Array { elem: Box<ResolvedTy>, len: usize },
}

impl fmt::Display for ResolvedTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolvedTy::Tensor { dtype } => write!(f, "Tensor<{}>", scalar_ty_name(dtype)),
            ResolvedTy::Variable { dtype } => write!(f, "Variable<{}>", scalar_ty_name(dtype)),
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
            ResolvedTy::Str => write!(f, "str"),
            ResolvedTy::Struct { name, .. } => write!(f, "{}", name),
            ResolvedTy::Enum { name, .. } => write!(f, "{}", name),
            ResolvedTy::Array { elem, len } => write!(f, "Array<{}, {}>", elem, len),
        }
    }
}

impl ResolvedTy {
    pub fn is_tensor(&self) -> bool {
        matches!(self, ResolvedTy::Tensor { .. })
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, ResolvedTy::Variable { .. })
    }

    pub fn tensor_dtype(&self) -> Option<&ScalarTy> {
        match self {
            ResolvedTy::Tensor { dtype } => Some(dtype),
            _ => None,
        }
    }

    pub fn is_struct(&self) -> bool {
        matches!(self, ResolvedTy::Struct { .. })
    }

    /// Returns the fields of a struct type.
    pub fn struct_fields(&self) -> Option<&[(String, ResolvedTy)]> {
        match self {
            ResolvedTy::Struct { fields, .. } => Some(fields),
            _ => None,
        }
    }

    pub fn is_enum(&self) -> bool {
        matches!(self, ResolvedTy::Enum { .. })
    }

    /// Returns the variants of an enum type.
    pub fn enum_variants(&self) -> Option<&[(String, Vec<(String, ResolvedTy)>)]> {
        match self {
            ResolvedTy::Enum { variants, .. } => Some(variants),
            _ => None,
        }
    }

    pub fn is_tuple(&self) -> bool {
        matches!(self, ResolvedTy::Tuple(_))
    }

    pub fn tuple_elements(&self) -> Option<&[ResolvedTy]> {
        match self {
            ResolvedTy::Tuple(ts) => Some(ts),
            _ => None,
        }
    }

    pub fn is_array(&self) -> bool {
        matches!(self, ResolvedTy::Array { .. })
    }

    pub fn array_elem(&self) -> Option<&ResolvedTy> {
        match self {
            ResolvedTy::Array { elem, .. } => Some(elem),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            ResolvedTy::Array { len, .. } => Some(*len),
            _ => None,
        }
    }

    pub fn is_str(&self) -> bool {
        matches!(self, ResolvedTy::Str)
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
