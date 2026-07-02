use malus_syntax::ast::ScalarTy;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedTy {
    Tensor { dtype: ScalarTy },
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
    /// Runtime-length mutable staging buffer `Buffer<dtype>`.
    Buffer { dtype: ScalarTy },
    /// `List<T>` (V4/M28) — fixed-length-at-construction sequence. Reference-counted
    /// aggregate at runtime (ARC header + length word + N element slots), NOT the
    /// headerless static-drop layout `Array` uses. See ADR-0034.
    List { elem: Box<ResolvedTy> },
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
            ResolvedTy::Str => write!(f, "str"),
            ResolvedTy::Struct { name, .. } => write!(f, "{}", name),
            ResolvedTy::Enum { name, .. } => write!(f, "{}", name),
            ResolvedTy::Array { elem, len } => write!(f, "Array<{}, {}>", elem, len),
            ResolvedTy::Buffer { dtype } => write!(f, "Buffer<{}>", scalar_ty_name(dtype)),
            ResolvedTy::List { elem } => write!(f, "List<{}>", elem),
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

    pub fn is_buffer(&self) -> bool {
        matches!(self, ResolvedTy::Buffer { .. })
    }

    pub fn is_list(&self) -> bool {
        matches!(self, ResolvedTy::List { .. })
    }

    pub fn list_elem(&self) -> Option<&ResolvedTy> {
        match self {
            ResolvedTy::List { elem } => Some(elem),
            _ => None,
        }
    }

    /// M34: the single predicate for "does a value of this type own heap
    /// resources a drop must release?" — shared by CTMM's droppable-field
    /// filters and codegen's recursive drop emitter, so a container element
    /// or aggregate field of any of these types is never silently skipped.
    /// `Str` is a leaked whole-program-lifetime buffer (ADR-0018) and
    /// `Buffer` drops only as a standalone binding (`DropBuffer`), matching
    /// pre-M34 behavior.
    pub fn owns_heap_resources(&self) -> bool {
        match self {
            ResolvedTy::Tensor { .. }
            | ResolvedTy::Struct { .. }
            | ResolvedTy::Enum { .. }
            | ResolvedTy::List { .. }
            // The array box itself is heap-allocated, whatever its elements.
            | ResolvedTy::Array { .. } => true,
            ResolvedTy::Tuple(elems) => elems.iter().any(|t| t.owns_heap_resources()),
            _ => false,
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
