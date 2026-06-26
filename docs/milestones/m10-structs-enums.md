# M10 — Structs + Enums

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`

User-defined product types (structs) and sum types (data-carrying enums with match).

## Done-When

```malus
struct Layer:
    weights: Tensor<f32>
    bias: Tensor<f32>

enum Activation:
    Relu
    Sigmoid

fn activate(x: Tensor<f32>, act: Activation) -> Tensor<f32>:
    match act:
        Relu:
            return relu(x)
        Sigmoid:
            return sigmoid(x)

fn linear(x: Tensor<f32>, layer: Layer, act: Activation) -> Tensor<f32>:
    let h = x @ layer.weights + layer.bias
    return activate(h, act)

fn main():
    let l = Layer(weights=ones(3, 4), bias=zeros(1, 4))
    let x = ones(2, 3)
    let out = linear(x, l, Activation.Relu)
    println("linear+relu output: {}", out)
```

## Scope

### 1. Struct Declarations

Both `struct` and `enum` keywords are already lexed (`TokenKind::Struct`, `TokenKind::Enum`).

**AST** (`malus-syntax/src/ast.rs`):
- Add `ItemKind::Struct { name: String, fields: Vec<FieldDef> }` where `FieldDef = { name: String, ty: Ty }`

**Parser** (`malus-syntax/src/parser.rs`):
- Parse `struct <Name>: NEWLINE INDENT (<field>: <type> NEWLINE)* DEDENT`

**Type system** (`malus-sema/src/ty.rs`):
- Add `ResolvedTy::Struct { name: String, fields: Vec<(String, ResolvedTy)> }`

**Sema** (`malus-sema/src/check.rs`, `malus-sema/src/env.rs`):
- Pass 1 (signature collection): register all struct types in a new `env.structs: HashMap<String, Vec<(String, ResolvedTy)>>` map
- Resolve `Ty::Named(name)` by looking up `env.structs` — currently all `Named` types produce `UnknownType` errors
- Type-check struct construction `Name(field1=expr1, field2=expr2)`:
  - Match field names against the struct definition
  - Check field expression types
  - Allow positional construction (fields in declaration order) as an alternative
- Type-check field access `expr.field`: look up field type in the struct definition

### 2. Struct Codegen (`malus-codegen-cpu/src/lib.rs`)

Structs are heap-allocated. Represent a struct value as a `Box<[u64]>` — a flat array of field values packed as 64-bit slots (tensors are already `i64` handles; scalars widen to `i64`/`f64`).

**Construction:** `malloc(field_count * 8)`, store each field value at `ptr + field_index * 8`. Return the pointer as an `i64` (same opaque handle pattern as tensors).

**Field access:** Load from `ptr + field_index * 8`. Cast appropriately based on the field's resolved type.

**Passing structs:** Structs are passed by pointer (`i64`). When passed as a function argument, the pointer is passed directly — no copy.

**Returning structs:** Return a pointer to a heap-allocated struct. Caller is responsible for freeing.

### 3. Struct CTMM (`malus-sema/src/ctmm.rs`)

When a tensor is stored into a struct field, its lifetime becomes structurally ambiguous — the struct controls when it dies. Use RC (per ADR-0002 and M9).

Rules:
- When a tensor binding is stored into a struct field during construction: emit `tensor_retain` for that binding before the struct constructor call.
- When a struct binding goes out of scope (`Drop` for the struct): emit `tensor_release` for each tensor field.
- If a struct field is a nested struct, recursively apply the same rules.

In `collect_escaping` (or a new `collect_struct_escapes` pass), detect bindings that flow into struct field slots. Mark them as `Rc`-managed rather than `Static`.

### 4. Enum Declarations

**AST** (`malus-syntax/src/ast.rs`):
- Add `ItemKind::Enum { name: String, variants: Vec<VariantDef> }` where `VariantDef = { name: String, fields: Vec<FieldDef> }` (fields empty for tag-only variants)

**Parser** (`malus-syntax/src/parser.rs`):
- Parse `enum <Name>: NEWLINE INDENT (<Variant>[(<field>: <type>, ...)] NEWLINE)* DEDENT`

**Type system** (`malus-sema/src/ty.rs`):
- Add `ResolvedTy::Enum { name: String, variants: Vec<(String, Vec<(String, ResolvedTy)>)> }`

**Sema**:
- Pass 1: register all enum types in `env.enums`
- Resolve `Ty::Named(name)` against both `env.structs` and `env.enums`
- Type-check variant construction `Enum.Variant` (tag-only) and `Enum.Variant(field=val)` (data-carrying)

### 5. Match Expression

**Lexer:** `"match" => TokenKind::Match` — add to `scan_ident_or_keyword` (was noted as needed in M7 for the `mut` keyword change).

**AST** (`malus-syntax/src/ast.rs`):
- Add `StmtKind::Match { scrutinee: Expr, arms: Vec<MatchArm> }` where `MatchArm = { variant: String, bindings: Vec<String>, body: Vec<Stmt> }`

**Parser** (`malus-syntax/src/parser.rs`):
- Parse `match <expr>: NEWLINE INDENT (<Variant>[(<bindings>)]: <body>)* DEDENT`

**Typed IR** (`malus-sema/src/typed_ir.rs`):
- Add `TypedStmt::Match { scrutinee: TypedExpr, arms: Vec<TypedMatchArm> }`

**Sema**:
- Verify scrutinee is an enum type
- Check exhaustiveness — all variants must have exactly one arm
- Each arm body is checked with the binding names added to scope, typed as the corresponding field types

### 6. Enum Codegen

Enums are represented as tagged unions: a `u32` tag (variant index) followed by the largest-variant payload, allocated on the heap as a `Box<[u8]>`.

**Construction:** Allocate `sizeof(u32) + sizeof(largest_variant)`, write tag, write field values.

**Match:** Switch on the tag value. Each arm reads its fields from the payload region at the appropriate offsets.

In Cranelift: emit a jump table or chain of `brif` comparisons on the tag field. For V1 (few variants), a chain of comparisons is simpler.

## Out of Scope

- `Option<T>` and generics — no generic types in V1
- Recursive types (struct containing itself)
- Enum variants with more than ~4 fields (no hard limit, but testing focus is small variants)
- Enum CTMM (enums with tensor fields follow the same RC rules as struct tensor fields, but M10 can defer this to M11 integration if it's too much)
- `match` as an expression (returns a value) — V1 match is a statement only
