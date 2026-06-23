# M1 — Syntax

**Crate:** `malus-syntax`
**Done when:** `malus-syntax` parses `examples/add_tensors.malus` without errors and produces a correct AST that can be pretty-printed back to equivalent source.

## Scope

Implement a lexer and recursive-descent parser that handles the full syntax needed by the MVP demo:

```malus
fn main():
    let a = Tensor.gpu<f32>([1.0, 2.0, 3.0, 4.0])
    let b = Tensor.gpu<f32>([5.0, 6.0, 7.0, 8.0])
    let c = add(a, b)
    print(c)

kernel add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    return a + b
```

## Lexer tokens

- Keywords: `fn`, `kernel`, `let`, `return`
- Literals: integer (`42`), float (`1.0`), bool (`true`, `false`)
- Identifiers
- Operators: `+`, `-`, `*`, `/`, `@`, `=`, `->`, `<`, `>`
- Delimiters: `(`, `)`, `[`, `]`, `,`, `.`, `:`
- `NEWLINE` — logical line end (not emitted inside brackets)
- `INDENT` — emitted when indentation level increases after a `:` line
- `DEDENT` — emitted when indentation level decreases (one per level)
- Whitespace (leading — drives INDENT/DEDENT), blank lines, and comments (`#`) — otherwise discarded

**INDENT/DEDENT rules:**
- The lexer tracks an indentation stack of column widths
- A `:` at the end of a line signals that the next non-blank line must be indented further → emit `INDENT`
- When a line's indentation is less than the top of the stack → emit one `DEDENT` per level popped
- Inside `(`, `)`, `[`, `]`, newlines and indentation changes are ignored (Python's implicit line continuation rule)

## AST nodes

```
Program
  Item::Fn { name, params, return_ty, body }
  Item::Kernel { name, params, return_ty, body }

Stmt::Let { name, expr }
Stmt::Return { expr }
Stmt::Expr { expr }

Expr::Call { callee, args }
Expr::BinOp { op, lhs, rhs }
Expr::Lit(Literal)
Expr::Ident(String)
Expr::TensorLiteral { placement, dtype, elements }
Expr::Index { base, indices }

Type::Tensor { dtype }
Type::Scalar(ScalarTy)
Type::Bool
Type::Tuple(Vec<Type>)

Dtype: F32 | F16 | BF16 | I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64
Placement: Cpu | Gpu
```

## Source spans

Every AST node carries a `Span { file, start, end }` for error reporting. Do not skip this — it is required by M2's type error messages.

## Tests

- Lex and parse `examples/add_tensors.malus` → no errors
- Parse a `fn` with no body → error at correct span
- Parse `Tensor.gpu<f32>([1.0])` → correct `TensorLiteral` node
- Parse `a + b` → `BinOp { op: Add, lhs: Ident("a"), rhs: Ident("b") }`
- Parse `a @ b` → `BinOp { op: Matmul, ... }`
- Round-trip: pretty-print the AST and re-parse, result equals original

## Out of scope for M1

- Slicing syntax (`a[1:3]`, `a[:, 0]`) — deferred to v1 stdlib milestone
- Struct and enum declarations — deferred to v1
- Annotations (`@threadgroup_size(...)`) — deferred to v1 kernel milestone
- `inout` parameters — deferred to v1
