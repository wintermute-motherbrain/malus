# 02 — Syntax

## Lexical structure

malus source files are UTF-8 encoded. Comments begin with `#` and run to end of line.

### Keywords

```
fn  kernel  let  return  if  else  for  in  while  struct  enum  inout  true  false
```

### Identifiers

A sequence of letters, digits, and underscores, not starting with a digit. Case-sensitive.

### Literals

| Kind | Examples |
|---|---|
| Integer | `0`, `42`, `1_000_000` |
| Float | `1.0`, `3.14`, `1e-4`, `1.5e10` |
| Bool | `true`, `false` |
| String literal | `"path/to/file.safetensors"` — only valid as arguments to I/O functions; not a type |

### Operators

| Operator | Meaning |
|---|---|
| `+` `-` `*` `/` | Arithmetic (element-wise on tensors) |
| `@` | Matrix multiplication |
| `==` `!=` `<` `>` `<=` `>=` | Comparison |
| `and` `or` `not` | Boolean |
| `=` | Assignment (let binding only; malus has no mutation of let bindings) |
| `->` | Return type annotation |
| `<` `>` | Type parameter brackets (e.g. `Tensor<f32>`) |

## Indentation

malus is indentation-sensitive. Blocks are delimited by indentation level, not braces.

### INDENT and DEDENT tokens

The lexer tracks an indentation stack. Synthetic `INDENT` and `DEDENT` tokens are emitted as the indentation level changes:

- A line ending with `:` signals that the next non-blank line must be indented further. When that line is read, an `INDENT` token is emitted before its first real token.
- When a line's leading whitespace is less than the top of the indentation stack, one `DEDENT` token is emitted per level popped until the stack matches.
- `NEWLINE` is emitted at the end of each logical line.
- Blank lines are ignored everywhere.

### Implicit line continuation

Inside `(`, `)`, `[`, `]` brackets, `NEWLINE`, `INDENT`, and `DEDENT` tokens are suppressed. This allows multi-line expressions without explicit continuation:

```malus
let a = Tensor.gpu<f32>([
    1.0, 2.0,
    3.0, 4.0,
])
```

### Indentation rules

- Use spaces or tabs consistently within a file. Mixing is an error.
- The canonical style is 4 spaces per level.
- The REPL treats a blank line as end-of-block at the top level.

## Grammar (informal)

```
program       := item*

item          := fn_def | kernel_def

fn_def        := 'fn' ident '(' params ')' ('->' type)? ':' NEWLINE INDENT stmt+ DEDENT

kernel_def    := 'kernel' ident '(' kernel_params ')' '->' type ':' NEWLINE INDENT stmt+ DEDENT

params        := (param (',' param)*)?
kernel_params := (kernel_param (',' kernel_param)*)?
param         := ident ':' type
kernel_param  := 'inout'? ident ':' type

stmt          := let_stmt | return_stmt | expr_stmt | if_stmt | for_stmt | while_stmt

let_stmt      := 'let' ident '=' expr NEWLINE
return_stmt   := 'return' expr NEWLINE
expr_stmt     := expr NEWLINE

if_stmt       := 'if' expr ':' NEWLINE INDENT stmt+ DEDENT
                 ('else' ':' NEWLINE INDENT stmt+ DEDENT)?

for_stmt      := 'for' ident 'in' expr ':' NEWLINE INDENT stmt+ DEDENT

while_stmt    := 'while' expr ':' NEWLINE INDENT stmt+ DEDENT

expr          := or_expr
or_expr       := and_expr ('or' and_expr)*
and_expr      := not_expr ('and' not_expr)*
not_expr      := 'not' not_expr | cmp_expr
cmp_expr      := add_expr (('==' | '!=' | '<' | '>' | '<=' | '>=') add_expr)*
add_expr      := mul_expr (('+' | '-') mul_expr)*
mul_expr      := unary_expr (('*' | '/' | '@') unary_expr)*
unary_expr    := '-' unary_expr | postfix_expr
postfix_expr  := primary_expr ('[' index_args ']' | '(' call_args ')' | '.' ident)*
primary_expr  := ident | literal | '(' expr ')' | tensor_lit

tensor_lit    := 'Tensor' '.' placement '<' dtype '>' '(' '[' (expr (',' expr)*)? ']' ')'
placement     := 'cpu' | 'gpu'

index_args    := index_expr (',' index_expr)*
index_expr    := expr? ':' expr?   # slice
               | expr              # single index

call_args     := (expr (',' expr)*)?

type          := 'Tensor' '<' dtype '>'
               | scalar_type
               | 'bool'
               | '(' type (',' type)+ ')'    # tuple
               | ident                        # struct or enum name

scalar_type   := 'f32' | 'f16' | 'bf16'
               | 'i8' | 'i16' | 'i32' | 'i64'
               | 'u8' | 'u16' | 'u32' | 'u64'

dtype         := scalar_type
```

## Kernel annotations `[v1]`

Annotations appear on the line immediately before a `kernel` declaration:

```malus
@threadgroup_size(16, 16)
@shared_memory(tile: f32[16][16])
kernel matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>:
    ...
```

Annotations set the static **launch configuration** for the kernel (threadgroup dimensions, shared memory allocation). They are resolved at compile time.
