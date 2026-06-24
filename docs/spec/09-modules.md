# 09 — Modules

## Overview

malus programs can be split across multiple files. Each file is a module. The module system is file-based, static, and follows Python conventions so that ML researchers find it familiar.

## Import syntax

Two forms are supported:

```malus
# Qualified import — module name becomes a prefix for its symbols
import ops

# Selective import — named symbols are brought into the current scope directly
from ops import add, mul
```

Import declarations **must appear at the top of the file**, before any `fn` or `kernel` definitions. An import that appears after a definition is a compile-time error.

## Path resolution

Module paths are dot-separated identifiers:

```malus
import models.transformer
from utils.math import clamp
```

Resolution is always relative to the **importing file's directory**:
- Each segment except the last is a directory component.
- The last segment is the filename, with `.ml` appended.

Examples (given that the importing file is in `src/`):

| Import statement | File resolved |
|---|---|
| `import ops` | `src/ops.ml` |
| `import models.net` | `src/models/net.ml` |
| `from utils.math import clamp` | `src/utils/math.ml` |

Paths are always relative — there are no absolute imports and no search path. The standard library is always available without an import.

## Qualified vs selective imports

**Qualified** (`import ops`): the module's exported symbols are accessed with the module name as a prefix — `ops.add(a, b)`. The module name is the last segment of the path (`import models.net` → prefix `net`).

**Selective** (`from ops import add, mul`): the named symbols are brought directly into scope and called without a prefix — `add(a, b)`. Only names that are actually defined in the target module may be imported; importing an undefined name is a compile-time error.

There are no wildcard imports (`from ops import *`). Every imported name must be listed explicitly.

## Scope rules

- `import ops` makes `ops` available as a qualified prefix in the current file only. It does not re-export `ops` to files that import the current file.
- `from ops import add` brings `add` into scope in the current file only.
- Definitions in imported modules are visible to the compiler as part of the flat program but are not re-exported.

## Circular imports

Circular imports are a compile-time hard error:

```
error: circular import
  a.ml → b.ml → a.ml
```

The full cycle is reported so the dependency can be broken.

## Diamond dependencies

If module `common` is imported by both `a` and `b`, and `main` imports both `a` and `b`, `common` is loaded exactly once. Its definitions appear once in the flat program.

```
main → a → common
main → b → common   # common loaded once, deduplicated
```

## The loader

At compile time, `malus-loader` resolves all imports before semantic analysis:

1. Start from the entry file.
2. Parse each file. For each import declaration, recursively load the target.
3. Detect cycles via an in-progress stack.
4. Deduplicate via canonical path canonicalization.
5. Flatten: produce a single `Program` with all `fn`/`kernel` items, dependencies first, entry file last.
6. Pass the flat `Program` to sema and codegen.

## Errors

| Condition | Error |
|---|---|
| Import after fn/kernel definition | Parse error |
| File not found | `module not found: path/to/mod.ml` |
| Circular dependency | `circular import: a.ml → b.ml → a.ml` |
| Name not defined in target module | `cannot import 'name' from 'module': name not defined` |
| Cannot read file | I/O error with path |
