# M28 — Module Trait + Generic Optimizer

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`  
**Track:** frontend  
**Depends on:** M27; generics frontend work (parallel track, developed during M24–M26)

Land generics, `impl` blocks, one trait mechanism, and `List<T>`. Author the `Module` trait, `impl Module for GPT`, and a single generic `fn adamw<M: Module>`. The nanoGPT capstone must use zero hand-unrolled parameter loops. See the V4 plan fenced scope (ADR-0007).

## Done-When

1. `examples/nanogpt.ml` uses `impl Module for GPT` and calls a single generic `fn adamw<M: Module>(model: M, ...)`. Zero hand-unrolled optimizer loops in `main`.
2. **No-unroll lint passes:** an AST check confirms the capstone contains exactly one `fn adamw<M: Module>`, zero manual `.grad` arithmetic in `main`'s train loop outside the optimizer function, and the AdamW state struct appears ≤ 1× in the program.
3. Loss still decreases (M26 gate still holds under the new abstraction).
4. A unit test exercises generic monomorphization: `fn id<T>(x: T) -> T: return x` called with `Tensor<f32>` and `i32` produces distinct correctly-typed codegen.
5. `cargo test --workspace` passes.

## Scope

### 1. Generics (`malus-syntax/src/parser.rs`, `malus-sema/src/check.rs`)

**Syntax:**
```malus
fn f<T: Trait>(x: T) -> T:
    ...

struct Wrapper<T>:
    value: T

fn id<T>(x: T) -> T:    # no bound — unconstrained type parameter
    return x
```

AST: `FnDef` gains `type_params: Vec<TypeParam>` where `TypeParam { name, bound: Option<TraitName> }`. Same for `StructDef`.

**Sema:**
- At the call site `f(my_tensor)`, instantiate the type parameter: substitute `T → Tensor<f32>`.
- Produce one `TypedFn` per instantiation (monomorphization). Store in the `TypedProgram.fns` vec alongside non-generic fns.
- Check that the passed type satisfies the bound (implements the trait). Error if not.
- V4 scope: one type parameter per item, one trait bound, no associated types, no higher-kinded types.

**Codegen-cpu:** compile each monomorphization to a distinct Cranelift function. JIT-link them. Call sites use the monomorphized symbol.

### 2. Trait mechanism (`malus-sema/src/check.rs`, new `src/traits.rs`)

**Syntax:**
```malus
trait Module:
    fn parameters(self) -> List<Tensor<f32>>

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return [self.tok_emb, self.pos_emb, ...]
```

**Sema:**
- `TraitDef { name, methods: Vec<MethodSig> }` — a method signature is `(name, param_types, return_type)`.
- `ImplBlock { trait_name, for_type, methods: Vec<FnDef> }` — type-check each method against the corresponding signature.
- Build a trait-impl registry: `HashMap<(TraitName, TypeName), Vec<TypedFn>>`.
- When type-checking a generic call `f<M: Module>(model: M)`, look up `Module` impl for `M`'s concrete type at the call site; substitute the `self` parameter.

V4 scope: one built-in trait (`Module`). The mechanism must generalize, but only `Module` is exercised in the capstone.

### 3. `List<T>` (`malus-sema/src/ty.rs`, `malus-codegen-cpu/src/lib.rs`)

**Type:** `ResolvedTy::List { elem: Box<ResolvedTy> }`.

**Runtime representation:** heap-allocated `Vec<i64>` (for `List<Tensor<f32>>`: a vec of tensor handles). Allocated with `libc::malloc`; freed with `DropList` (analogous to `DropArray`).

**Sema:** `[e1, e2, e3]` literal is inferred as `List<T>` when the context expects `List<T>` (e.g. returning from `parameters()`). When no context is available, the same literal is still `Array<T, N>` (backward compat). Disambiguate by return-type context.

**Operations:** index `lst[i]`, `for x in lst` iteration. Append (`lst.push(x)`) deferred — the capstone only needs construction + iteration.

**CTMM:** `DropList` releases each tensor-element handle via `tensor_release`, then frees the vec. Same pattern as `DropArray`.

**Grad-inference (M27 integration):** `List<Tensor<f32>>` returned from `parameters()` — its elements participate in grad-tracking. If the list escapes into the tape (because `adamw` feeds params into the optimizer which uses `no_grad`, so they do NOT escape to tape), they get static-free. If they do escape, they get RC. The M27 grad-inference pass must handle `List` elements.

### 4. Method call syntax

`model.parameters()` — syntactic sugar for `Module::parameters(model)`. Parser: `Expr::MethodCall { receiver, method, args }`. Sema: look up `method` in the trait-impl registry for the concrete type of `receiver`; dispatch to the monomorphized impl.

### 5. `examples/nanogpt.ml` rewrite

Replace the hand-unrolled parameter management:

**Before (V3 style):**
```malus
# 82 lines of hand-unrolled AdamW for each of wq, wk, wv, wo, ...
let new_m_wq = beta1 * m_wq + (1.0 - beta1) * wq.grad
...
```

**After (V4 style):**
```malus
impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return [self.tok_emb, self.pos_emb,
                self.blocks[0].wq, self.blocks[0].wk, ...]

fn adamw<M: Module>(model: M, states: AdamWState, lr: f32, ...):
    for p in model.parameters():
        ...   # one loop body, operates on each Tensor<f32>

# in main:
adamw(gpt, states, lr, ...)
```

### 6. No-unroll lint

An AST-level check (in `malus-cli` or a new lint pass) for the capstone file:
- Count occurrences of `fn adamw` with a generic type parameter → must be exactly 1.
- Count direct `.grad` arithmetic in the `main` function's train-loop body (outside any `fn` call) → must be 0.
- Count struct definitions containing `_m: Tensor<f32>` and `_v: Tensor<f32>` fields → must be ≤ 1.

Report as a warning or error; fail CI if any violation found in `examples/nanogpt.ml`.

## Out of Scope

- `Dict<K, V>` — post-V4.
- Named submodule nesting / `state_dict` — post-V4.
- Multiple trait bounds (`T: A + B`) — post-V4.
- Default method implementations in traits — post-V4.
- `Option<T>` — post-V4.
- Enum generics — not needed for `Module`; defer.
