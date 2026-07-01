# M28 — Module Trait + Generic Optimizer

**Crates:** `malus-syntax`, `malus-sema`, `malus-codegen-cpu`  
**Track:** frontend  
**Depends on:** M27; generics frontend work (parallel track, developed during M24–M26)

Land generics, `impl` blocks, one trait mechanism, and `List<T>`. Author the `Module` trait, `impl Module for GPT`, and a single generic `fn adamw<M: Module>`. The nanoGPT capstone must use zero hand-unrolled parameter loops. See the V4 plan fenced scope (ADR-0007) and ADR-0034 (write-back model, List-as-RC-aggregate, monomorphize-before-CTMM).

**V4 scope note (post-grilling correction, see ADR-0034):** generic *functions* only — `fn adamw<M: Module>`, `fn id<T>` — plus `trait`/`impl` and built-in `List<T>`. User-defined generic structs (`struct Wrapper<T>`, shown in the original syntax sketch below) are deferred post-V4: no done-when requires them, and they add aggregate-monomorphization complexity (field layout, per-instantiation `DropStruct`, per-instantiation `struct_field_grad` keys) that nothing in this milestone exercises.

## Done-When

1. `examples/nanogpt.ml` uses `impl Module for GPT` and calls a single generic `fn adamw<M: Module>(model: M, ...)`. Zero hand-unrolled optimizer loops in `main`.
2. **No-unroll lint passes (retargeted, see §6/ADR-0034):** an AST check confirms the capstone contains exactly one `fn adamw<M: Module>`, zero `.grad` reads anywhere outside that one function, and that function contains exactly one loop over `model.parameters()`.
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

AST: `FnDef` gains `type_params: Vec<TypeParam>` where `TypeParam { name, bound: Option<TraitName> }`. **`StructDef` does NOT gain this field in M28** — per the V4 scope note above, generic structs (the `struct Wrapper<T>` sketch) are deferred post-V4; only `fn` items take type parameters this milestone.

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
        return self.params   # by IDENTITY — see correction below
```

**Correction (post-grilling, ADR-0034):** the model must store its parameters in one
canonical `List<Tensor<f32>>` field (`GPT.params`), and `parameters()` must return that
field **by identity**, not a freshly-built literal (`[self.tok_emb, self.pos_emb, ...]` as
originally sketched above). A fresh literal is a *snapshot* — the optimizer's slot
reassignment (`ps[i] = variable(...)`, §5) would mutate the snapshot, and the mutation
would never reach `forward()`, which reads the model's own field. Returning the field by
identity means the optimizer mutates the same `List` box the model holds, so the next
`forward()` call sees the update. This identity-return is why `List<T>` is a
reference-counted aggregate (§3) rather than an `Array`-style static-drop container — the
aliasing it creates crosses a call boundary and is not resolvable by M28's or M29's
(intraprocedural-only) static analysis.

**Sema:**
- `TraitDef { name, methods: Vec<MethodSig> }` — a method signature is `(name, param_types, return_type)`.
- `ImplBlock { trait_name, for_type, methods: Vec<FnDef> }` — type-check each method against the corresponding signature.
- Build a trait-impl registry: `HashMap<(TraitName, TypeName), Vec<TypedFn>>`.
- When type-checking a generic call `f<M: Module>(model: M)`, look up `Module` impl for `M`'s concrete type at the call site; substitute the `self` parameter.

V4 scope: one built-in trait (`Module`). The mechanism must generalize, but only `Module` is exercised in the capstone.

### 3. `List<T>` (`malus-sema/src/ty.rs`, `malus-codegen-cpu/src/lib.rs`)

**Type:** `ResolvedTy::List { elem: Box<ResolvedTy> }`.

**Runtime representation (corrected, ADR-0034):** **not** a headerless `libc::malloc`'d `Vec<i64>` as originally sketched. `List<T>` is a reference-counted aggregate — an 8-byte ARC-header box (allocated via the existing `call_aggregate_alloc`, the same helper struct/tuple/enum boxes use) plus an 8-byte length word, then one 8-byte slot per element: `[refcount | len | h0 | h1 | ...]`. `RetainAgg`/`ReleaseAgg` — dormant sema IR nodes that already have working codegen (`aggregate_retain`/`aggregate_release`) but are never emitted pre-M28 — are activated for `List` values. Rationale: `parameters()` returning `self.params` by identity (see §2 correction) creates aliasing across a call boundary that neither M28's nor M29's (intraprocedural-only) static analysis resolves — RC is the sound fallback, exactly the case CTMM's design reserves it for.

**Sema:** `[e1, e2, e3]` literal is inferred as `List<T>` when the context expects `List<T>` (e.g. a struct field typed `List<Tensor<f32>>`, not just a `parameters()` return). When no context is available, the same literal is still `Array<T, N>` (backward compat). Disambiguate by expected-type context threaded into the array-literal checker.

**Operations:** index `lst[i]`, `for x in lst` iteration, and a `len(lst) -> i64` builtin (reads the length word) — added because AdamW must update `params[i]`/`ms[i]`/`vs[i]` in lockstep, and bare `for p in list` iteration gives neither an index nor access to the parallel state lists. Append (`lst.push(x)`) deferred — the capstone only needs construction, indexing, iteration, and length.

**CTMM:** `DropList` decrements the refcount; at 0, releases each element (type-directed — `tensor_release` for tensor elements) then frees the box. Modeled on `DropArray` plus the refcount/length header. See the RetainAgg/ReleaseAgg activation above for how aliased `List`s (e.g. the returned `self.params`) avoid a premature `DropList`.

**Grad-inference (M27 integration):** `List<Tensor<f32>>` returned from `parameters()` — its elements participate in grad-tracking exactly as struct/array elements already do (a list is grad-tracked if any element is; indexing propagates the flag). Per the ADR-0030 "implementation gotcha," sites that emit scalar `tensor_retain`/`tensor_release` off the grad-tracked flag must additionally gate on `.ty.is_tensor()` — a `List` can be `grad_tracked == true` while its own type is not `Tensor`, and calling a tensor RC op on the list's aggregate box pointer would be type confusion. The list's own container lifetime is governed by `RetainAgg`/`ReleaseAgg` (above), not by the tensor-RC path.

### 4. Method call syntax

`model.parameters()` — **no new AST node needed.** It already parses as ordinary `ExprKind::Call { callee: FieldAccess { base: model, field: "parameters" }, args: [] }` (method-call-shaped call syntax exists in the parser today; the surface form was never method-call-specific). Sema: when a `Call`'s callee is a `FieldAccess` whose base has a concrete type with a matching trait-impl method (and it isn't a module-alias or tensor-pseudo-field access, both of which already use this same `FieldAccess`-callee shape), resolve to that method, pass `base` as the `self` argument, and lower to a monomorphized `Call` by mangled name (e.g. `GPT__parameters`). Dispatch is static — the receiver's type is known at the call site — so no vtable is needed.

### 5. `examples/nanogpt.ml` rewrite

Replace the hand-unrolled parameter management:

**Before (V3 style):**
```malus
# 82 lines of hand-unrolled AdamW for each of wq, wk, wv, wo, ...
let new_m_wq = beta1 * m_wq + (1.0 - beta1) * wq.grad
...
```

**After (V4 style, corrected per ADR-0034 — model collapses to one struct with one param list, and `parameters()` returns it by identity so writes land):**
```malus
struct GPT:
    params: List<Tensor<f32>>   # all 12 weight tensors, single-block model

impl Module for GPT:
    fn parameters(self) -> List<Tensor<f32>>:
        return self.params   # by IDENTITY — see §2/§3 corrections

fn adamw<M: Module>(model: M, mut ms: List<Tensor<f32>>, mut vs: List<Tensor<f32>>,
                     opt: AdamW, t: i64):
    let ps = model.parameters()
    for i in range(len(ps)):
        let g = ps[i].grad + opt.wd * ps[i].data
        ms[i] = opt.beta1 * ms[i] + (1.0 - opt.beta1) * g
        vs[i] = opt.beta2 * vs[i] + (1.0 - opt.beta2) * g * g
        ps[i] = variable(ps[i].data - opt.lr * (ms[i] / bc1) / (sqrt(vs[i] / bc2) + opt.eps))

# in main:
adamw(gpt, ms, vs, opt, step)
```
`forward(model: GPT, toks, B, T, C)` binds readable named locals off indices at the top
(`let wq = model.params[1]`); the transformer math below is unchanged from the V3 body.

### 6. No-unroll lint (retargeted per ADR-0034 — the `_m`/`_v` struct heuristic below no longer applies since optimizer state lives in parallel `List`s, not a struct)

An AST-level check (in `malus-cli`) for the capstone file:
- Exactly one `fn` has a generic type parameter bounded by `Module` → must be exactly 1.
- `.grad` is read **only** inside that one function — zero `.grad` reads anywhere else in the program (`main`, `forward`, or any other fn). This is the check with teeth: a hand-unrolled update *anywhere* must read `.grad`, so reintroducing one anywhere fails the gate. Strictly stronger than policing `main`'s loop alone, which would miss a hand-unrolled helper fn (as V3's `adamw_block`/`adamw_gpt_params` were).
- That one function contains exactly one loop over `model.parameters()`.

Report as a warning or error; fail CI if any violation found in `examples/nanogpt.ml`.

## Out of Scope

- `Dict<K, V>` — post-V4.
- Named submodule nesting / `state_dict` — post-V4.
- Multiple trait bounds (`T: A + B`) — post-V4.
- Default method implementations in traits — post-V4.
- `Option<T>` — post-V4.
- Enum generics — not needed for `Module`; defer.
- **User-defined generic structs (`struct Wrapper<T>`) — post-V4 (added post-grilling, ADR-0034).** Not exercised by any done-when; adds aggregate-monomorphization complexity (field layout, per-instantiation `DropStruct`, per-instantiation `struct_field_grad` keys) with no capstone benefit.
- **`lst.push(x)` / mutable-length `List` growth — post-V4.** The capstone only needs fixed-size construction, indexing, iteration, and `len()`.
- **`mut self` trait methods — post-V4.** `self` is an immutable borrow in M28; write-back propagates through the shared `List` returned by identity, not through mutating the receiver itself.
