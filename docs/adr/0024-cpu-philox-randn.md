# ADR-0024: CPU-side Philox4x32-10 for `randn`

## Status

Accepted — M19

## Context

`randn(d0, d1, …)` initialises weight tensors with standard-normal values. The choice of
random-number generator has three hard constraints for malus V3:

1. **Reproducibility across sessions** — training must be restartable; the same seed must
   produce the same weights so bugs are reproducible.
2. **GPU portability later** — `randn` today runs CPU-side; it must migrate to Metal
   Shading Language in a later milestone without changing its statistical contract.
3. **Determinism under concurrency** — per ADR-0006, malus avoids global mutable state
   where possible; the RNG state should not be shared across threads implicitly.

### Alternatives considered

| Option | Reproducibility | GPU portable | Notes |
|---|---|---|---|
| Mersenne Twister (MT19937) | ✓ seeded | ✗ stateful, 624-word state | Standard in Python / NumPy; sequential state machine; cannot be re-created on GPU without a loop |
| `rand` crate (ChaCha) | ✓ seeded | ✗ stateful | Good quality; same GPU-portability problem |
| Philox4x32-10 (counter-based) | ✓ counter | ✓ counter + key → 4 u32 | Used in Theano, JAX, PyTorch XLA; stateless; each (counter, key) pair independent |
| Xoshiro256** | ✓ seeded | Partial | Small state; not as well-vetted for ML simulation |

## Decision

Use **Philox4x32-10** (Salmon et al., 2011) as the PRNG, evaluated CPU-side in M19, with a
**per-`randn`-call counter** as the key and the **element index** as the counter.

The implementation in `malus-runtime/src/metal.rs`:

```
key   = [call_counter as u32, (call_counter >> 32) as u32]
ctr   = [element_pair_index as u32, (element_pair_index >> 32) as u32, 0, 0]
r[0..4] = philox4x32_10(ctr, key)       # → 4 raw u32 words
u1 = (r[0] as f64 + 0.5) / 4_294_967_296.0
u2 = (r[1] as f64 + 0.5) / 4_294_967_296.0
mag = sqrt(-2 * ln(u1))
z0  = mag * cos(2π * u2)                # Box-Muller
z1  = mag * sin(2π * u2)
```

Two outputs per Philox call; odd-length tensors drop the last extra.

The **call counter** is a thread-local `Cell<u64>` incremented each time `tensor_randn` is
called. This makes each `randn` call deterministically independent of call order across
*separate* `randn` calls; the element sequence within one call is determined by the element
index alone (no sequential state dependency). The counter is NOT reset between calls in the
same process — consecutive `randn` calls therefore produce statistically independent
streams.

## User-visible contract (M19)

- No user-settable seed in M19. The default seed is the accumulated call-counter value,
  which depends on how many prior `randn` calls were made in the same process. Programs that
  call `randn` once at startup get a stable stream for that element count and call index.
- The counter-based design means a user-settable seed can be added post-V3 by simply
  exposing the call counter as a `set_randn_seed(u64)` function — no statistical contract
  changes required.

## Consequences

- Box-Muller requires `ln` and `cos`/`sin`. These are evaluated in f64 for accuracy, then
  cast to f32. The small f64 → f32 cast error is negligible for ML weight initialization.
- GPU migration (post-V3): Metal Shading Language lacks standard `ln`/`cos`, but Philox is a
  pure arithmetic function (multiply–high, XOR, add) trivially translatable to MSL. The
  Box-Muller step is also trivially MSL-portable. No statistical change needed on migration.
- Thread safety: each thread has its own call counter (thread-local); no shared mutable
  state. Two threads calling `randn` independently may produce the same stream if their
  counters happen to coincide — acceptable for M19 (single-threaded programs).
