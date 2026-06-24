# Panic-only error model; no Result types or exceptions

Errors in malus crash the program with a contextual message. Shape mismatches, dtype errors, and out-of-memory are programmer errors in the ML research context — not recoverable runtime conditions. Users who want explicit failure handling can model it with `Option<T>` using the enum type system, which ships in v1.

## Considered Options

- **Result types**: Add friction to every function call in exploratory ML code where the user is iterating rapidly. PyTorch researchers are accustomed to panics, not monadic error propagation.
- **Exceptions**: Exception unwinding interacts poorly with CTMM's escape analysis — the compiler inserts `free` points based on static control flow, and exceptions create non-local exits that break that model.
