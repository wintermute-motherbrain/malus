# Built-in element-wise kernel ids are appended after user kernels

Built-in element-wise kernels (`malus_add`, `malus_sub`, `malus_mul`, `malus_div`) receive `kernel_id`s sequentially appended after user kernels (`N, N+1, …`) rather than a reserved range (`u64::MAX` counting down) or a separate id space.

## Rationale

The runtime is id-agnostic: `runtime_init` derives the MSL function name as `malus_kernel_{id}` uniformly for all kernels, and `kernel_dispatch` looks up the `MTLComputePipelineState` by `id` without distinguishing built-in vs. user. Sequential appending avoids magic numbers and reuses the existing `name_to_id` lookup path in codegen-cpu — built-ins are just four more entries in the map, looked up by name (`"malus_add"` etc.).

## Considered Options

- **Reserved range (`u64::MAX` counting down):** Rejected — magic numbers in codegen-gpu, fiddly bookkeeping if built-ins are added later.
- **Separate id space:** Rejected — invasive changes to `KernelRegistry`, `runtime_init`, `kernel_dispatch`, and `name_to_id` return type for zero functional benefit, since the runtime doesn't care.
- **Sequential append:** Chosen — simplest, collision-free by construction, preserves the property that built-ins are indistinguishable from user kernels below codegen-gpu.
