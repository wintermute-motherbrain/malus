// Metal API bindings, tensor memory management, and kernel dispatch.
// Manages MTLDevice, MTLCommandQueue, and MTLBuffer lifecycle.
// Uses MTLResourceStorageModeShared for zero-copy CPU/GPU access on Apple Silicon.
// Stdlib ops (matmul, reductions) delegate to Metal Performance Shaders (MPS).
// User-written kernels are dispatched as compiled MSL compute pipelines.
