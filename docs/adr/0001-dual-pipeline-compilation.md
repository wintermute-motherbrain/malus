# Dual-pipeline compilation: Cranelift for `fn`, MSL for `kernel`

`fn` bodies are JIT-compiled via Cranelift; `kernel` bodies are lowered to Metal Shading Language (MSL) and JIT-compiled by the Apple Metal driver. This cleanly separates CPU orchestration from GPU execution at the language level, matching how the hardware actually works. A single compilation backend cannot target both CPU and GPU efficiently, and generating intermediate IR (like LLVM) for GPU code adds unnecessary complexity when Metal is the only target.

## Considered Options

- **LLVM for both**: LLVM has a Metal backend but it is not production-quality for Apple's GPU architecture. Apple invests in Metal directly; fighting through LLVM would yield worse codegen.
- **Cranelift for both via CPU fallback**: Viable for correctness testing but defeats the purpose of GPU-accelerated ML workloads.
