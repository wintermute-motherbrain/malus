// Entry point for the `malus` CLI.
// Usage:
//   malus <script.ml>   — JIT-compile and run a script
//   malus               — drop into the interactive REPL (v1)

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some(path) => run_script(path),
        None => run_repl(),
    }
}

fn run_script(path: &str) {
    let abs = std::path::Path::new(path);
    let loaded = match malus_loader::ModuleLoader::new().load(abs) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("malus: {}", e);
            std::process::exit(1);
        }
    };
    let typed = match malus_sema::check(&loaded.program, &loaded.module_aliases) {
        Ok(t) => t,
        Err(errors) => {
            for e in &errors {
                eprintln!("malus: {}", e);
            }
            std::process::exit(1);
        }
    };

    #[cfg(target_os = "macos")]
    {
        let (registry, kernel_ids) = match malus_codegen_gpu::compile_kernels(&typed) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("malus: {}", e);
                std::process::exit(1);
            }
        };

        malus_runtime::runtime_init(&registry.into_hashmap());

        let symbols = malus_codegen_cpu::RuntimeSymbols {
            tensor_alloc_gpu:       malus_runtime::tensor_alloc_gpu,
            tensor_free:            malus_runtime::tensor_free,
            tensor_print:           malus_runtime::tensor_print,
            kernel_dispatch:        malus_runtime::kernel_dispatch,
            gpu_barrier:            malus_runtime::gpu_barrier,
            tensor_alloc_zeros_gpu: malus_runtime::tensor_alloc_zeros_gpu,
            tensor_alloc_ones_gpu:  malus_runtime::tensor_alloc_ones_gpu,
            tensor_matmul:          malus_runtime::tensor_matmul,
            tensor_transpose:       malus_runtime::tensor_transpose,
            tensor_sum:             malus_runtime::tensor_sum,
            tensor_len:             malus_runtime::tensor_len,
        };
        if let Err(e) = malus_codegen_cpu::compile_and_run(&typed, &symbols, &kernel_ids) {
            eprintln!("malus: {}", e);
            std::process::exit(1);
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = &typed;
        eprintln!("malus: Metal runtime requires macOS");
        std::process::exit(1);
    }
}

fn run_repl() {
    eprintln!("malus: REPL not yet implemented");
    std::process::exit(1);
}
