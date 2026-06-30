use ariadne::{sources, Color, Label, Report, ReportKind};
use malus_loader::LoadError;
use malus_sema::SemaError;
use malus_syntax::FileId;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some(path) => run_script(path),
        None => run_repl(),
    }
}

fn path_str(srcs: &HashMap<FileId, (PathBuf, String)>, fid: FileId) -> String {
    srcs.get(&fid)
        .map(|(p, _)| p.display().to_string())
        .unwrap_or_else(|| "<unknown>".into())
}

fn src_pairs(srcs: &HashMap<FileId, (PathBuf, String)>) -> Vec<(String, String)> {
    srcs.values()
        .map(|(p, s)| (p.display().to_string(), s.clone()))
        .collect()
}

fn emit_sema_error(e: &SemaError, srcs: &HashMap<FileId, (PathBuf, String)>) {
    let Some(s) = e.primary_span() else {
        eprintln!("error: {e}");
        return;
    };

    let fname = path_str(srcs, s.file);
    let mut b = Report::build(ReportKind::Error, fname.clone(), s.start as usize)
        .with_message(format!("{e}"))
        .with_label(
            Label::new((fname.clone(), s.start as usize..s.end as usize))
                .with_message(e.label())
                .with_color(Color::Red),
        );

    if let Some(s2) = e.secondary_span() {
        let fname2 = path_str(srcs, s2.file);
        b = b.with_label(
            Label::new((fname2, s2.start as usize..s2.end as usize))
                .with_message("previously defined here")
                .with_color(Color::Yellow),
        );
    }

    if let Some(note) = e.note() {
        b = b.with_note(note);
    }

    b.finish().eprint(sources(src_pairs(srcs))).unwrap();
}

fn emit_load_error(e: &LoadError) {
    match e {
        LoadError::Parse { error, path, source } => {
            let fname = path.display().to_string();
            let s = error.span;
            Report::build(ReportKind::Error, fname.clone(), s.start as usize)
                .with_message(format!("parse error: {error}"))
                .with_label(
                    Label::new((fname.clone(), s.start as usize..s.end as usize))
                        .with_message("unexpected token here")
                        .with_color(Color::Red),
                )
                .finish()
                .eprint(sources([(fname, source.clone())]))
                .unwrap();
        }
        other => eprintln!("error: {other}"),
    }
}

fn run_script(path: &str) {
    let abs = std::path::Path::new(path);
    let loaded = match malus_loader::ModuleLoader::new().load(abs) {
        Ok(l) => l,
        Err(e) => {
            emit_load_error(&e);
            std::process::exit(1);
        }
    };

    // Prepend stdlib items so stdlib fns/kernels are visible to the user program.
    let mut stdlib_items = malus_stdlib::stdlib_items();
    stdlib_items.extend(loaded.program.items.into_iter());
    let full_program = malus_syntax::ast::Program { items: stdlib_items };

    let typed = match malus_sema::check(&full_program, &loaded.module_aliases) {
        Ok(t) => t,
        Err(errors) => {
            for e in &errors {
                emit_sema_error(e, &loaded.sources);
            }
            std::process::exit(1);
        }
    };

    #[cfg(target_os = "macos")]
    {
        let (registry, kernel_ids) = match malus_codegen_gpu::compile_kernels(&typed) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("error: {e}");
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
            tensor_len:             malus_runtime::tensor_len,
            tensor_retain:          malus_runtime::tensor_retain,
            tensor_release:         malus_runtime::tensor_release,
            tape_record_binop:      malus_runtime::tape_record_binop,
            tape_record_unary:      malus_runtime::tape_record_unary,
            tape_register_leaf:     malus_runtime::tape_register_leaf,
            tape_pause:             malus_runtime::tape_pause,
            tape_resume:            malus_runtime::tape_resume,
            tape_clear:             malus_runtime::tape_clear,
            tape_get_grad:          malus_runtime::tape_get_grad,
            backward:               malus_runtime::backward,
            tape_zero_grad:         malus_runtime::tape_zero_grad,
            tape_record_reduce:     malus_runtime::tape_record_reduce,
            tensor_reshape:         malus_runtime::tensor_reshape,
            tensor_permute:         malus_runtime::tensor_permute,
            tape_record_perm:       malus_runtime::tape_record_perm,
            // M18 transformer stdlib.
            tensor_causal_mask:        malus_runtime::tensor_causal_mask,
            tape_record_layernorm:     malus_runtime::tape_record_layernorm,
            tape_record_cross_entropy: malus_runtime::tape_record_cross_entropy,
            // M19 randn.
            tensor_randn:              malus_runtime::tensor_randn,
            tape_record_embedding:     malus_runtime::tape_record_embedding,
            // M22 string I/O.
            malus_str_box:             malus_runtime::malus_str_box,
            malus_read_file:           malus_runtime::malus_read_file,
            malus_str_len:             malus_runtime::malus_str_len,
            malus_str_char_at:         malus_runtime::malus_str_char_at,
            malus_str_from_char:       malus_runtime::malus_str_from_char,
            // M22 rand_uniform.
            malus_rand_uniform:        malus_runtime::malus_rand_uniform,
            // M22 Buffer<i32>.
            malus_buffer_i32:          malus_runtime::malus_buffer_i32,
            malus_buffer_get_i32:      malus_runtime::malus_buffer_get_i32,
            malus_buffer_set_i32:      malus_runtime::malus_buffer_set_i32,
            malus_buffer_free:         malus_runtime::malus_buffer_free,
            malus_buffer_freeze_i32:   malus_runtime::malus_buffer_freeze_i32,
            // M22 rand_int + tensor_get_f32.
            malus_rand_int:            malus_runtime::malus_rand_int,
            malus_tensor_get_f32:      malus_runtime::malus_tensor_get_f32,
            // M25 metadata accessors + kernel_dispatch_v2.
            tensor_ndim:               malus_runtime::tensor_ndim,
            tensor_dim:                malus_runtime::tensor_dim,
            kernel_dispatch_v2:        malus_runtime::kernel_dispatch_v2,
            tape_register_backward_fn: malus_runtime::tape_register_backward_fn,
            malus_record_diff:         malus_runtime::malus_record_diff,
        };
        if let Err(e) = malus_codegen_cpu::compile_and_run(&typed, &symbols, &kernel_ids) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = &typed;
        eprintln!("error: Metal runtime requires macOS");
        std::process::exit(1);
    }
}

fn run_repl() {
    eprintln!("error: REPL not yet implemented");
    std::process::exit(1);
}
