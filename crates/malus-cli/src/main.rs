// Entry point for the `malus` CLI.
// Usage:
//   malus <script.malus>   — JIT-compile and run a script
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
    match malus_loader::ModuleLoader::new().load(abs) {
        Ok(loaded) => println!("{:#?}", loaded.program),
        Err(e) => {
            eprintln!("malus: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_repl() {
    eprintln!("malus: REPL not yet implemented");
    std::process::exit(1);
}
