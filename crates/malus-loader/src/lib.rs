use malus_syntax::ast::{Item, ItemKind, ModulePath, Program};
use malus_syntax::{parse, FileId, ParseError, Span};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LoadError {
    Parse { error: ParseError, path: PathBuf, source: String },
    FileNotFound { path: PathBuf, import_span: Span },
    CircularImport { cycle: Vec<PathBuf>, import_span: Span },
    UnresolvedName { name: String, module: String, span: Span },
    Io { path: PathBuf, error: std::io::Error, import_span: Span },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Parse { error: e, path, .. } =>
                write!(f, "{}: parse error: {}", path.display(), e),
            LoadError::FileNotFound { path, .. } =>
                write!(f, "module not found: {}", path.display()),
            LoadError::CircularImport { cycle, .. } => {
                let names: Vec<_> = cycle.iter().map(|p| p.display().to_string()).collect();
                write!(f, "circular import: {}", names.join(" → "))
            }
            LoadError::UnresolvedName { name, module, .. } =>
                write!(f, "cannot import '{}' from '{}': name not defined", name, module),
            LoadError::Io { path, error, .. } =>
                write!(f, "cannot read '{}': {}", path.display(), error),
        }
    }
}

// ── Result types ──────────────────────────────────────────────────────────────

/// A fully-loaded and flattened program ready for sema/codegen.
#[derive(Debug)]
pub struct LoadedProgram {
    /// All fn/kernel items from all files, import items stripped, dependencies first.
    pub program: Program,
    /// Canonical path + source text for each file, keyed by FileId (for error reporting).
    pub sources: HashMap<FileId, (PathBuf, String)>,
    /// Qualified import aliases: module name → exported fn/kernel names.
    /// Populated by `import ops`-style declarations; used by sema for name resolution.
    pub module_aliases: HashMap<String, HashSet<String>>,
}

// ── Loader ────────────────────────────────────────────────────────────────────

pub struct ModuleLoader {
    /// Canonical path → (FileId, non-import items), populated on first load.
    loaded: HashMap<PathBuf, (FileId, Vec<Item>)>,
    /// Load stack for cycle detection (ordered, innermost last).
    in_progress: Vec<PathBuf>,
    next_file_id: u32,
    sources: HashMap<FileId, (PathBuf, String)>,
    /// Accumulated module aliases from all `import X` declarations seen during loading.
    module_aliases: HashMap<String, HashSet<String>>,
}

impl ModuleLoader {
    pub fn new() -> Self {
        Self {
            loaded: HashMap::new(),
            in_progress: Vec::new(),
            next_file_id: 0,
            sources: HashMap::new(),
            module_aliases: HashMap::new(),
        }
    }

    /// Load `entry_path` and all its transitive imports. Returns a flattened program.
    pub fn load(mut self, entry_path: &Path) -> Result<LoadedProgram, LoadError> {
        self.load_file(entry_path, None)?;

        let canonical_entry = std::fs::canonicalize(entry_path).unwrap();

        // Flatten: deps first (in load order), entry file last.
        let mut all_items: Vec<Item> = Vec::new();
        for (path, (_, items)) in &self.loaded {
            if *path != canonical_entry {
                all_items.extend(items.iter().cloned());
            }
        }
        if let Some((_, items)) = self.loaded.get(&canonical_entry) {
            all_items.extend(items.iter().cloned());
        }

        Ok(LoadedProgram {
            program: Program { items: all_items },
            sources: self.sources,
            module_aliases: self.module_aliases,
        })
    }

    /// Recursively load one file and its imports.
    fn load_file(&mut self, path: &Path, import_span: Option<Span>) -> Result<FileId, LoadError> {
        let canonical = std::fs::canonicalize(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                LoadError::FileNotFound {
                    path: path.to_path_buf(),
                    import_span: import_span.unwrap_or_else(|| Span::new(FileId(0), 0, 0)),
                }
            } else {
                LoadError::Io {
                    path: path.to_path_buf(),
                    error: e,
                    import_span: import_span.unwrap_or_else(|| Span::new(FileId(0), 0, 0)),
                }
            }
        })?;

        // Already loaded — return the cached FileId.
        if let Some((fid, _)) = self.loaded.get(&canonical) {
            return Ok(*fid);
        }

        // Cycle detection.
        if let Some(cycle_start) = self.in_progress.iter().position(|p| p == &canonical) {
            let mut cycle = self.in_progress[cycle_start..].to_vec();
            cycle.push(canonical.clone());
            return Err(LoadError::CircularImport {
                cycle,
                import_span: import_span.unwrap_or_else(|| Span::new(FileId(0), 0, 0)),
            });
        }

        self.in_progress.push(canonical.clone());

        let file_id = self.alloc_file_id();
        let source = std::fs::read_to_string(&canonical).map_err(|e| LoadError::Io {
            path: canonical.clone(),
            error: e,
            import_span: import_span.unwrap_or_else(|| Span::new(FileId(0), 0, 0)),
        })?;
        self.sources.insert(file_id, (canonical.clone(), source.clone()));

        let program = parse(file_id, &source).map_err(|e| LoadError::Parse {
            error: e,
            path: canonical.clone(),
            source: source.clone(),
        })?;

        let base_dir = canonical.parent().unwrap().to_path_buf();
        let mut definition_items: Vec<Item> = Vec::new();

        for item in &program.items {
            match &item.kind {
                ItemKind::Import { path: mod_path } => {
                    let target = self.resolve_path(&base_dir, mod_path);
                    let dep_fid = self.load_file(&target, Some(mod_path.span))?;
                    // Record the alias for qualified access (sema will use this).
                    let alias = mod_path.name().to_owned();
                    let exported = self.exported_names(dep_fid);
                    self.module_aliases.insert(alias, exported);
                }
                ItemKind::FromImport { path: mod_path, names } => {
                    let target = self.resolve_path(&base_dir, mod_path);
                    let dep_fid = self.load_file(&target, Some(mod_path.span))?;
                    self.validate_names(dep_fid, names, mod_path)?;
                }
                _ => definition_items.push(item.clone()),
            }
        }

        self.loaded.insert(canonical.clone(), (file_id, definition_items));
        self.in_progress.pop();

        Ok(file_id)
    }

    /// Resolve a `ModulePath` relative to `base_dir`.
    fn resolve_path(&self, base_dir: &Path, mod_path: &ModulePath) -> PathBuf {
        let mut result = base_dir.to_path_buf();
        let n = mod_path.segments.len();
        for (i, seg) in mod_path.segments.iter().enumerate() {
            if i < n - 1 {
                result.push(seg);
            } else {
                result.push(format!("{}.ml", seg));
            }
        }
        result
    }

    /// Collect the exported fn/kernel names for an already-loaded file.
    fn exported_names(&self, file_id: FileId) -> HashSet<String> {
        self.loaded.values()
            .find(|(fid, _)| *fid == file_id)
            .map(|(_, items)| {
                items.iter().filter_map(|item| match &item.kind {
                    ItemKind::Fn { name, .. } | ItemKind::Kernel { name, .. } => {
                        Some(name.clone())
                    }
                    _ => None,
                }).collect()
            })
            .unwrap_or_default()
    }

    /// Verify that all names in a `from ... import` list are defined in the target module.
    fn validate_names(
        &self,
        dep_fid: FileId,
        names: &[(String, Span)],
        mod_path: &ModulePath,
    ) -> Result<(), LoadError> {
        let exported = self.exported_names(dep_fid);
        for (name, span) in names {
            if !exported.contains(name) {
                return Err(LoadError::UnresolvedName {
                    name: name.clone(),
                    module: mod_path.segments.join("."),
                    span: *span,
                });
            }
        }
        Ok(())
    }

    fn alloc_file_id(&mut self) -> FileId {
        let id = FileId(self.next_file_id);
        self.next_file_id += 1;
        id
    }
}

impl Default for ModuleLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use malus_syntax::ast::ItemKind;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("malus_test_{}_{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, content).unwrap();
        path
    }

    fn fn_names(prog: &Program) -> Vec<&str> {
        prog.items.iter().filter_map(|i| match &i.kind {
            ItemKind::Fn { name, .. } => Some(name.as_str()),
            ItemKind::Kernel { name, .. } => Some(name.as_str()),
            _ => None,
        }).collect()
    }

    #[test]
    fn single_file_no_imports() {
        let dir = tmp_dir("single");
        let main = write(&dir, "main.ml", "fn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        assert_eq!(fn_names(&loaded.program), vec!["main"]);
    }

    #[test]
    fn simple_import() {
        let dir = tmp_dir("simple_import");
        write(&dir, "ops.ml", "fn add():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "import ops\n\nfn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        let names = fn_names(&loaded.program);
        assert!(names.contains(&"add"), "add missing: {:?}", names);
        assert!(names.contains(&"main"), "main missing: {:?}", names);
    }

    #[test]
    fn from_import() {
        let dir = tmp_dir("from_import");
        write(&dir, "ops.ml", "fn add():\n    return 0\nfn mul():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "from ops import add\n\nfn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        let names = fn_names(&loaded.program);
        // All of ops's items are still flattened in; visibility is enforced by sema.
        assert!(names.contains(&"add"));
        assert!(names.contains(&"mul"));
        assert!(names.contains(&"main"));
    }

    #[test]
    fn dotted_path() {
        let dir = tmp_dir("dotted_path");
        write(&dir, "models/net.ml", "fn forward():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "import models.net\n\nfn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        let names = fn_names(&loaded.program);
        assert!(names.contains(&"forward"));
    }

    #[test]
    fn transitive_imports() {
        let dir = tmp_dir("transitive");
        write(&dir, "base.ml", "fn base_fn():\n    return 0\n");
        write(&dir, "mid.ml", "import base\n\nfn mid_fn():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "import mid\n\nfn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        let names = fn_names(&loaded.program);
        assert!(names.contains(&"base_fn"));
        assert!(names.contains(&"mid_fn"));
        assert!(names.contains(&"main"));
    }

    #[test]
    fn diamond_deduplication() {
        let dir = tmp_dir("diamond");
        write(&dir, "common.ml", "fn shared():\n    return 0\n");
        write(&dir, "a.ml", "import common\n\nfn a_fn():\n    return 0\n");
        write(&dir, "b.ml", "import common\n\nfn b_fn():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "import a\nimport b\n\nfn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        let names = fn_names(&loaded.program);
        // `shared` must appear exactly once.
        let shared_count = names.iter().filter(|n| **n == "shared").count();
        assert_eq!(shared_count, 1, "expected shared once, got {:?}", names);
    }

    #[test]
    fn circular_import_detected() {
        let dir = tmp_dir("circular");
        write(&dir, "b.ml", "import a\n\nfn b_fn():\n    return 0\n");
        write(&dir, "a.ml", "import b\n\nfn a_fn():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "import a\n\nfn main():\n    return 0\n");
        let err = ModuleLoader::new().load(&main).unwrap_err();
        assert!(matches!(err, LoadError::CircularImport { .. }),
            "expected CircularImport, got: {}", err);
    }

    #[test]
    fn missing_file_error() {
        let dir = tmp_dir("missing");
        let main = write(&dir, "main.ml",
            "import nonexistent\n\nfn main():\n    return 0\n");
        let err = ModuleLoader::new().load(&main).unwrap_err();
        assert!(matches!(err, LoadError::FileNotFound { .. }),
            "expected FileNotFound, got: {}", err);
    }

    #[test]
    fn unresolved_name_error() {
        let dir = tmp_dir("unresolved");
        write(&dir, "ops.ml", "fn add():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "from ops import nonexistent\n\nfn main():\n    return 0\n");
        let err = ModuleLoader::new().load(&main).unwrap_err();
        assert!(matches!(err, LoadError::UnresolvedName { ref name, .. } if name == "nonexistent"),
            "expected UnresolvedName, got: {}", err);
    }

    #[test]
    fn module_alias_recorded() {
        let dir = tmp_dir("alias");
        write(&dir, "ops.ml", "fn add():\n    return 0\nfn mul():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "import ops\n\nfn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        let aliases = &loaded.module_aliases;
        assert!(aliases.contains_key("ops"), "alias 'ops' missing");
        let exported = &aliases["ops"];
        assert!(exported.contains("add"));
        assert!(exported.contains("mul"));
    }

    #[test]
    fn deps_come_before_entry_in_flat_program() {
        let dir = tmp_dir("dep_order");
        write(&dir, "ops.ml", "fn helper():\n    return 0\n");
        let main = write(&dir, "main.ml",
            "import ops\n\nfn main():\n    return 0\n");
        let loaded = ModuleLoader::new().load(&main).unwrap();
        let names = fn_names(&loaded.program);
        // `helper` must come before `main` in the flat list.
        let helper_pos = names.iter().position(|n| *n == "helper").unwrap();
        let main_pos   = names.iter().position(|n| *n == "main").unwrap();
        assert!(helper_pos < main_pos, "expected helper before main: {:?}", names);
    }
}
