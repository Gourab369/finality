//! Path guard (DESIGN §12.2): decides whether a path argument stays inside the working directory.

use std::io;
use std::path::{Component, Path, PathBuf};

const MAX_SYMLINK_DEPTH: u32 = 40;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathVerdict {
    Inside {
        resolved: PathBuf,
    },
    Outside {
        resolved: Option<PathBuf>,
        reason: String,
    },
}

impl PathVerdict {
    pub fn is_inside(&self) -> bool {
        matches!(self, PathVerdict::Inside { .. })
    }
}

#[derive(Debug, Clone)]
pub struct PathGuard {
    workdir: PathBuf,
    windows_rules: bool,
}

impl PathGuard {
    pub fn new(workdir: &Path) -> io::Result<Self> {
        Ok(Self {
            workdir: dunce::canonicalize(workdir)?,
            windows_rules: cfg!(windows),
        })
    }

    /// Applies Windows-only string rules (device names, ADS, UNC) regardless of host OS.
    pub fn with_windows_rules(mut self, on: bool) -> Self {
        self.windows_rules = on;
        self
    }

    pub fn workdir(&self) -> &Path {
        &self.workdir
    }

    pub fn check(&self, raw: &str) -> PathVerdict {
        self.check_from(&self.workdir, raw)
    }

    /// Resolves `raw` relative to `base` (an absolute, already-resolved directory).
    pub fn check_from(&self, base: &Path, raw: &str) -> PathVerdict {
        if raw.contains('\0') {
            return outside(None, "path contains a NUL byte");
        }
        if self.windows_rules
            && let Some(reason) = windows_rejection(raw)
        {
            return outside(None, reason);
        }
        let rel = Path::new(if raw.is_empty() { "." } else { raw });

        // Windows collapses `..` lexically while POSIX follows symlinks first; requiring both views
        // to stay inside is correct on either.
        let lexical = canonical_prefix(&normalize_lexical(&base.join(rel)));
        let physical = match resolve_physical(base, rel, 0) {
            Ok(p) => canonical_prefix(&p),
            Err(reason) => return outside(None, &reason),
        };
        for candidate in [&lexical, &physical] {
            if !self.contains(candidate) {
                return outside(
                    Some(candidate.clone()),
                    "resolves outside the working directory",
                );
            }
        }
        PathVerdict::Inside { resolved: physical }
    }

    pub fn contains(&self, path: &Path) -> bool {
        let mut inner = path.components();
        self.workdir
            .components()
            .all(|w| inner.next().is_some_and(|c| same_component(w, c)))
    }
}

fn outside(resolved: Option<PathBuf>, reason: &str) -> PathVerdict {
    PathVerdict::Outside {
        resolved,
        reason: reason.to_owned(),
    }
}

fn same_component(a: Component<'_>, b: Component<'_>) -> bool {
    if cfg!(windows) {
        a.as_os_str().to_string_lossy().to_lowercase()
            == b.as_os_str().to_string_lossy().to_lowercase()
    } else {
        a == b
    }
}

const DEVICE_NAMES: &[&str] = &["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"];

fn is_device_name(component: &str) -> bool {
    let stem = component
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end()
        .to_ascii_uppercase();
    if DEVICE_NAMES.contains(&stem.as_str()) {
        return true;
    }
    let numbered = |prefix: &str| {
        stem.strip_prefix(prefix).is_some_and(|n| {
            matches!(
                n,
                "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    };
    numbered("COM") || numbered("LPT")
}

fn windows_rejection(raw: &str) -> Option<&'static str> {
    let b = raw.as_bytes();
    if raw.starts_with(r"\\?\")
        || raw.starts_with(r"\\.\")
        || raw.starts_with("//?/")
        || raw.starts_with("//./")
    {
        return Some("verbatim or device namespace path");
    }
    if raw.starts_with(r"\\") || raw.starts_with("//") {
        return Some("UNC network path");
    }
    let has_drive = b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':';
    if has_drive && b.get(2).is_none_or(|c| *c != b'\\' && *c != b'/') {
        return Some("drive-relative path");
    }
    let rest = if has_drive { &raw[2..] } else { raw };
    if rest.contains(':') {
        return Some("alternate data stream or provider path");
    }
    if raw.split(['/', '\\']).any(is_device_name) {
        return Some("reserved device name");
    }
    None
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::Prefix(_) | Component::RootDir => out.push(c.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(n) => out.push(n),
        }
    }
    out
}

/// Walks `rel` from `base` the way the OS does on POSIX: each symlink is replaced by its target before `..` applies.
fn resolve_physical(base: &Path, rel: &Path, depth: u32) -> Result<PathBuf, String> {
    let rooted = rel.has_root() || matches!(rel.components().next(), Some(Component::Prefix(_)));
    let mut cur = if rooted {
        PathBuf::new()
    } else {
        base.to_path_buf()
    };
    for comp in rel.components() {
        match comp {
            Component::Prefix(_) | Component::RootDir => cur.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                cur.pop();
            }
            Component::Normal(name) => {
                cur.push(name);
                let is_link =
                    std::fs::symlink_metadata(&cur).is_ok_and(|m| m.file_type().is_symlink());
                if is_link {
                    if depth >= MAX_SYMLINK_DEPTH {
                        return Err("too many levels of symbolic links".into());
                    }
                    let target =
                        std::fs::read_link(&cur).map_err(|e| format!("cannot read link: {e}"))?;
                    let parent = cur.parent().map(Path::to_path_buf).unwrap_or_default();
                    cur = resolve_physical(&parent, &target, depth + 1)?;
                }
            }
        }
    }
    Ok(cur)
}

/// Canonicalizes the longest existing ancestor (resolving case, 8.3 names, junctions) and re-appends the rest.
fn canonical_prefix(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut tail = Vec::new();
    loop {
        if let Ok(mut c) = dunce::canonicalize(&existing) {
            for t in tail.iter().rev() {
                c.push(t);
            }
            return c;
        }
        match (
            existing.file_name().map(ToOwned::to_owned),
            existing.parent(),
        ) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                existing = parent.to_path_buf();
            }
            _ => return path.to_path_buf(),
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    struct Fixture {
        _tmp: tempfile::TempDir,
        root: PathBuf,
        guard: PathGuard,
    }

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        std::fs::create_dir_all(root.join("work/a/b")).unwrap();
        std::fs::create_dir_all(root.join("work/src")).unwrap();
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::create_dir_all(root.join("other")).unwrap();
        let guard = PathGuard::new(&root.join("work")).unwrap();
        Fixture {
            _tmp: tmp,
            root,
            guard,
        }
    }

    #[test]
    fn relative_paths() {
        let f = fixture();
        for inside in [
            "",
            ".",
            "src",
            "src/new.rs",
            "a/b/../b",
            "a/../src",
            "./a//b",
            "missing/dir/file",
        ] {
            assert!(f.guard.check(inside).is_inside(), "{inside}");
        }
        for outside in [
            "..",
            "../other",
            "a/../../other",
            "../work/../other",
            "../../x",
        ] {
            assert!(!f.guard.check(outside).is_inside(), "{outside}");
        }
        assert!(f.guard.check("../work/src").is_inside());
    }

    #[test]
    fn absolute_paths() {
        let f = fixture();
        let inside = f.root.join("work").join("src").display().to_string();
        let outside = f.root.join("other").display().to_string();
        assert!(f.guard.check(&inside).is_inside());
        assert!(!f.guard.check(&outside).is_inside());
        assert!(!f.guard.check("/").is_inside());
    }

    #[test]
    fn check_from_uses_the_given_base() {
        let f = fixture();
        let src = f.root.join("work/src");
        assert!(f.guard.check_from(&src, "../a").is_inside());
        assert!(!f.guard.check_from(&src, "../../other").is_inside());
    }

    #[test]
    fn nul_bytes_are_rejected() {
        let f = fixture();
        assert!(!f.guard.check("src\0/x").is_inside());
    }

    #[test]
    fn windows_string_rules() {
        let f = fixture();
        let g = f.guard.clone().with_windows_rules(true);
        for (raw, why) in [
            (r"\\server\share\x", "UNC"),
            ("//server/share", "UNC"),
            (r"\\?\C:\x", "verbatim"),
            (r"\\.\PhysicalDrive0", "verbatim"),
            ("C:foo", "drive-relative"),
            ("D:", "drive-relative"),
            ("file.txt:secret", "alternate data stream"),
            ("src/file.txt:stream:$DATA", "alternate data stream"),
            ("HKLM:/Software", "alternate data stream"),
            ("nul", "device"),
            ("src/CON.txt", "device"),
            ("com1", "device"),
            ("LPT9.log", "device"),
            ("aux ", "device"),
        ] {
            match g.check(raw) {
                PathVerdict::Outside { reason, .. } => {
                    assert!(reason.contains(why), "{raw}: {reason}")
                }
                v => panic!("{raw} should be rejected, got {v:?}"),
            }
        }
        for ok in ["console.log", "src/com10.txt", "nullable.rs", "auxiliary"] {
            assert!(g.check(ok).is_inside(), "{ok}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_followed() {
        use std::os::unix::fs::symlink;
        let f = fixture();
        let w = f.root.join("work");
        symlink(f.root.join("other"), w.join("out")).unwrap();
        symlink(w.join("src"), w.join("in")).unwrap();
        symlink(f.root.join("other/missing"), w.join("dangling")).unwrap();
        symlink("/", w.join("rootlink")).unwrap();
        symlink(w.join("loop2"), w.join("loop1")).unwrap();
        symlink(w.join("loop1"), w.join("loop2")).unwrap();

        assert!(!f.guard.check("out").is_inside());
        assert!(!f.guard.check("out/file").is_inside());
        assert!(f.guard.check("in/file").is_inside());
        assert!(
            !f.guard.check("dangling").is_inside(),
            "writing through a dangling link escapes"
        );
        assert!(
            !f.guard.check("rootlink/../etc").is_inside(),
            "physical .. after a symlink"
        );
        assert!(!f.guard.check("loop1/x").is_inside());
    }

    #[cfg(windows)]
    #[test]
    fn junctions_are_followed() {
        let f = fixture();
        let w = f.root.join("work");
        let ok = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(w.join("jout"))
            .arg(f.root.join("other"))
            .output()
            .is_ok_and(|o| o.status.success());
        if !ok {
            eprintln!("skipping: mklink /J unavailable");
            return;
        }
        assert!(!f.guard.check("jout").is_inside());
        assert!(!f.guard.check(r"jout\file.txt").is_inside());
    }

    #[cfg(windows)]
    #[test]
    fn case_and_short_names_are_normalized() {
        let f = fixture();
        assert!(f.guard.check(r"SRC\x.rs").is_inside());
        let upper = f.root.join("work").display().to_string().to_uppercase();
        assert!(f.guard.check(&upper).is_inside());

        std::fs::create_dir(f.root.join("outsidelongdirectoryname")).unwrap();
        let out = std::process::Command::new("cmd")
            .args([
                "/C",
                "for %I in (\"outsidelongdirectoryname\") do @echo %~sI",
            ])
            .current_dir(&f.root)
            .output()
            .unwrap();
        let short = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if short.is_empty() || short.ends_with("outsidelongdirectoryname") {
            eprintln!("skipping 8.3 check: short names disabled");
            return;
        }
        assert!(!f.guard.check(&short).is_inside(), "{short}");
    }

    /// Independent oracle: lexical normalization of the joined path (no symlinks exist in the fixture).
    fn oracle_inside(f: &Fixture, raw: &str) -> bool {
        let joined = f.guard.workdir().join(raw);
        let mut parts: Vec<String> = Vec::new();
        let mut root_len = 0;
        for c in joined.components() {
            match c {
                Component::ParentDir => {
                    if parts.len() > root_len {
                        parts.pop();
                    }
                }
                Component::CurDir => {}
                Component::Prefix(_) | Component::RootDir => {
                    root_len += 1;
                    parts.push(c.as_os_str().to_string_lossy().to_string());
                }
                Component::Normal(n) => parts.push(n.to_string_lossy().to_string()),
            }
        }
        let norm: PathBuf = parts.iter().collect();
        f.guard.contains(&norm)
    }

    fn component() -> impl Strategy<Value = &'static str> {
        prop::sample::select(vec!["a", "b", "src", "work", "other", "..", ".", "x.rs"])
    }

    proptest! {
        #[test]
        fn relative_paths_match_oracle(parts in prop::collection::vec(component(), 0..8), backslash in any::<bool>()) {
            let f = fixture();
            let sep = if backslash && cfg!(windows) { "\\" } else { "/" };
            let raw = parts.join(sep);
            prop_assert_eq!(f.guard.check(&raw).is_inside(), oracle_inside(&f, &raw), "{}", raw);
        }

        #[test]
        fn absolute_paths_match_oracle(parts in prop::collection::vec(component(), 0..8)) {
            let f = fixture();
            let raw = format!("{}/{}", f.root.display(), parts.join("/"));
            prop_assert_eq!(f.guard.check(&raw).is_inside(), oracle_inside(&f, &raw), "{}", raw);
        }

        #[test]
        fn arbitrary_input_never_panics(raw in ".{0,40}") {
            let f = fixture();
            let _ = f.guard.clone().with_windows_rules(true).check(&raw);
            let _ = f.guard.check(&raw);
        }
    }
}
