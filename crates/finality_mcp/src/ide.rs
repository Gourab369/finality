//! The Finality VS Code extension serves editor language features over MCP and announces itself with a lock
//! file `<data>/ide/<port>.json`: `{ "port", "token", "workspaceFolders", "pid", "ideName" }`.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::config::{Origin, ServerDef, Transport};

pub const SERVER_NAME: &str = "vscode";
/// The working directory as a `file:` URI; the extension resolves paths against it and stays inside it.
pub const WORKDIR_HEADER: &str = "X-Finality-Workdir";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Lock {
    port: u16,
    token: String,
    #[serde(default)]
    workspace_folders: Vec<PathBuf>,
}

/// The editor server, once the extension has run at least once. Its address is looked up when it starts,
/// so an editor opened or reloaded later is still found.
pub fn server_def(ide_dir: &Path) -> Option<ServerDef> {
    ide_dir.is_dir().then(|| ServerDef {
        name: SERVER_NAME.to_owned(),
        origin: Origin::Ide,
        transport: Transport::Http {
            url: "http://127.0.0.1/mcp".into(),
            headers: Default::default(),
        },
        read_only_hints: true,
        tools: None,
        // Stable, so the tool list cache survives port changes.
        fingerprint: "ide-vscode-1".into(),
    })
}

fn normalized(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    let s = s.trim_end_matches('/');
    if cfg!(windows) {
        s.to_lowercase()
    } else {
        s.to_owned()
    }
}

/// Whether `workdir` is `folder` or inside it.
fn contains(folder: &Path, workdir: &Path) -> bool {
    let (f, w) = (normalized(folder), normalized(workdir));
    !f.is_empty() && (w == f || w.strip_prefix(&f).is_some_and(|r| r.starts_with('/')))
}

/// URL and bearer token of the editor window whose workspace holds `workdir`: the most specific folder wins,
/// then the newest lock file.
pub fn locate(ide_dir: &Path, workdir: &Path) -> Result<(String, String), String> {
    let mut best: Option<(usize, std::time::SystemTime, Lock)> = None;
    for entry in std::fs::read_dir(ide_dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let Some(lock) = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice::<Lock>(&b).ok())
        else {
            continue;
        };
        let Some(depth) = lock
            .workspace_folders
            .iter()
            .filter(|f| contains(f, workdir))
            .map(|f| normalized(f).len())
            .max()
        else {
            continue;
        };
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        if best
            .as_ref()
            .is_none_or(|(d, m, _)| (depth, modified) > (*d, *m))
        {
            best = Some((depth, modified, lock));
        }
    }
    best.map(|(_, _, lock)| (format!("http://127.0.0.1:{}/mcp", lock.port), lock.token))
        .ok_or_else(|| {
            format!(
                "no VS Code window with {} open is running the Finality extension",
                workdir.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_window_holding_the_workdir() {
        let tmp = tempfile::tempdir().unwrap();
        let ide = tmp.path();
        let write = |name: &str, port: u16, folders: &[&str]| {
            let body = serde_json::json!({ "port": port, "token": format!("t{port}"), "workspaceFolders": folders });
            std::fs::write(ide.join(name), body.to_string()).unwrap();
        };
        write("1.json", 1, &["/src/app"]);
        write("2.json", 2, &["/src", "/other"]);
        write("3.json", 3, &["/src/application"]);
        std::fs::write(ide.join("broken.json"), "{").unwrap();

        let found = locate(ide, Path::new("/src/app/crates")).unwrap();
        assert_eq!(found, ("http://127.0.0.1:1/mcp".into(), "t1".into()));
        assert_eq!(locate(ide, Path::new("/src/lib")).unwrap().1, "t2");
        assert!(locate(ide, Path::new("/elsewhere")).is_err());
        assert!(server_def(ide).is_some());
        assert!(server_def(&ide.join("none")).is_none());
    }
}
