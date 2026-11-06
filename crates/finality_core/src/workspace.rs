use std::io;
use std::path::{Path, PathBuf};

/// The canonical working directory, fixed for the lifetime of the process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    workdir: PathBuf,
}

impl Workspace {
    pub fn from_dir(dir: impl AsRef<Path>) -> io::Result<Self> {
        let workdir = dunce::canonicalize(dir)?;
        if !workdir.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("{} is not a directory", workdir.display()),
            ));
        }
        Ok(Self { workdir })
    }

    pub fn from_current_dir() -> io::Result<Self> {
        Self::from_dir(std::env::current_dir()?)
    }

    pub fn workdir(&self) -> &Path {
        &self.workdir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_relative_components() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("a");
        std::fs::create_dir(&sub).unwrap();
        let ws = Workspace::from_dir(sub.join("..").join("a")).unwrap();
        assert_eq!(ws.workdir(), dunce::canonicalize(&sub).unwrap());
    }

    #[test]
    fn rejects_files() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("f.txt");
        std::fs::write(&file, "x").unwrap();
        assert!(Workspace::from_dir(&file).is_err());
    }
}
