//! Optional OS sandbox for shell commands (DESIGN §12.7): writes are limited to the working directory
//! and a few system locations, and TCP can be cut off or limited to listed ports. Linux uses Landlock
//! through a launcher process that restricts itself and then execs the command; macOS runs the command
//! under `sandbox-exec`. Windows has no equivalent.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// First argument that turns the `finality` binary into the sandbox launcher.
pub const LAUNCHER_ARG: &str = "__sandbox";
/// Exit code when the sandbox could not be set up; the command did not run.
pub const SETUP_FAILED: i32 = 126;
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// TCP policy for sandboxed commands. Landlock and Seatbelt filter by port, not by address, and
/// neither restricts UDP (DNS keeps working).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Network {
    /// No restriction.
    Allow,
    /// No TCP bind or connect, except connects to the listed ports.
    Deny { allowed_ports: Vec<u16> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sandbox {
    launcher: PathBuf,
    launcher_args: Vec<OsString>,
    writable: Vec<PathBuf>,
    network: Network,
}

impl Sandbox {
    /// Whether this OS has a sandbox implementation.
    pub fn supported() -> bool {
        cfg!(any(target_os = "linux", target_os = "macos"))
    }

    /// `launcher` with `launcher_args` must run [`launcher_main`] (used on Linux).
    /// Paths in `writable` are canonicalized; missing ones are dropped.
    pub fn new(launcher: PathBuf, launcher_args: Vec<OsString>, writable: Vec<PathBuf>) -> Self {
        let mut dirs: Vec<PathBuf> = Vec::new();
        for w in writable {
            if let Ok(c) = std::fs::canonicalize(&w)
                && !dirs.contains(&c)
            {
                dirs.push(c);
            }
        }
        Self {
            launcher,
            launcher_args,
            writable: dirs,
            network: Network::Allow,
        }
    }

    pub fn with_network(mut self, network: Network) -> Self {
        self.network = network;
        self
    }

    /// The working directory, the temp directory and the device nodes commands write to
    /// (not all of `/dev`, which holds block devices too).
    pub fn default_writable(workdir: &Path) -> Vec<PathBuf> {
        let mut dirs = vec![workdir.to_path_buf(), std::env::temp_dir()];
        if cfg!(unix) {
            dirs.extend(
                [
                    "/tmp",
                    "/dev/null",
                    "/dev/zero",
                    "/dev/full",
                    "/dev/random",
                    "/dev/urandom",
                    "/dev/tty",
                    "/dev/ptmx",
                    "/dev/pts",
                    "/dev/shm",
                ]
                .map(PathBuf::from),
            );
        }
        dirs
    }

    pub fn writable(&self) -> &[PathBuf] {
        &self.writable
    }

    pub fn network(&self) -> &Network {
        &self.network
    }

    /// The executable and leading arguments that run `program` inside the sandbox; the program's own
    /// arguments follow. Unsupported systems get `program` unchanged.
    pub fn wrap(&self, program: &Path) -> (PathBuf, Vec<OsString>) {
        if cfg!(target_os = "macos") {
            (
                PathBuf::from(SANDBOX_EXEC),
                vec![
                    "-p".into(),
                    macos_profile(&self.writable, &self.network).into(),
                    program.into(),
                ],
            )
        } else if cfg!(target_os = "linux") {
            let mut args = self.launcher_args.clone();
            for w in &self.writable {
                args.push("--write".into());
                args.push(w.into());
            }
            if let Network::Deny { allowed_ports } = &self.network {
                args.push("--tcp-deny".into());
                for p in allowed_ports {
                    args.push("--tcp-port".into());
                    args.push(p.to_string().into());
                }
            }
            args.push("--".into());
            args.push(program.into());
            (self.launcher.clone(), args)
        } else {
            (program.to_path_buf(), Vec::new())
        }
    }
}

/// A Seatbelt profile that allows everything except writes outside `writable` and, with
/// [`Network::Deny`], network use beyond outbound TCP to the listed ports.
pub fn macos_profile(writable: &[PathBuf], network: &Network) -> String {
    let mut profile =
        String::from("(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write*");
    for w in writable {
        let escaped = w
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        profile.push_str(&format!("\n  (subpath \"{escaped}\")"));
    }
    profile.push_str(")\n");
    if let Network::Deny { allowed_ports } = network {
        profile.push_str("(deny network*)\n");
        for p in allowed_ports {
            profile.push_str(&format!(
                "(allow network-outbound (remote tcp \"*:{p}\"))\n"
            ));
        }
    }
    profile
}

/// The launcher: `[--write DIR]... [--tcp-deny [--tcp-port N]...] -- PROGRAM [ARGS...]`. On success
/// it does not return (the process becomes PROGRAM); otherwise it prints why and returns an exit code.
pub fn launcher_main(args: Vec<OsString>) -> i32 {
    let mut writable = Vec::new();
    let mut network = Network::Allow;
    let mut it = args.into_iter();
    loop {
        match it.next() {
            Some(a) if a == "--write" => match it.next() {
                Some(dir) => writable.push(PathBuf::from(dir)),
                None => return usage(),
            },
            Some(a) if a == "--tcp-deny" => {
                network = Network::Deny {
                    allowed_ports: Vec::new(),
                };
            }
            Some(a) if a == "--tcp-port" => {
                let port = it.next().and_then(|p| p.to_str()?.parse::<u16>().ok());
                match (port, &mut network) {
                    (Some(p), Network::Deny { allowed_ports }) => allowed_ports.push(p),
                    _ => return usage(),
                }
            }
            Some(a) if a == "--" => break,
            _ => return usage(),
        }
    }
    let Some(program) = it.next() else {
        return usage();
    };
    run(&writable, &network, program, it.collect())
}

fn usage() -> i32 {
    eprintln!(
        "finality sandbox: usage: [--write DIR]... [--tcp-deny [--tcp-port N]...] -- PROGRAM [ARGS...]"
    );
    SETUP_FAILED
}

#[cfg(target_os = "linux")]
fn run(writable: &[PathBuf], network: &Network, program: OsString, args: Vec<OsString>) -> i32 {
    use std::os::unix::process::CommandExt;

    if let Err(e) = linux::restrict(writable, network) {
        eprintln!("finality sandbox: {e}");
        return SETUP_FAILED;
    }
    let err = std::process::Command::new(&program).args(&args).exec();
    eprintln!(
        "finality sandbox: cannot run {}: {err}",
        Path::new(&program).display()
    );
    127
}

#[cfg(not(target_os = "linux"))]
fn run(_writable: &[PathBuf], _network: &Network, _program: OsString, _args: Vec<OsString>) -> i32 {
    eprintln!("finality sandbox: the launcher is only used on Linux");
    SETUP_FAILED
}

#[cfg(target_os = "linux")]
mod linux {
    use std::path::PathBuf;

    use landlock::{
        ABI, AccessFs, AccessNet, CompatLevel, Compatible, NetPort, Ruleset, RulesetAttr,
        RulesetCreatedAttr, RulesetStatus, path_beneath_rules,
    };

    use super::Network;

    /// Restricts this process (and what it execs) to writing beneath `writable`, and to the TCP
    /// policy in `network`. Write rules are best effort across kernel versions; a TCP policy that the
    /// kernel cannot enforce is an error, because the user asked for it explicitly.
    pub fn restrict(writable: &[PathBuf], network: &Network) -> Result<(), String> {
        let access = AccessFs::from_write(ABI::V5);
        let mut ruleset = Ruleset::default()
            .handle_access(access)
            .map_err(|e| e.to_string())?;
        if matches!(network, Network::Deny { .. }) {
            ruleset = ruleset
                .set_compatibility(CompatLevel::HardRequirement)
                .handle_access(AccessNet::BindTcp | AccessNet::ConnectTcp)
                .map_err(|_| {
                    "this kernel cannot restrict TCP (Landlock ABI 4, Linux 6.7 or newer, is needed); \
                     set safety.sandbox_network = \"allow\" or upgrade the kernel"
                        .to_owned()
                })?
                .set_compatibility(CompatLevel::BestEffort);
        }
        let mut created = ruleset
            .create()
            .map_err(|e| e.to_string())?
            .add_rules(path_beneath_rules(writable, access))
            .map_err(|e| e.to_string())?;
        if let Network::Deny { allowed_ports } = network {
            for port in allowed_ports {
                created = created
                    .add_rule(NetPort::new(*port, AccessNet::ConnectTcp))
                    .map_err(|e| e.to_string())?;
            }
        }
        let status = created.restrict_self().map_err(|e| e.to_string())?;
        if status.ruleset == RulesetStatus::NotEnforced {
            return Err(
                "Landlock is not available on this kernel (Linux 5.13+ with Landlock enabled is needed), \
                 so the command was not run. Turn off safety.os_sandbox, or approve the command as one \
                 that may write outside the working directory."
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_allows_only_the_given_writes() {
        let dirs = [PathBuf::from("/Users/a/my \"proj\""), PathBuf::from("/tmp")];
        let p = macos_profile(&dirs, &Network::Allow);
        assert_eq!(
            p,
            "(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write*\n  (subpath \"/Users/a/my \\\"proj\\\"\")\n  (subpath \"/tmp\"))\n"
        );
        let p = macos_profile(
            &dirs,
            &Network::Deny {
                allowed_ports: vec![443],
            },
        );
        assert!(p.ends_with("(deny network*)\n(allow network-outbound (remote tcp \"*:443\"))\n"));
    }

    #[test]
    fn wrap_matches_the_platform() {
        let dir = std::env::temp_dir();
        let s = Sandbox::new(
            PathBuf::from("/opt/h_agent"),
            vec![LAUNCHER_ARG.into()],
            vec![dir.clone(), PathBuf::from("/definitely/missing/dir")],
        )
        .with_network(Network::Deny {
            allowed_ports: vec![443, 80],
        });
        assert_eq!(s.writable().len(), 1, "missing paths are dropped");
        let (exe, args) = s.wrap(Path::new("/bin/sh"));
        if cfg!(target_os = "linux") {
            assert_eq!(exe, PathBuf::from("/opt/h_agent"));
            let args: Vec<String> = args
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            assert_eq!(args[0], LAUNCHER_ARG);
            assert_eq!(args[1], "--write");
            assert_eq!(
                &args[3..],
                [
                    "--tcp-deny",
                    "--tcp-port",
                    "443",
                    "--tcp-port",
                    "80",
                    "--",
                    "/bin/sh"
                ]
            );
        } else if cfg!(target_os = "macos") {
            assert_eq!(exe, PathBuf::from(SANDBOX_EXEC));
            assert_eq!(args.len(), 3);
            assert!(args[1].to_string_lossy().contains("(deny network*)"));
        } else {
            assert_eq!((exe, args), (PathBuf::from("/bin/sh"), Vec::new()));
        }
    }

    #[test]
    fn launcher_rejects_bad_arguments() {
        assert_eq!(launcher_main(vec![]), SETUP_FAILED);
        assert_eq!(launcher_main(vec!["--write".into()]), SETUP_FAILED);
        assert_eq!(launcher_main(vec!["--".into()]), SETUP_FAILED);
        assert_eq!(launcher_main(vec!["sh".into()]), SETUP_FAILED);
        assert_eq!(
            launcher_main(vec![
                "--tcp-port".into(),
                "80".into(),
                "--".into(),
                "sh".into()
            ]),
            SETUP_FAILED,
            "ports need --tcp-deny first"
        );
        assert_eq!(
            launcher_main(vec!["--tcp-deny".into(), "--tcp-port".into(), "x".into()]),
            SETUP_FAILED
        );
    }
}
