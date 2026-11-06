//! Phase 16 acceptance: with the OS sandbox on, an inside-classified script cannot write outside the
//! working directory, while a command the user approved as outside runs unsandboxed.
#![cfg(any(target_os = "linux", target_os = "macos"))]
#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use h_agent_core::config::SafetyConfig;
use h_agent_core::{AgentId, Mode};
use h_agent_safety::{PathGuard, PermissionEngine};
use h_agent_sandbox::Sandbox;
use h_agent_tools::{
    ApprovalRequest, ApprovalResponse, Approver, ToolCtx, ToolEnv, ToolExecutor, ToolLimits,
    ToolRegistry,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Approves everything and remembers how many times it was asked.
#[derive(Default)]
struct CountingApprover(Mutex<usize>);

#[async_trait]
impl Approver for CountingApprover {
    async fn request(&self, _req: ApprovalRequest) -> ApprovalResponse {
        *self.0.lock().unwrap() += 1;
        ApprovalResponse::AllowOnce
    }
}

const LAUNCHER: &str = env!("CARGO_BIN_EXE_h_agent_sandbox_launcher");

/// Whether this machine can enforce the sandbox; CI runners without Landlock skip the test.
fn available(work: &Path) -> bool {
    if cfg!(target_os = "macos") {
        return Path::new("/usr/bin/sandbox-exec").exists();
    }
    let status = std::process::Command::new(LAUNCHER)
        .arg("--write")
        .arg(work)
        .args(["--", "/bin/sh", "-c", "true"])
        .status()
        .unwrap();
    status.success()
}

#[tokio::test]
async fn inside_script_cannot_write_outside_but_approved_outside_command_can() {
    let tmp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    if !available(&work) {
        eprintln!("skipping: the OS sandbox is not available on this machine");
        return;
    }
    let script = work.join("w.sh");
    std::fs::write(&script, "#!/bin/sh\ntouch ../x\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

    let safety = SafetyConfig::default();
    // The temp directory holds `root`, so it is deliberately not writable here.
    let sandbox = Sandbox::new(
        PathBuf::from(LAUNCHER),
        Vec::new(),
        vec![work.clone(), PathBuf::from("/dev/null")],
    );
    let env = ToolEnv::new(
        PathGuard::new(&work).unwrap(),
        std::env::vars(),
        &safety.env_denylist,
    )
    .with_sandbox(sandbox);
    let ctx = ToolCtx {
        env: Arc::new(env),
        limits: ToolLimits::default(),
        cancel: CancellationToken::new(),
        agent: AgentId::new(),
    };
    let approver = Arc::new(CountingApprover::default());
    let exec = ToolExecutor::new(
        Arc::new(ToolRegistry::with_builtins()),
        Arc::new(Mutex::new(PermissionEngine::new(
            Mode::Auto,
            &safety.always_ask_patterns,
        ))),
        approver.clone(),
    );
    let shell = |command: &str| json!({ "command": command, "shell": "sh" });

    let ok = exec
        .execute("shell", shell("touch made_inside"), &ctx, None)
        .await;
    assert!(!ok.output.is_error, "{}", ok.output.content);
    assert!(work.join("made_inside").exists());

    let blocked = exec.execute("shell", shell("./w.sh"), &ctx, None).await;
    assert_eq!(
        *approver.0.lock().unwrap(),
        0,
        "inside commands run without asking in auto mode"
    );
    assert!(blocked.output.is_error, "{}", blocked.output.content);
    assert!(
        blocked.output.content.contains("OS sandbox"),
        "{}",
        blocked.output.content
    );
    assert!(
        !root.join("x").exists(),
        "the sandbox let the script write outside"
    );

    let approved = exec.execute("shell", shell("touch ../y"), &ctx, None).await;
    assert_eq!(
        *approver.0.lock().unwrap(),
        1,
        "outside commands ask even in auto mode"
    );
    assert!(!approved.output.is_error, "{}", approved.output.content);
    assert!(root.join("y").exists());
}
