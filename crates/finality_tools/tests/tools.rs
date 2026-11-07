#![allow(clippy::unwrap_used)]

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use finality_core::config::SafetyConfig;
use finality_core::{AgentId, ApprovalDecision, Mode};
use finality_safety::{PathGuard, PermissionEngine};
use finality_tools::{
    ApprovalRequest, ApprovalResponse, Approver, Checkpoints, Execution, ToolCtx, ToolEnv,
    ToolExecutor, ToolLimits, ToolRegistry,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

type Hook = Box<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct Scripted {
    responses: Mutex<VecDeque<ApprovalResponse>>,
    seen: Mutex<Vec<ApprovalRequest>>,
    hook: Mutex<Option<Hook>>,
}

impl Scripted {
    fn push(&self, r: ApprovalResponse) {
        self.responses.lock().unwrap().push_back(r);
    }
    fn seen(&self) -> Vec<ApprovalRequest> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Approver for Scripted {
    async fn request(&self, req: ApprovalRequest) -> ApprovalResponse {
        self.seen.lock().unwrap().push(req);
        if let Some(h) = self.hook.lock().unwrap().as_ref() {
            h();
        }
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ApprovalResponse::Deny {
                note: Some("unexpected approval request".into()),
            })
    }
}

struct Fx {
    _tmp: tempfile::TempDir,
    work: PathBuf,
    ctx: ToolCtx,
    approver: Arc<Scripted>,
    exec: ToolExecutor,
}

fn fx(mode: Mode) -> Fx {
    fx_with(mode, false)
}

fn fx_with(mode: Mode, checkpoints: bool) -> Fx {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(work.join("src")).unwrap();
    std::fs::create_dir_all(work.join("ignored")).unwrap();
    std::fs::create_dir_all(root.join("other")).unwrap();
    std::fs::write(
        work.join("src/main.rs"),
        "fn main() {\n    println!(\"hi\");\n}\n",
    )
    .unwrap();
    std::fs::write(work.join("src/lib.rs"), "pub fn add() {}\n// TODO: more\n").unwrap();
    std::fs::write(work.join("README.md"), "# Demo\nTODO: docs\n").unwrap();
    std::fs::write(work.join(".gitignore"), "ignored/\n").unwrap();
    std::fs::write(work.join("ignored/skip.rs"), "TODO hidden\n").unwrap();
    std::fs::write(root.join("other/secret.txt"), "top secret\n").unwrap();

    let safety = SafetyConfig::default();
    let vars = std::env::vars().chain([("OPENAI_API_KEY".to_owned(), "sk-test".to_owned())]);
    let mut env = ToolEnv::new(PathGuard::new(&work).unwrap(), vars, &safety.env_denylist);
    if checkpoints {
        let cps = Checkpoints::new(root.join("checkpoints"));
        cps.set_session(Some("s1"));
        env = env.with_checkpoints(cps);
    }
    let ctx = ToolCtx {
        env: Arc::new(env),
        limits: ToolLimits::default(),
        cancel: CancellationToken::new(),
        agent: AgentId::new(),
    };
    let perms = Arc::new(Mutex::new(PermissionEngine::new(
        mode,
        &safety.always_ask_patterns,
    )));
    let approver = Arc::new(Scripted::default());
    let exec = ToolExecutor::new(
        Arc::new(ToolRegistry::with_builtins()),
        perms,
        approver.clone(),
    );
    Fx {
        _tmp: tmp,
        work,
        ctx,
        approver,
        exec,
    }
}

impl Fx {
    async fn run(&self, tool: &str, args: Value) -> Execution {
        self.exec.execute(tool, args, &self.ctx, None).await
    }
}

#[tokio::test]
async fn outside_path_asks_even_in_auto() {
    let f = fx(Mode::Auto);
    f.approver.push(ApprovalResponse::Deny {
        note: Some("stay in the project".into()),
    });
    let e = f
        .run("read_file", json!({ "path": "../other/secret.txt" }))
        .await;
    assert_eq!(e.approval, Some(ApprovalDecision::Denied));
    assert!(e.output.is_error && e.output.content.contains("stay in the project"));
    assert!(!e.output.content.contains("top secret"));
    let asked = &f.approver.seen()[0];
    assert!(asked.default_deny && !asked.allow_session);

    f.approver.push(ApprovalResponse::AllowSession);
    let e = f
        .run("read_file", json!({ "path": "../other/secret.txt" }))
        .await;
    assert_eq!(
        e.approval,
        Some(ApprovalDecision::AllowedOnce),
        "outside paths never get session grants"
    );
    assert!(e.output.content.contains("top secret"));
}

#[tokio::test]
async fn inside_reads_are_automatic_in_auto() {
    let f = fx(Mode::Auto);
    let e = f
        .run(
            "read_file",
            json!({ "path": "src/main.rs", "start_line": 2, "end_line": 2 }),
        )
        .await;
    assert_eq!(e.approval, Some(ApprovalDecision::AutoAllowed));
    assert_eq!(
        e.output.content,
        "     2\t    println!(\"hi\");\n[lines 2-2 of 3]"
    );
    assert!(f.approver.seen().is_empty());
}

#[tokio::test]
async fn cowork_reads_and_writes_freely_but_session_grants_cover_other_tools() {
    let f = fx(Mode::Cowork);
    let read = f.run("read_file", json!({ "path": "README.md" })).await;
    assert_eq!(read.approval, Some(ApprovalDecision::AutoAllowed));
    let write = f
        .run(
            "edit_file",
            json!({ "path": "notes.txt", "operations": [{ "op": "write", "content": "x\n" }] }),
        )
        .await;
    assert_eq!(write.approval, Some(ApprovalDecision::AutoAllowed));
    assert!(f.approver.seen().is_empty());
}

#[tokio::test]
async fn plan_mode_denies_writes() {
    let f = fx(Mode::Plan);
    let e = f
        .run(
            "edit_file",
            json!({ "path": "README.md", "operations": [{ "op": "write", "content": "x" }] }),
        )
        .await;
    assert_eq!(e.approval, Some(ApprovalDecision::PolicyDenied));
    assert_eq!(
        std::fs::read_to_string(f.work.join("README.md")).unwrap(),
        "# Demo\nTODO: docs\n"
    );
}

#[tokio::test]
async fn rejected_calls_never_reach_authorization() {
    let f = fx(Mode::Auto);
    assert_eq!(f.run("nope", json!({})).await.approval, None);
    assert_eq!(
        f.run("read_file", json!({ "file": "x" })).await.approval,
        None
    );
    let allowed: HashSet<String> = ["read_file".to_owned()].into();
    let e = f
        .exec
        .execute("shell", json!({ "command": "ls" }), &f.ctx, Some(&allowed))
        .await;
    assert!(e.output.content.starts_with("ToolNotAllowed"));
    assert_eq!(e.approval, None);
}

#[tokio::test]
async fn failing_edit_batch_leaves_file_untouched() {
    let f = fx(Mode::Auto);
    let before = std::fs::read(f.work.join("src/lib.rs")).unwrap();
    let e = f
        .run(
            "edit_file",
            json!({ "path": "src/lib.rs", "operations": [
                { "op": "replace_text", "old": "add", "new": "sum" },
                { "op": "replace_lines", "start_line": 5, "end_line": 9, "content": "" }
            ]}),
        )
        .await;
    assert!(
        e.output.is_error && e.output.content.contains("operation 2 (replace_lines)"),
        "{}",
        e.output.content
    );
    assert_eq!(e.approval, None);
    assert_eq!(std::fs::read(f.work.join("src/lib.rs")).unwrap(), before);
}

#[tokio::test]
async fn edit_applies_and_preserves_line_endings() {
    let f = fx(Mode::Auto);
    std::fs::write(f.work.join("crlf.txt"), "one\r\ntwo\r\n").unwrap();
    let e = f
        .run("edit_file", json!({ "path": "crlf.txt", "operations": [{ "op": "replace_text", "old": "two", "new": "2" }] }))
        .await;
    assert!(!e.output.is_error, "{}", e.output.content);
    assert!(e.output.content.starts_with("edited crlf.txt (+1 -1)"));
    assert_eq!(
        std::fs::read_to_string(f.work.join("crlf.txt")).unwrap(),
        "one\r\n2\r\n"
    );

    let e = f
        .run("edit_file", json!({ "path": "new/dir/file.txt", "operations": [{ "op": "write", "content": "hello\n" }] }))
        .await;
    assert!(
        e.output.content.starts_with("created new/dir/file.txt"),
        "{}",
        e.output.content
    );
    assert_eq!(
        std::fs::read_to_string(f.work.join("new/dir/file.txt")).unwrap(),
        "hello\n"
    );
}

#[tokio::test]
async fn edit_detects_changes_made_while_awaiting_approval() {
    // Edits outside the working directory still ask in cowork; their preview reads nothing, so the
    // operations are shown and applied to the file as it is after approval.
    let f = fx(Mode::Cowork);
    let target = f.work.parent().unwrap().join("other/secret.txt");
    let t = target.clone();
    *f.approver.hook.lock().unwrap() = Some(Box::new(move || {
        std::fs::write(&t, "changed by someone else\n").unwrap()
    }));
    f.approver.push(ApprovalResponse::AllowOnce);
    let e = f
        .run("edit_file", json!({ "path": "../other/secret.txt", "operations": [{ "op": "replace_text", "old": "top", "new": "very" }] }))
        .await;
    let shown = f.approver.seen()[0].detail.clone().unwrap();
    assert!(shown.contains("replace:\ntop\nwith:\nvery"), "{shown}");
    assert!(
        e.output.is_error && e.output.content.contains("not found"),
        "{}",
        e.output.content
    );
    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        "changed by someone else\n"
    );
}

#[tokio::test]
async fn cowork_asks_before_deleting_and_checkpoints_undo_edits() {
    let f = fx_with(Mode::Cowork, true);
    let cps = f.ctx.env.checkpoints.clone().unwrap();
    cps.begin("rename things");
    let edit = f
        .run("edit_file", json!({ "path": "src/lib.rs", "operations": [{ "op": "replace_text", "old": "add", "new": "sum" }] }))
        .await;
    assert_eq!(edit.approval, Some(ApprovalDecision::AutoAllowed));

    f.approver.push(ApprovalResponse::AllowOnce);
    let delete = f
        .run(
            "edit_file",
            json!({ "path": "README.md", "operations": [{ "op": "delete" }] }),
        )
        .await;
    assert_eq!(delete.approval, Some(ApprovalDecision::AllowedOnce));
    assert_eq!(delete.output.content, "deleted README.md");
    let asked = &f.approver.seen()[0];
    assert!(!asked.allow_session, "deletes get no session grant");
    assert!(!f.work.join("README.md").exists());

    let list = f.run("checkpoint_list", json!({})).await;
    assert!(
        list.output
            .content
            .contains("\"rename things\" · README.md, src/lib.rs")
            || list
                .output
                .content
                .contains("\"rename things\" · src/lib.rs, README.md"),
        "{}",
        list.output.content
    );
    let restore = f
        .run("checkpoint_restore", json!({ "checkpoint": 1 }))
        .await;
    assert_eq!(restore.approval, Some(ApprovalDecision::AutoAllowed));
    assert_eq!(
        std::fs::read_to_string(f.work.join("README.md")).unwrap(),
        "# Demo\nTODO: docs\n"
    );
    assert_eq!(
        std::fs::read_to_string(f.work.join("src/lib.rs")).unwrap(),
        "pub fn add() {}\n// TODO: more\n"
    );
    assert!(cps.list().is_empty());
}

#[tokio::test]
async fn search_names_contents_globs_and_gitignore() {
    let f = fx(Mode::Auto);
    let content = f
        .run("search_files", json!({ "pattern": "todo" }))
        .await
        .output
        .content;
    assert!(content.contains("src/lib.rs:2: // TODO: more"), "{content}");
    assert!(content.contains("README.md:2: TODO: docs"));
    assert!(
        !content.contains("ignored/"),
        "gitignored files are skipped"
    );

    let rs = f
        .run(
            "search_files",
            json!({ "pattern": "todo", "include_glob": "*.rs" }),
        )
        .await
        .output
        .content;
    assert!(rs.contains("src/lib.rs") && !rs.contains("README"));

    let names = f
        .run(
            "search_files",
            json!({ "pattern": "^(src|main\\.rs)$", "target": "name" }),
        )
        .await
        .output
        .content;
    assert!(
        names.contains("src/\n") && names.contains("src/main.rs"),
        "{names}"
    );

    let sensitive = f
        .run(
            "search_files",
            json!({ "pattern": "todo", "case_sensitive": true }),
        )
        .await
        .output
        .content;
    assert_eq!(sensitive, "no matches");

    let limited = f
        .run("search_files", json!({ "pattern": ".", "max_results": 2 }))
        .await
        .output;
    assert!(limited.truncated && limited.content.contains("results limited to 2"));

    let bad = f.run("search_files", json!({ "pattern": "(" })).await;
    assert!(bad.output.content.contains("invalid regex") && bad.approval.is_none());
}

/// Creates a file symlink; `None` when this machine cannot (Windows without developer mode).
fn symlink(target: &std::path::Path, link: &std::path::Path) -> Option<()> {
    #[cfg(unix)]
    let made = std::os::unix::fs::symlink(target, link);
    #[cfg(windows)]
    let made = std::os::windows::fs::symlink_file(target, link);
    made.ok()
}

#[tokio::test]
async fn search_skips_symlinks_that_leave_the_workdir() {
    let f = fx(Mode::Auto);
    let root = f.work.parent().unwrap();
    if symlink(&root.join("other/secret.txt"), &f.work.join("leak.txt")).is_none() {
        eprintln!("skipping: cannot create symlinks here");
        return;
    }
    symlink(&f.work.join("README.md"), &f.work.join("readme-link.md")).unwrap();
    let content = f
        .run(
            "search_files",
            json!({ "pattern": "top secret|TODO: docs", "target": "both" }),
        )
        .await
        .output
        .content;
    assert!(
        !content.contains("top secret") && !content.contains("leak.txt"),
        "{content}"
    );
    assert!(
        content.contains("readme-link.md:2: TODO: docs"),
        "links that stay inside are searched: {content}"
    );
}

#[tokio::test]
async fn outside_edit_reads_nothing_before_approval() {
    let f = fx(Mode::Auto);
    f.approver.push(ApprovalResponse::Deny { note: None });
    let args = json!({
        "path": "../other/secret.txt",
        "operations": [{ "op": "replace_text", "old": "no such text", "new": "x" }]
    });
    let e = f.run("edit_file", args.clone()).await;
    assert_eq!(e.approval, Some(ApprovalDecision::Denied));
    assert!(
        !e.output.content.contains("not found"),
        "the model must not learn whether the text exists: {}",
        e.output.content
    );
    let asked = &f.approver.seen()[0];
    assert!(asked.default_deny, "{asked:?}");
    let detail = asked.detail.as_deref().unwrap_or("");
    assert!(
        detail.contains("replace:\nno such text\nwith:\nx") && !detail.contains("top secret"),
        "{detail}"
    );

    // Once approved, the edit runs and reports normally.
    f.approver.push(ApprovalResponse::AllowOnce);
    let e = f.run("edit_file", args).await;
    assert_eq!(e.approval, Some(ApprovalDecision::AllowedOnce));
    assert!(
        e.output.is_error && e.output.content.contains("not found"),
        "{}",
        e.output.content
    );
    assert_eq!(
        std::fs::read_to_string(root_other(&f).join("secret.txt")).unwrap(),
        "top secret\n"
    );
}

fn root_other(f: &Fx) -> std::path::PathBuf {
    f.work.parent().unwrap().join("other")
}

fn shell_cmd(windows: &'static str, unix: &'static str) -> &'static str {
    if cfg!(windows) { windows } else { unix }
}

#[tokio::test]
async fn shell_runs_in_workdir_with_filtered_env() {
    let f = fx(Mode::Cowork);
    f.approver.push(ApprovalResponse::AllowOnce);
    let e = f
        .run(
            "shell",
            json!({ "command": shell_cmd("Write-Output hello; exit 3", "echo hello; exit 3") }),
        )
        .await;
    assert!(e.output.is_error);
    assert!(
        e.output.content.starts_with("exit code: 3") && e.output.content.contains("hello"),
        "{}",
        e.output.content
    );

    f.approver.push(ApprovalResponse::AllowOnce);
    let e = f
        .run(
            "shell",
            json!({ "command": shell_cmd(
                "Write-Output \"key=[$env:OPENAI_API_KEY]\"; (Get-Location).Path",
                "echo \"key=[$OPENAI_API_KEY]\"; pwd"
            )}),
        )
        .await;
    assert!(e.output.content.contains("key=[]"), "{}", e.output.content);
    let cwd_line = e.output.content.lines().last().unwrap().trim().to_owned();
    assert_eq!(dunce::canonicalize(cwd_line).unwrap(), f.work);
}

fn alive(pid: u32) -> bool {
    if cfg!(windows) {
        let out = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).contains(&pid.to_string())
    } else {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .is_ok_and(|s| s.success())
    }
}

#[tokio::test]
async fn shell_timeout_kills_the_process_tree() {
    let f = fx(Mode::Cowork);
    f.approver.push(ApprovalResponse::AllowOnce);
    let command = shell_cmd(
        "$p = Start-Process ping -ArgumentList '-n','999','127.0.0.1' -NoNewWindow -PassThru; $p.Id | Out-File -Encoding ascii pid.txt; Wait-Process -Id $p.Id",
        "sleep 999 & echo $! > pid.txt; wait",
    );
    let started = Instant::now();
    let e = f
        .run("shell", json!({ "command": command, "timeout_secs": 3 }))
        .await;
    assert!(
        e.output.content.starts_with("timed out after 3s"),
        "{}",
        e.output.content
    );
    assert!(started.elapsed() < Duration::from_secs(15));

    let pid: u32 = std::fs::read_to_string(f.work.join("pid.txt"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive(pid) && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!alive(pid), "grandchild {pid} survived the timeout");
}

#[tokio::test]
async fn shell_cancellation_stops_the_command() {
    let f = fx(Mode::Cowork);
    f.approver.push(ApprovalResponse::AllowOnce);
    let cancel = f.ctx.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        cancel.cancel();
    });
    let started = Instant::now();
    let e = f
        .run(
            "shell",
            json!({ "command": shell_cmd("Start-Sleep 999", "sleep 999") }),
        )
        .await;
    assert!(
        e.output.content.contains("cancelled"),
        "{}",
        e.output.content
    );
    assert!(started.elapsed() < Duration::from_secs(10));
}
