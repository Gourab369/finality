mod cli;
mod redact;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use finality_agents::{Core, CoreDeps};
use finality_core::Workspace;
use finality_core::config::{ConfigLoader, ENV_PREFIX};
use finality_core::events::{CoreCommand, SlashCommand};
use finality_llm::OllamaProvider;
use finality_store::DataDir;
use finality_store::paths::HOME_ENV;
use finality_tui::App;
use tokio::sync::mpsc;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::EnvFilter;

use crate::cli::Cli;

const SHUTDOWN_WAIT: Duration = Duration::from_secs(15);

fn main() -> Result<()> {
    // The sandbox launcher must restrict itself before any other thread exists, so it runs before the
    // async runtime and argument parsing.
    let mut args = std::env::args_os().skip(1);
    if args
        .next()
        .is_some_and(|a| a == finality_sandbox::LAUNCHER_ARG)
    {
        std::process::exit(finality_sandbox::launcher_main(args.collect()));
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("cannot start the async runtime")?
        .block_on(run())
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let workspace = Workspace::from_current_dir().context("cannot resolve working directory")?;

    let env_home = std::env::var(HOME_ENV).ok();
    let bootstrap_dir = DataDir::resolve(env_home.as_deref(), None)?;
    let config_path = cli
        .config
        .clone()
        .unwrap_or_else(|| bootstrap_dir.config_file());
    let env_vars = std::env::vars().filter(|(k, _)| k.starts_with(ENV_PREFIX));
    let mut config = ConfigLoader::new()
        .file(&config_path)
        .env_vars(env_vars)
        .load()?;
    cli.apply_to(&mut config);
    config.validate()?;

    let data_dir = DataDir::resolve(env_home.as_deref(), config.general.data_dir.as_deref())?;
    data_dir
        .ensure()
        .with_context(|| format!("cannot create data directory {}", data_dir.root().display()))?;

    let _log_guard = init_tracing(&data_dir)?;
    install_panic_hook();
    tracing::info!(workdir = %workspace.workdir().display(), data_dir = %data_dir.root().display(), "starting");

    let provider = Arc::new(OllamaProvider::new(&config.provider.ollama, &config.llm)?);
    let attachments_dir = data_dir.attachments_dir();
    let deps = CoreDeps {
        config: config.clone(),
        workspace,
        data_dir,
        provider,
        memory_provider: None,
        extra_tools: vec![],
        env_vars: std::env::vars().collect(),
        api_keys: None,
    };
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let mut core = Core::new(deps, ui_tx)?;
    // A new session starts by picking a model, unless one was named on the command line.
    if cli.model.is_none() && cli.resume.is_none() {
        core.choose_model_at_start();
    }
    let mut app = App::new(core.status().clone());
    app.set_attachments_dir(attachments_dir);
    if cli.direct {
        cmd_tx.send(CoreCommand::Slash(SlashCommand::Direct(Some(true))))?;
    }
    if let Some(id) = cli.resume {
        cmd_tx.send(CoreCommand::Slash(SlashCommand::Resume {
            id: Some(id.to_string()),
        }))?;
    }
    if let Some(prompt) = cli.initial_prompt() {
        cmd_tx.send(app.prompt(&prompt))?;
    }
    let core = tokio::spawn(core.run(cmd_rx));
    let ui_result = finality_tui::run(app, ui_rx, cmd_tx).await;
    // The UI has dropped its command sender; the core closes the session and exits.
    if tokio::time::timeout(SHUTDOWN_WAIT, core).await.is_err() {
        tracing::warn!("core did not shut down in time");
    }
    ui_result?;
    Ok(())
}

fn init_tracing(data_dir: &DataDir) -> Result<WorkerGuard> {
    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix("finality")
        .filename_suffix("log")
        .max_log_files(7)
        .build(data_dir.logs_dir())
        .context("cannot open log file")?;
    let (writer, guard) = tracing_appender::non_blocking(appender);
    let filter = EnvFilter::try_from_env("FINALITY_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_writer(redact::Redacting(writer))
        .with_ansi(false)
        .with_env_filter(filter)
        .init();
    Ok(guard)
}

/// Restores the terminal and logs panics; the message printed to stderr is redacted like the log.
fn install_panic_hook() {
    std::panic::set_hook(Box::new(move |info| {
        finality_tui::restore_terminal();
        tracing::error!(panic = %info, "panic");
        eprintln!("{}", finality_core::secrets::redact(&info.to_string()));
    }));
}
