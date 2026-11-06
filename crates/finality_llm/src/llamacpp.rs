//! llama.cpp: starts `llama-server` with a `.gguf` file from `provider.llamacpp.models_dir` (or uses a server
//! already running at `base_url`) and talks to its OpenAI-compatible API. One model is loaded at a time.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use finality_core::config::{LlamaCppConfig, LlmConfig, OpenAiConfig};
use process_wrap::tokio::{ChildWrapper, CommandWrap, KillOnDrop};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::{ChatRequest, ChatStream, LlmError, LlmProvider, ModelInfo, OpenAiProvider};

const LOG_LINES: usize = 40;
const ERROR_LINES: usize = 12;
const SCAN_DEPTH: usize = 3;
const POLL: Duration = Duration::from_millis(250);
const STOP_GRACE: Duration = Duration::from_secs(5);

struct Running {
    model: String,
    child: Box<dyn ChildWrapper>,
    api: Arc<OpenAiProvider>,
}

pub struct LlamaCppProvider {
    cfg: LlamaCppConfig,
    llm: LlmConfig,
    /// The server at `base_url`, when one is configured.
    external: Option<Arc<OpenAiProvider>>,
    server: Mutex<Option<Running>>,
    /// Last lines the server printed, for error messages.
    log: Arc<StdMutex<VecDeque<String>>>,
    http: reqwest::Client,
}

fn api(base_url: String, context_window: u32, llm: &LlmConfig) -> Result<OpenAiProvider, LlmError> {
    let cfg = OpenAiConfig {
        base_url,
        api_key_env: String::new(),
        model: String::new(),
        context_window,
    };
    Ok(OpenAiProvider::new(&cfg, llm, None)?.with_template_thinking())
}

/// `http://host:8080` and `http://host:8080/v1/` both become `http://host:8080/v1`.
fn v1_url(base: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    if base.ends_with("/v1") {
        base.to_owned()
    } else {
        format!("{base}/v1")
    }
}

/// Parts after the first of a split model (`name-00002-of-00003.gguf`) load with the first one.
fn is_later_shard(lower: &str) -> bool {
    let Some(stem) = lower.strip_suffix(".gguf") else {
        return false;
    };
    let Some((left, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, part)) = left.rsplit_once('-') else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    digits(total) && digits(part) && part.parse::<u32>().is_ok_and(|n| n != 1)
}

/// A loadable model file: `.gguf`, not a vision projector (`mmproj`) and not a later shard.
pub fn is_model_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf") && !lower.contains("mmproj") && !is_later_shard(&lower)
}

/// Model files under `root`, as `/`-separated paths relative to it, sorted.
pub fn find_models(root: &Path) -> std::io::Result<Vec<String>> {
    fn walk(dir: &Path, rel: &str, depth: usize, out: &mut Vec<String>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = rel_join(rel, &name);
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() && depth > 1 && !name.starts_with('.') {
                let _ = walk(&entry.path(), &path, depth - 1, out);
            } else if !kind.is_dir() && is_model_file(&name) {
                out.push(path);
            }
        }
        Ok(())
    }
    fn rel_join(rel: &str, name: &str) -> String {
        if rel.is_empty() {
            name.to_owned()
        } else {
            format!("{rel}/{name}")
        }
    }
    let mut out = Vec::new();
    walk(root, "", SCAN_DEPTH, &mut out)?;
    out.sort_by_key(|p| p.to_lowercase());
    Ok(out)
}

fn free_port() -> Result<u16, LlmError> {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .map_err(|e| LlmError::Other(format!("cannot find a free port for llama-server: {e}")))
}

async fn pump(stream: impl AsyncRead + Unpin, log: Arc<StdMutex<VecDeque<String>>>) {
    let mut lines = BufReader::new(stream).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::debug!(target: "llama_server", "{line}");
        if let Ok(mut l) = log.lock() {
            if l.len() == LOG_LINES {
                l.pop_front();
            }
            l.push_back(line);
        }
    }
}

async fn stop(child: &mut Box<dyn ChildWrapper>) {
    if let Err(e) = child.start_kill() {
        tracing::warn!(error = %e, "cannot stop llama-server");
    }
    let _ = tokio::time::timeout(STOP_GRACE, child.wait()).await;
}

impl LlamaCppProvider {
    pub fn new(cfg: &LlamaCppConfig, llm: &LlmConfig) -> Result<Self, LlmError> {
        let external = if cfg.base_url.trim().is_empty() {
            None
        } else {
            Some(Arc::new(api(
                v1_url(&cfg.base_url),
                cfg.context_window,
                llm,
            )?))
        };
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| LlmError::Other(format!("http client: {e}")))?;
        Ok(Self {
            cfg: cfg.clone(),
            llm: llm.clone(),
            external,
            server: Mutex::new(None),
            log: Arc::new(StdMutex::new(VecDeque::new())),
            http,
        })
    }

    /// `model` relative to `models_dir`, or absolute.
    fn model_path(&self, model: &str) -> Result<PathBuf, LlmError> {
        let given = Path::new(model.trim());
        let path = if given.is_absolute() {
            given.to_path_buf()
        } else if self.cfg.models_dir.trim().is_empty() {
            return Err(LlmError::Other(
                "set provider.llamacpp.models_dir to the folder with your .gguf files, or give an absolute path".into(),
            ));
        } else {
            Path::new(self.cfg.models_dir.trim()).join(given)
        };
        if !path.is_file() {
            return Err(LlmError::Other(format!(
                "model file {} not found",
                path.display()
            )));
        }
        Ok(path)
    }

    fn tail(&self) -> String {
        let lines: Vec<String> = self
            .log
            .lock()
            .map(|l| l.iter().rev().take(ERROR_LINES).rev().cloned().collect())
            .unwrap_or_default();
        if lines.is_empty() {
            String::new()
        } else {
            format!("\nllama-server output:\n{}", lines.join("\n"))
        }
    }

    /// The API serving `model`, starting or switching the server when needed.
    async fn ensure(
        &self,
        model: &str,
        cancel: &CancellationToken,
    ) -> Result<Arc<OpenAiProvider>, LlmError> {
        if let Some(api) = &self.external {
            return Ok(api.clone());
        }
        let mut server = tokio::select! {
            s = self.server.lock() => s,
            () = cancel.cancelled() => return Err(LlmError::Cancelled),
        };
        if let Some(r) = server.as_mut()
            && r.model == model
            && matches!(r.child.try_wait(), Ok(None))
        {
            return Ok(r.api.clone());
        }
        if let Some(mut old) = server.take() {
            tracing::info!(from = %old.model, to = %model, "switching llama-server model");
            stop(&mut old.child).await;
        }
        // Cancelling drops a half-started server, which kills it.
        let running = tokio::select! {
            r = self.start(model) => r?,
            () = cancel.cancelled() => return Err(LlmError::Cancelled),
        };
        let api = running.api.clone();
        *server = Some(running);
        Ok(api)
    }

    async fn start(&self, model: &str) -> Result<Running, LlmError> {
        let path = self.model_path(model)?;
        let port = match self.cfg.port {
            0 => free_port()?,
            p => p,
        };
        let mut args: Vec<String> = vec![
            "--model".into(),
            path.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "--ctx-size".into(),
            self.cfg.context_window.to_string(),
            "--parallel".into(),
            "1".into(),
            // Tool calling needs the model's chat template.
            "--jinja".into(),
        ];
        if let Some(n) = self.cfg.gpu_layers {
            args.extend(["--n-gpu-layers".into(), n.to_string()]);
        }
        args.extend(self.cfg.extra_args.iter().cloned());
        let program = self.cfg.server_path.trim().to_owned();
        let mut cmd = CommandWrap::with_new(&program, move |c| {
            c.args(&args)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
        });
        #[cfg(unix)]
        cmd.wrap(process_wrap::tokio::ProcessGroup::leader());
        // With KillOnDrop the job object also kills the server if Finality itself dies.
        #[cfg(windows)]
        cmd.wrap(process_wrap::tokio::JobObject);
        cmd.wrap(KillOnDrop);
        let mut child = cmd.spawn().map_err(|e| {
            LlmError::Other(if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "cannot start `{program}`: not found. Install llama.cpp and put llama-server on PATH, \
                     or set provider.llamacpp.server_path"
                )
            } else {
                format!("cannot start `{program}`: {e}")
            })
        })?;
        if let Ok(mut l) = self.log.lock() {
            l.clear();
        }
        if let Some(out) = child.stdout().take() {
            tokio::spawn(pump(out, self.log.clone()));
        }
        if let Some(err) = child.stderr().take() {
            tokio::spawn(pump(err, self.log.clone()));
        }
        tracing::info!(model = %path.display(), port, "starting llama-server");
        self.wait_ready(&mut child, port, model).await?;
        let api = Arc::new(api(
            format!("http://127.0.0.1:{port}/v1"),
            self.cfg.context_window,
            &self.llm,
        )?);
        Ok(Running {
            model: model.to_owned(),
            child,
            api,
        })
    }

    /// Polls `/health` until the model is loaded (it answers 503 while loading).
    async fn wait_ready(
        &self,
        child: &mut Box<dyn ChildWrapper>,
        port: u16,
        model: &str,
    ) -> Result<(), LlmError> {
        let limit = self.cfg.startup_timeout_secs.max(1);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(limit);
        let url = format!("http://127.0.0.1:{port}/health");
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                return Err(LlmError::Other(format!(
                    "llama-server exited ({status}) while loading `{model}`{}",
                    self.tail()
                )));
            }
            if let Ok(resp) = self
                .http
                .get(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                && resp.status().is_success()
            {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(LlmError::Other(format!(
                    "llama-server did not finish loading `{model}` within {limit}s \
                     (provider.llamacpp.startup_timeout_secs){}",
                    self.tail()
                )));
            }
            tokio::time::sleep(POLL).await;
        }
    }
}

#[async_trait]
impl LlmProvider for LlamaCppProvider {
    fn id(&self) -> &'static str {
        "llamacpp"
    }

    /// `.gguf` files under `models_dir`, or the models a configured server reports.
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        if let Some(api) = &self.external {
            return api.list_models().await;
        }
        let dir = self.cfg.models_dir.trim().to_owned();
        if dir.is_empty() {
            return Err(LlmError::Other(
                "set provider.llamacpp.models_dir to the folder with your .gguf files".into(),
            ));
        }
        let root = PathBuf::from(&dir);
        tokio::task::spawn_blocking(move || find_models(&root))
            .await
            .map_err(|e| LlmError::Other(e.to_string()))?
            .map_err(|e| LlmError::Other(format!("cannot read models folder {dir}: {e}")))
    }

    /// Loads the model, so a bad file or a missing server shows up when it is chosen.
    async fn model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        let api = self.ensure(model, &CancellationToken::new()).await?;
        let mut info = api.model_info(model).await?;
        // Whether the model reasons depends on its chat template, which only the server knows.
        info.supports_thinking = true;
        Ok(info)
    }

    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let api = self.ensure(&req.model, &cancel).await?;
        api.chat(req, cancel).await
    }

    async fn shutdown(&self) {
        if let Some(mut r) = self.server.lock().await.take() {
            tracing::info!(model = %r.model, "stopping llama-server");
            stop(&mut r.child).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_loadable_gguf_files() {
        let tmp = std::env::temp_dir().join(format!("finality_llamacpp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        for f in [
            "Qwen3-14B-Q4_K_M.gguf",
            "notes.txt",
            "gemma/gemma-3-12b-it-Q4_0.gguf",
            "gemma/mmproj-gemma-3-12b-f16.gguf",
            "big/model-00001-of-00002.gguf",
            "big/model-00002-of-00002.gguf",
            "a/b/c/too-deep.gguf",
        ] {
            let p = tmp.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"").unwrap();
        }
        let found = find_models(&tmp).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(
            found,
            [
                "big/model-00001-of-00002.gguf",
                "gemma/gemma-3-12b-it-Q4_0.gguf",
                "Qwen3-14B-Q4_K_M.gguf"
            ]
        );
    }

    #[test]
    fn base_urls_get_v1() {
        assert_eq!(v1_url("http://127.0.0.1:8080"), "http://127.0.0.1:8080/v1");
        assert_eq!(
            v1_url("http://127.0.0.1:8080/v1/"),
            "http://127.0.0.1:8080/v1"
        );
    }

    #[test]
    fn missing_folder_or_file_is_explained() {
        let llm = LlmConfig::default();
        let p = LlamaCppProvider::new(&LlamaCppConfig::default(), &llm).unwrap();
        let err = p.model_path("x.gguf").unwrap_err().to_string();
        assert!(err.contains("models_dir"), "{err}");
        let cfg = LlamaCppConfig {
            models_dir: std::env::temp_dir().display().to_string(),
            ..LlamaCppConfig::default()
        };
        let p = LlamaCppProvider::new(&cfg, &llm).unwrap();
        let err = p.model_path("no-such-model.gguf").unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
    }

    #[tokio::test]
    async fn missing_server_binary_is_explained() {
        let dir =
            std::env::temp_dir().join(format!("finality_llamacpp_bin_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.gguf"), b"").unwrap();
        let cfg = LlamaCppConfig {
            models_dir: dir.display().to_string(),
            server_path: "finality-no-such-llama-server".into(),
            ..LlamaCppConfig::default()
        };
        let p = LlamaCppProvider::new(&cfg, &LlmConfig::default()).unwrap();
        let err = p.model_info("m.gguf").await.unwrap_err().to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(err.contains("server_path"), "{err}");
    }
}
