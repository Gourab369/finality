//! OAuth tokens for HTTP MCP servers: OS keyring storage and the `/mcp login` PKCE flow (DESIGN §9.4).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rmcp::transport::auth::{
    AuthError, AuthorizationManager, AuthorizationRequest, CredentialStore, OAuthState,
    StoredCredentials,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

const KEYRING_SERVICE: &str = "h_agent";
/// Windows Credential Manager caps a secret at 2560 bytes; OAuth tokens can be longer.
const CHUNK_CHARS: usize = 1000;
pub const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

/// Where serialized OAuth credentials live, per server name.
pub trait TokenVault: Send + Sync {
    fn load(&self, server: &str) -> Result<Option<String>, String>;
    fn save(&self, server: &str, value: &str) -> Result<(), String>;
    fn clear(&self, server: &str) -> Result<(), String>;
}

/// The OS keyring, split over several entries when the value is long.
pub struct KeyringVault;

fn entry(server: &str, part: usize) -> Result<keyring::Entry, String> {
    let user = if part == 0 {
        format!("mcp-oauth:{server}")
    } else {
        format!("mcp-oauth:{server}#{part}")
    };
    keyring::Entry::new(KEYRING_SERVICE, &user).map_err(|e| format!("keyring: {e}"))
}

fn read(server: &str, part: usize) -> Result<Option<String>, String> {
    match entry(server, part)?.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(format!("keyring: {e}")),
    }
}

impl KeyringVault {
    fn parts(server: &str) -> Result<usize, String> {
        Ok(read(server, 0)?
            .and_then(|head| head.split_once('\n').and_then(|(n, _)| n.parse().ok()))
            .unwrap_or(0))
    }
}

impl TokenVault for KeyringVault {
    fn load(&self, server: &str) -> Result<Option<String>, String> {
        let Some(head) = read(server, 0)? else {
            return Ok(None);
        };
        let Some((n, first)) = head.split_once('\n') else {
            return Ok(None);
        };
        let n: usize = n.parse().map_err(|_| "keyring: unreadable token entry")?;
        let mut value = first.to_owned();
        for part in 1..n {
            match read(server, part)? {
                Some(chunk) => value.push_str(&chunk),
                None => return Ok(None),
            }
        }
        Ok(Some(value))
    }

    fn save(&self, server: &str, value: &str) -> Result<(), String> {
        let chars: Vec<char> = value.chars().collect();
        let chunks: Vec<String> = chars
            .chunks(CHUNK_CHARS)
            .map(|c| c.iter().collect())
            .collect();
        let old = Self::parts(server)?;
        for (part, chunk) in chunks.iter().enumerate().skip(1) {
            entry(server, part)?
                .set_password(chunk)
                .map_err(|e| format!("keyring: {e}"))?;
        }
        let first = chunks.first().map_or("", String::as_str);
        entry(server, 0)?
            .set_password(&format!("{}\n{first}", chunks.len().max(1)))
            .map_err(|e| format!("keyring: {e}"))?;
        for part in chunks.len().max(1)..old {
            let _ = entry(server, part)?.delete_credential();
        }
        Ok(())
    }

    fn clear(&self, server: &str) -> Result<(), String> {
        let n = Self::parts(server)?.max(1);
        for part in 0..n {
            match entry(server, part)?.delete_credential() {
                Ok(()) | Err(keyring::Error::NoEntry) => {}
                Err(e) => return Err(format!("keyring: {e}")),
            }
        }
        Ok(())
    }
}

/// In-memory vault for tests.
#[derive(Default)]
pub struct MemoryVault(Mutex<HashMap<String, String>>);

impl TokenVault for MemoryVault {
    fn load(&self, server: &str) -> Result<Option<String>, String> {
        Ok(self
            .0
            .lock()
            .map_err(|e| e.to_string())?
            .get(server)
            .cloned())
    }

    fn save(&self, server: &str, value: &str) -> Result<(), String> {
        self.0
            .lock()
            .map_err(|e| e.to_string())?
            .insert(server.to_owned(), value.to_owned());
        Ok(())
    }

    fn clear(&self, server: &str) -> Result<(), String> {
        self.0.lock().map_err(|e| e.to_string())?.remove(server);
        Ok(())
    }
}

/// rmcp's credential store for one server, backed by a vault.
struct VaultStore {
    vault: Arc<dyn TokenVault>,
    server: String,
}

fn store_error(e: String) -> AuthError {
    AuthError::CredentialStoreError(e)
}

#[async_trait]
impl CredentialStore for VaultStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        match self.vault.load(&self.server).map_err(store_error)? {
            Some(text) => serde_json::from_str(&text)
                .map(Some)
                .map_err(|e| store_error(e.to_string())),
            None => Ok(None),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let text = serde_json::to_string(&credentials).map_err(|e| store_error(e.to_string()))?;
        self.vault.save(&self.server, &text).map_err(store_error)
    }

    async fn clear(&self) -> Result<(), AuthError> {
        self.vault.clear(&self.server).map_err(store_error)
    }
}

async fn manager(
    url: &str,
    server: &str,
    vault: &Arc<dyn TokenVault>,
) -> Result<AuthorizationManager, String> {
    let mut m = AuthorizationManager::new(url)
        .await
        .map_err(|e| format!("OAuth setup for {url} failed: {e}"))?;
    m.set_credential_store(VaultStore {
        vault: vault.clone(),
        server: server.to_owned(),
    });
    Ok(m)
}

/// A current access token from stored credentials (refreshed when expired), if logged in.
pub async fn access_token(
    url: &str,
    server: &str,
    vault: &Arc<dyn TokenVault>,
) -> Result<Option<String>, String> {
    if vault.load(server)?.is_none() {
        return Ok(None);
    }
    let mut m = manager(url, server, vault).await?;
    if !m
        .initialize_from_store()
        .await
        .map_err(|e| format!("stored login for `{server}` is unusable: {e}"))?
    {
        return Ok(None);
    }
    m.get_access_token().await.map(Some).map_err(|e| {
        format!("the login for `{server}` expired or was revoked ({e}); run /mcp login {server}")
    })
}

/// OAuth 2.1 authorization code flow with PKCE and a loopback redirect. `show_url` gets the
/// authorization URL to open; tokens are saved to the vault.
pub async fn login(
    url: &str,
    server: &str,
    vault: &Arc<dyn TokenVault>,
    show_url: impl FnOnce(&str),
    cancel: &CancellationToken,
) -> Result<(), String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("cannot open a local port for the login redirect: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let redirect = format!("http://127.0.0.1:{port}/callback");
    let mut state = OAuthState::Unauthorized(manager(url, server, vault).await?);
    state
        .start_authorization(
            AuthorizationRequest::new(&redirect).with_client_name(h_agent_core::APP_NAME),
        )
        .await
        .map_err(|e| format!("cannot start the login for `{server}`: {e}"))?;
    let auth_url = state
        .get_authorization_url()
        .await
        .map_err(|e| e.to_string())?;
    show_url(&auth_url);
    let path = tokio::select! {
        r = tokio::time::timeout(LOGIN_TIMEOUT, wait_for_callback(&listener)) => {
            r.map_err(|_| "the login was not completed within 5 minutes".to_owned())??
        }
        () = cancel.cancelled() => return Err("login cancelled".into()),
    };
    state
        .handle_callback_url(&format!("http://127.0.0.1:{port}{path}"))
        .await
        .map_err(|e| format!("login for `{server}` failed: {e}"))
}

/// Serves the loopback redirect and returns its path and query.
pub async fn wait_for_callback(listener: &TcpListener) -> Result<String, String> {
    loop {
        let (mut sock, _) = listener.accept().await.map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; 8192];
        let mut len = 0;
        while len < buf.len() {
            let n = sock
                .read(&mut buf[len..])
                .await
                .map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            len += n;
            if buf[..len].windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let head = String::from_utf8_lossy(&buf[..len]);
        let target = head
            .lines()
            .next()
            .and_then(|l| l.strip_prefix("GET "))
            .and_then(|l| l.split(' ').next())
            .unwrap_or("");
        if !target.starts_with("/callback?") {
            let _ = sock
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await;
            continue;
        }
        let (status, body) = if target.contains("code=") {
            (
                "200 OK",
                "Login complete. You can close this tab and return to Finality.",
            )
        } else {
            (
                "400 Bad Request",
                "Login failed: the authorization server returned no code.",
            )
        };
        let page = format!(
            "<!doctype html><title>{}</title><p>{body}</p>",
            h_agent_core::APP_NAME
        );
        let _ = sock
            .write_all(
                format!(
                    "HTTP/1.1 {status}\r\ncontent-type: text/html; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{page}",
                    page.len()
                )
                .as_bytes(),
            )
            .await;
        if target.contains("code=") {
            return Ok(target.to_owned());
        }
        let error = url::Url::parse(&format!("http://localhost{target}"))
            .ok()
            .and_then(|u| {
                u.query_pairs()
                    .find(|(k, _)| k == "error_description" || k == "error")
                    .map(|(_, v)| v.into_owned())
            })
            .unwrap_or_else(|| "no authorization code".into());
        return Err(format!("authorization was refused: {error}"));
    }
}

/// Opens `url` in the default browser; failures are ignored (the URL is also shown).
pub fn open_browser(url: &str) {
    #[cfg(windows)]
    let result = std::process::Command::new("rundll32.exe")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn();
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let result = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = result {
        tracing::warn!("cannot open a browser: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn callback_listener_returns_the_code() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = tokio::spawn(async move {
            let get = |path: &'static str| async move {
                let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
                s.write_all(format!("GET {path} HTTP/1.1\r\nhost: x\r\n\r\n").as_bytes())
                    .await
                    .unwrap();
                let mut out = String::new();
                s.read_to_string(&mut out).await.unwrap();
                out
            };
            let missing = get("/favicon.ico").await;
            let done = get("/callback?code=abc&state=xyz").await;
            (missing, done)
        });
        let path = wait_for_callback(&listener).await.unwrap();
        assert_eq!(path, "/callback?code=abc&state=xyz");
        let (missing, done) = client.await.unwrap();
        assert!(missing.starts_with("HTTP/1.1 404"));
        assert!(done.starts_with("HTTP/1.1 200") && done.contains("Login complete"));
    }

    #[tokio::test]
    async fn refused_authorization_is_reported() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
            s.write_all(b"GET /callback?error=access_denied&state=x HTTP/1.1\r\n\r\n")
                .await
                .unwrap();
            let mut out = String::new();
            let _ = s.read_to_string(&mut out).await;
        });
        let err = wait_for_callback(&listener).await.unwrap_err();
        assert!(err.contains("access_denied"), "{err}");
    }

    #[tokio::test]
    async fn vault_store_round_trips_credentials() {
        let vault: Arc<dyn TokenVault> = Arc::new(MemoryVault::default());
        let store = VaultStore {
            vault: vault.clone(),
            server: "docs".into(),
        };
        assert!(store.load().await.unwrap().is_none());
        store
            .save(StoredCredentials::new(
                "client-1".into(),
                None,
                vec!["read".into()],
                Some(5),
            ))
            .await
            .unwrap();
        let back = store.load().await.unwrap().unwrap();
        assert_eq!(back.client_id, "client-1");
        assert_eq!(back.granted_scopes, ["read"]);
        store.clear().await.unwrap();
        assert!(vault.load("docs").unwrap().is_none());
    }
}
