//! Minimal stdio MCP server used by the integration tests (not shipped).
//! Tools: `echo {text}`, `add {a, b}`, `fail {}`. Appends "started" to `$H_AGENT_MCP_ECHO_MARKER` on start.

use std::io::{BufRead, Write};

use serde_json::{Value, json};

fn main() {
    if let Ok(marker) = std::env::var("H_AGENT_MCP_ECHO_MARKER")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(marker)
    {
        let _ = writeln!(f, "started");
    }
    eprintln!("echo server ready");
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(id) = msg.get("id").cloned() else {
            continue;
        };
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let reply = match msg.get("method").and_then(Value::as_str).unwrap_or("") {
            "initialize" => json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!("2025-06-18")),
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "echo", "version": "1.0.0" }
                }
            }),
            "ping" => json!({ "jsonrpc": "2.0", "id": id, "result": {} }),
            "tools/list" => json!({
                "jsonrpc": "2.0", "id": id,
                "result": { "tools": [
                    {
                        "name": "echo",
                        "description": "Echo the given text back, for testing round trips",
                        "inputSchema": { "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] }
                    },
                    {
                        "name": "add",
                        "description": "Add two numbers",
                        "inputSchema": { "type": "object", "properties": { "a": { "type": "number" }, "b": { "type": "number" } } }
                    },
                    {
                        "name": "fail",
                        "description": "Always fails",
                        "inputSchema": { "type": "object" }
                    }
                ] }
            }),
            "tools/call" => {
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let (text, is_error) = match params.get("name").and_then(Value::as_str) {
                    Some("echo") => (
                        format!(
                            "echo: {}",
                            args.get("text").and_then(Value::as_str).unwrap_or("")
                        ),
                        false,
                    ),
                    Some("add") => {
                        let n = |k: &str| args.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                        (format!("{}", n("a") + n("b")), false)
                    }
                    _ => ("this tool always fails".to_owned(), true),
                };
                json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": { "content": [ { "type": "text", "text": text } ], "isError": is_error }
                })
            }
            other => json!({
                "jsonrpc": "2.0", "id": id,
                "error": { "code": -32601, "message": format!("method not found: {other}") }
            }),
        };
        if writeln!(stdout, "{reply}")
            .and_then(|()| stdout.flush())
            .is_err()
        {
            break;
        }
    }
}
