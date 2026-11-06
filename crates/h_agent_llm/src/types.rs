use h_agent_core::{ProviderKind, ThinkLevel};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        thinking: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        call_id: String,
        name: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn tool(
        call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Message::Tool {
            call_id: call_id.into(),
            name: name.into(),
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        match self {
            Message::System { content }
            | Message::User { content }
            | Message::Assistant { content, .. }
            | Message::Tool { content, .. } => content,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChatRequest {
    /// Which provider serves the request when it goes through a `ProviderRouter`; `None` is the router's default.
    pub provider: Option<ProviderKind>,
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSchema>,
    pub think: ThinkLevel,
    /// JSON Schema the response must conform to.
    pub response_format: Option<Value>,
    /// Overrides the provider's configured context window.
    pub num_ctx: Option<u32>,
    pub temperature: Option<f32>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum StopReason {
    #[default]
    Stop,
    Length,
    ToolUse,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatChunk {
    Text(String),
    Thinking(String),
    ToolCall(ToolCall),
    Done {
        usage: Usage,
        stop_reason: StopReason,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelInfo {
    pub name: String,
    pub context_length: Option<u32>,
    pub supports_tools: bool,
    pub supports_thinking: bool,
    pub supports_vision: bool,
}
