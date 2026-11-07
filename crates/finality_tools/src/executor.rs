//! The only path from a model's tool call to execution: preview → authorize → call (DESIGN §9.1, §12).

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use finality_core::{AgentId, ApprovalDecision, RiskClass};
use finality_safety::{ActionRequest, Decision, Location, PermissionEngine};
use serde_json::Value;

use crate::registry::ToolRegistry;
use crate::tool::{ToolCtx, ToolError, ToolOutput, truncate_middle};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    pub agent: AgentId,
    pub tool: String,
    pub risk: RiskClass,
    pub location: Location,
    pub summary: String,
    pub detail: Option<String>,
    pub reasons: Vec<String>,
    pub allow_session: bool,
    pub default_deny: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalResponse {
    AllowOnce,
    AllowSession,
    Deny { note: Option<String> },
}

/// Asks the user. Implemented by the TUI; tests use scripted approvers.
#[async_trait]
pub trait Approver: Send + Sync {
    async fn request(&self, req: ApprovalRequest) -> ApprovalResponse;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Execution {
    pub output: ToolOutput,
    /// `None` when the call failed before authorization (unknown tool, bad arguments).
    pub approval: Option<ApprovalDecision>,
    pub reasons: Vec<String>,
    pub note: Option<String>,
}

impl Execution {
    fn rejected(output: ToolOutput) -> Self {
        Self {
            output,
            approval: None,
            reasons: Vec::new(),
            note: None,
        }
    }
}

pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    permissions: Arc<Mutex<PermissionEngine>>,
    approver: Arc<dyn Approver>,
}

impl ToolExecutor {
    pub fn new(
        registry: Arc<ToolRegistry>,
        permissions: Arc<Mutex<PermissionEngine>>,
        approver: Arc<dyn Approver>,
    ) -> Self {
        Self {
            registry,
            permissions,
            approver,
        }
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// `allowed` restricts sub-agents to the Orchestrator's allowlist (DESIGN §12.4).
    pub async fn execute(
        &self,
        name: &str,
        args: Value,
        ctx: &ToolCtx,
        allowed: Option<&HashSet<String>>,
    ) -> Execution {
        let Some(tool) = self.registry.get(name) else {
            return Execution::rejected(ToolOutput::error(format!("unknown tool `{name}`")));
        };
        if allowed.is_some_and(|a| !a.contains(name)) {
            return Execution::rejected(ToolOutput::error(format!(
                "ToolNotAllowed: `{name}` is not permitted for this agent"
            )));
        }
        let preview = match tool.preview(&args, ctx).await {
            Ok(p) => p,
            Err(e) => return Execution::rejected(ToolOutput::error(format!("{name}: {e}"))),
        };

        let request = ActionRequest {
            tool: name.to_owned(),
            risk: tool.risk(),
            location: preview.location,
            reasons: preview.reasons.clone(),
            command: preview.command.clone(),
            shapes: preview.shapes.clone(),
            always_ask: preview.always_ask.clone(),
            deletes: preview.deletes,
            grant_keys: preview.grant_keys.clone(),
        };
        let decision = match self.permissions.lock() {
            Ok(engine) => engine.decide(&request),
            Err(_) => Decision::Deny {
                reason: "permission engine unavailable".into(),
            },
        };

        let mut reasons = preview.reasons.clone();
        let mut note = None;
        let approval = match decision {
            Decision::Allow => ApprovalDecision::AutoAllowed,
            Decision::Deny { reason } => {
                reasons.push(reason.clone());
                return Execution {
                    output: ToolOutput::error(format!("denied by policy: {reason}")),
                    approval: Some(ApprovalDecision::PolicyDenied),
                    reasons,
                    note,
                };
            }
            Decision::Ask(info) => {
                reasons.clone_from(&info.reasons);
                let ask = ApprovalRequest {
                    agent: ctx.agent,
                    tool: name.to_owned(),
                    risk: tool.risk(),
                    location: preview.location,
                    summary: preview.summary.clone(),
                    detail: preview.detail.clone(),
                    reasons: info.reasons,
                    allow_session: info.allow_session,
                    default_deny: info.default_deny,
                };
                let response = tokio::select! {
                    biased;
                    () = ctx.cancel.cancelled() => ApprovalResponse::Deny { note: Some("cancelled".into()) },
                    r = self.approver.request(ask) => r,
                };
                match response {
                    ApprovalResponse::AllowOnce => ApprovalDecision::AllowedOnce,
                    ApprovalResponse::AllowSession if info.allow_session => {
                        let granted = self
                            .permissions
                            .lock()
                            .is_ok_and(|mut e| e.grant_session(&request));
                        if granted {
                            ApprovalDecision::AllowedSession
                        } else {
                            ApprovalDecision::AllowedOnce
                        }
                    }
                    ApprovalResponse::AllowSession => ApprovalDecision::AllowedOnce,
                    ApprovalResponse::Deny { note: n } => {
                        let text = match &n {
                            Some(n) => format!("The user denied this {name} call: {n}"),
                            None => format!("The user denied this {name} call."),
                        };
                        note = n;
                        return Execution {
                            output: ToolOutput::error(text),
                            approval: Some(ApprovalDecision::Denied),
                            reasons,
                            note,
                        };
                    }
                }
            }
        };

        let mut output = tokio::select! {
            biased;
            () = ctx.cancel.cancelled() => ToolOutput::error("cancelled"),
            r = tool.call(args, ctx, &preview) => match r {
                Ok(out) => out,
                Err(ToolError::Cancelled) => ToolOutput::error("cancelled"),
                Err(e) => ToolOutput::error(format!("{name}: {e}")),
            },
        };
        let (content, cut) = truncate_middle(&output.content, ctx.limits.max_output_chars);
        output.content = content;
        output.truncated |= cut;
        Execution {
            output,
            approval: Some(approval),
            reasons,
            note,
        }
    }
}
