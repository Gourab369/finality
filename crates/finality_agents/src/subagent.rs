//! Runs one attempt of a task in a Worker or Research sub-agent (DESIGN §6.4).

use std::collections::{BTreeSet, HashSet};
use std::fmt::Write as _;
use std::pin::pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use finality_core::events::UsageSnapshot;
use finality_core::{AgentId, Mode, RiskClass, Role, TaskId};
use finality_llm::Message;
use finality_memory::agents::CustomAgent;
use finality_memory::skills::Skill;
use finality_store::Record;
use finality_tools::{ActiveToolSet, TOOL_SEARCH, Tier, ToolCtx, ToolLimits, ToolSpec};
use tokio_util::sync::CancellationToken;

use crate::conversation::{Conversation, fence};
use crate::engine::{Engine, EngineSettings, TurnEnd, TurnSink};
use crate::heartbeat::{Budget, Phase};
use crate::memory::{MemoryJob, SessionCtx};
use crate::messages::ToSubAgent;
use crate::plan::{Outcome, StepSpec};
use crate::supervisor::{AttemptEnd, Supervisor};

pub const SUBAGENT_TOOLS: &[&str] = &["report_result", "ask_orchestrator"];
const MAX_STEPS: u32 = 25;
const MAX_TOOL_CALLS: u32 = 50;

pub(crate) struct AttemptJob {
    pub number: u32,
    pub attempt: u32,
    pub agent: AgentId,
    pub label: String,
    pub role: Role,
    /// The custom agent named by the step, read when the attempt started.
    pub custom: Option<CustomAgent>,
    pub goal: String,
    pub spec: StepSpec,
    pub task_id: TaskId,
    pub strategy: String,
    /// (attempt, strategy, result) of earlier attempts.
    pub prior: Vec<(u32, String, String)>,
    pub cancel: CancellationToken,
    pub session: Option<SessionCtx>,
}

pub fn role_defaults(role: Role) -> &'static [&'static str] {
    match role {
        Role::Research => &[
            "read_file",
            "search_files",
            "web_search",
            "web_fetch",
            "browser_open",
            "browser_snapshot",
            "browser_click",
            "browser_close",
        ],
        _ => &["read_file", "search_files", "edit_file", "shell"],
    }
}

/// Tools a custom agent may list: the Worker and Research defaults and MCP tools (DESIGN §6.5).
pub fn delegable(name: &str) -> bool {
    name.starts_with("mcp__")
        || role_defaults(Role::Worker).contains(&name)
        || role_defaults(Role::Research).contains(&name)
}

/// Role defaults ∩ the task's allowlist ∩ the skills' tools ∩ what the mode permits, plus the reporting tools.
/// MCP tools (`mcp__*`) have no role default; a sub-agent gets one only when its allowlist names it.
pub fn effective_tools(
    role: Role,
    allowed: Option<&[String]>,
    skill_tools: Option<&[String]>,
    mode: Mode,
    specs: &[ToolSpec],
    risk: impl Fn(&str) -> Option<RiskClass>,
) -> Vec<ToolSpec> {
    agent_tools(role, None, allowed, skill_tools, mode, specs, risk)
}

/// [`effective_tools`] for a custom agent: its delegable `tools`, when set, replace the role defaults,
/// and MCP tools from the task's allowlist must also be in that list.
pub fn agent_tools(
    role: Role,
    custom: Option<&[String]>,
    allowed: Option<&[String]>,
    skill_tools: Option<&[String]>,
    mode: Mode,
    specs: &[ToolSpec],
    risk: impl Fn(&str) -> Option<RiskClass>,
) -> Vec<ToolSpec> {
    let base: Vec<&str> = match custom {
        Some(list) => list
            .iter()
            .map(String::as_str)
            .filter(|n| delegable(n))
            .collect(),
        None => role_defaults(role).to_vec(),
    };
    let mcp = allowed
        .into_iter()
        .flatten()
        .map(String::as_str)
        .filter(|n| n.starts_with("mcp__"))
        .filter(|n| custom.is_none_or(|c| c.iter().any(|x| x == n)));
    let names: HashSet<&str> = base
        .into_iter()
        .filter(|n| allowed.is_none_or(|a| a.iter().any(|x| x == n)))
        .chain(mcp)
        .filter(|n| skill_tools.is_none_or(|a| a.iter().any(|x| x == n)))
        .filter(|n| mode != Mode::Plan || risk(n) == Some(RiskClass::Read))
        .chain(SUBAGENT_TOOLS.iter().copied())
        .collect();
    specs
        .iter()
        .filter(|s| names.contains(s.name.as_str()))
        .cloned()
        .collect()
}

fn system_prompt(
    role: Role,
    custom: Option<&CustomAgent>,
    skills: &[&Skill],
    direct: bool,
    instructions: Option<&str>,
) -> String {
    let template = if role == Role::Research {
        include_str!("../prompts/research.md")
    } else {
        include_str!("../prompts/worker.md")
    };
    let mut prompt = template.to_owned();
    if direct {
        prompt.push('\n');
        prompt.push_str(crate::DIRECT_PROMPT);
    }
    if let Some(text) = instructions {
        prompt.push_str(text);
    }
    if let Some(a) = custom {
        let _ = write!(
            prompt,
            "\nYou are the custom agent \"{}\": {}\nFollow these instructions from the agent definition:\n<agent-instructions>\n{}\n</agent-instructions>\n",
            a.name,
            a.description,
            fence("agent-instructions", &a.instructions)
        );
    }
    if !skills.is_empty() {
        prompt.push_str("\nSkills for this task (follow them):\n");
        for s in skills {
            let _ = write!(
                prompt,
                "<skill name=\"{}\">\n{}\n</skill>\n",
                s.name,
                fence("skill", &s.body)
            );
        }
    }
    prompt
}

fn assignment(job: &AttemptJob, tools: &[ToolSpec], loadable: &[String]) -> String {
    let mut s = format!(
        "Plan goal: {}\nTask {}: {}\n\nInstructions:\n{}\n",
        job.goal, job.number, job.spec.title, job.spec.instructions
    );
    if !job.spec.success_criteria.trim().is_empty() {
        let _ = writeln!(s, "\nSuccess criteria: {}", job.spec.success_criteria);
    }
    let _ = writeln!(
        s,
        "\nAttempt {}. Strategy for this attempt: {}",
        job.attempt, job.strategy
    );
    if !job.prior.is_empty() {
        s.push_str("Previous attempts (do not repeat what failed):\n");
        for (n, strategy, result) in &job.prior {
            let _ = writeln!(s, "- attempt {n} ({strategy}): {result}");
        }
    }
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    let _ = write!(
        s,
        "\nAvailable tools: {}. Finish by calling report_result.",
        names.join(", ")
    );
    if !loadable.is_empty() {
        let _ = write!(s, "\nLoadable with tool_search: {}.", loadable.join(", "));
    }
    s
}

fn render(template: &str, workdir: &str) -> String {
    let shell = if cfg!(windows) {
        "PowerShell".to_owned()
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "sh".into())
    };
    template
        .replace("{{workdir}}", workdir)
        .replace("{{os}}", std::env::consts::OS)
        .replace("{{shell}}", &shell)
        .replace(
            "{{date}}",
            &chrono::Local::now().format("%Y-%m-%d").to_string(),
        )
}

pub(crate) async fn run_attempt(sup: Arc<Supervisor>, job: AttemptJob) {
    let n = job.number;
    let used = Arc::new(Mutex::new(BTreeSet::new()));
    let (end, tokens) = attempt(&sup, &job, &used).await;
    let tools: Vec<String> = used
        .lock()
        .map(|u| {
            u.iter()
                .filter(|t| !SUBAGENT_TOOLS.contains(&t.as_str()))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    sup.deps.board.remove(job.agent);
    sup.deps.approver.forget(job.agent);
    sup.deps.tool_sets.remove(job.agent);
    if let (Some(memory), Some(session)) = (&sup.deps.memory, &job.session) {
        memory.submit(MemoryJob::SubAgentExchanges {
            session: session.clone(),
            thread: job.agent,
            role: job.role,
            upto: u32::MAX,
            ended: true,
        });
    }
    let advice = retry_advice(&sup, &job, &end).await;
    sup.finish_attempt(n, end, tokens, tools, advice);
}

/// Retry vs give up from the decision model, for failures that still have retries left.
async fn retry_advice(sup: &Arc<Supervisor>, job: &AttemptJob, end: &AttemptEnd) -> Option<bool> {
    let decider = sup.deps.decider.as_ref().filter(|d| d.is_enabled())?;
    let failure = match end {
        AttemptEnd::Reported(r) if r.outcome == Outcome::Failed => r.summary.clone(),
        AttemptEnd::Error(e) => e.clone(),
        _ => return None,
    };
    if job.attempt >= sup.max_attempts() {
        return None;
    }
    let mut attempts = String::new();
    for (n, strategy, result) in &job.prior {
        let _ = writeln!(attempts, "{n}. strategy: {strategy}; result: {result}");
    }
    let _ = writeln!(
        attempts,
        "{}. strategy: {}; result: Failed: {failure}",
        job.attempt, job.strategy
    );
    let instructions: String = job.spec.instructions.chars().take(600).collect();
    let task = format!("{}\n{instructions}", job.spec.title);
    let advice = decider.retry_worthwhile(&task, &attempts).await;
    tracing::info!(task = job.number, ?advice, "decision model retry advice");
    advice
}

async fn attempt(
    sup: &Arc<Supervisor>,
    job: &AttemptJob,
    used: &Arc<Mutex<BTreeSet<String>>>,
) -> (AttemptEnd, u64) {
    let d = &sup.deps;
    let Some(log) = job.session.as_ref().and_then(|s| s.log.clone()) else {
        return (AttemptEnd::Error("no active session".into()), 0);
    };
    if let (Some(name), None) = (&job.spec.agent, &job.custom) {
        return (
            AttemptEnd::Error(format!(
                "custom agent `{name}` no longer exists (it was deleted); revise the step to use another agent or a plain role"
            )),
            0,
        );
    }
    let cfg = &d.config;
    let budget = Budget {
        tokens: cfg.retry.task_token_budget,
        time: Duration::from_secs(cfg.retry.task_time_budget_secs),
    };
    d.board.register(
        job.agent,
        job.role,
        &job.label,
        Some(job.number),
        Some(budget),
    );
    d.approver
        .label(job.agent, &job.label, Some(format!("task {}", job.number)));

    let mode = d.permissions.lock().map_or(Mode::Cowork, |p| p.mode());
    let registry = d.executor.registry();
    let catalog = d.skills.load();
    let skills: Vec<&Skill> = job
        .spec
        .skills
        .iter()
        .filter_map(|n| catalog.get(n))
        .collect();
    // Skills narrow tools only when every attached skill declares its tools.
    let skill_tools: Option<Vec<String>> =
        (!skills.is_empty() && skills.iter().all(|s| s.allowed_tools.is_some())).then(|| {
            skills
                .iter()
                .flat_map(|s| s.tool_names().unwrap_or_default())
                .collect()
        });
    let all = agent_tools(
        job.role,
        job.custom.as_ref().and_then(|a| a.tools.as_deref()),
        job.spec.allowed_tools.as_deref(),
        skill_tools.as_deref(),
        mode,
        &registry.specs(|_| true),
        |name| registry.get(name).map(|t| t.risk()),
    );
    let (deferred, mut tools): (Vec<ToolSpec>, Vec<ToolSpec>) = all.into_iter().partition(|t| {
        registry
            .get(&t.name)
            .is_some_and(|x| x.tier() == Tier::Deferred)
    });
    let mut set = ActiveToolSet::new(deferred, cfg.tools.offload_after_turns);
    set.begin_turn();
    for name in job.spec.allowed_tools.iter().flatten() {
        set.load(name);
    }
    let loadable: Vec<String> = set.unloaded().iter().map(|t| t.name.clone()).collect();
    if !loadable.is_empty()
        && let Some(search) = registry.get(TOOL_SEARCH)
    {
        tools.push(search.spec());
    }
    let allowed: HashSet<String> = tools
        .iter()
        .chain(set.pool())
        .map(|t| t.name.clone())
        .collect();
    let active = Arc::new(Mutex::new(set));
    d.tool_sets.register(job.agent, active.clone());
    let text = assignment(job, &tools, &loadable);

    let _ = log.append(
        d.orchestrator,
        Record::AgentSpawn {
            child: job.agent,
            role: job.role,
            task_id: Some(job.task_id),
        },
    );
    let assign = ToSubAgent::Assign {
        task: job.number,
        attempt: job.attempt,
        strategy: job.strategy.clone(),
        instructions: text.clone(),
        allowed_tools: allowed.iter().cloned().collect(),
    };
    let _ = log.append(
        d.orchestrator,
        Record::AgentMessage {
            from: d.orchestrator,
            to: job.agent,
            msg_json: serde_json::to_string(&assign).unwrap_or_default(),
        },
    );

    let target = d.models.resolve(job.role);
    let window = match d
        .provider
        .model_info_for(Some(target.provider), &target.model)
        .await
    {
        Ok(info) => {
            let configured = d.models.window(target.provider);
            info.context_length
                .map_or(configured, |m| configured.min(m))
        }
        Err(e) => {
            return (
                AttemptEnd::Error(format!(
                    "model `{}` ({}) unavailable: {e}",
                    target.model, target.provider
                )),
                0,
            );
        }
    };
    let settings = EngineSettings {
        provider: Some(target.provider),
        model: target.model,
        think: cfg.thinking.level,
        window,
        compaction_threshold: cfg.memory.compaction_threshold,
        max_steps: MAX_STEPS,
        max_tool_calls: MAX_TOOL_CALLS,
        allowed: Some(allowed),
        active: Some(active),
    };
    let mut conv = Conversation::new(render(
        &system_prompt(
            job.role,
            job.custom.as_ref(),
            &skills,
            d.direct.load(Ordering::Relaxed),
            crate::instructions::load(&d.data.instructions_file(), d.tool_env.guard.workdir())
                .as_deref(),
        ),
        &d.tool_env.guard.workdir().display().to_string(),
    ));
    conv.begin_exchange(&text);
    let ctx = ToolCtx {
        env: d.tool_env.clone(),
        limits: ToolLimits::from_config(&cfg.tools),
        cancel: job.cancel.clone(),
        agent: job.agent,
    };
    let on_step: Option<Arc<dyn Fn() + Send + Sync>> = match (&d.memory, &job.session) {
        (Some(memory), Some(session)) => {
            let (memory, session) = (memory.clone(), session.clone());
            let (thread, role) = (job.agent, job.role);
            let every = cfg.memory.subagent_exchange_trigger.max(1);
            let steps = AtomicU32::new(0);
            Some(Arc::new(move || {
                let done = steps.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(every) {
                    memory.submit(MemoryJob::SubAgentExchanges {
                        session: session.clone(),
                        thread,
                        role,
                        upto: done,
                        ended: false,
                    });
                }
            }))
        }
        _ => None,
    };
    let sink = TurnSink {
        ui: d.ui.clone(),
        log,
        agent: job.agent,
        chat: false,
        board: Some(d.board.clone()),
        totals: d.totals.clone(),
        interjections: None,
        on_step,
        tools_used: Some(used.clone()),
    };
    let engine = Engine::new(d.provider.clone(), d.executor.clone());
    let mut ledger = UsageSnapshot::default();
    let interval = Duration::from_secs(cfg.agents.heartbeat_interval_secs.max(1));

    let end = {
        let mut run = pin!(engine.run_turn(&mut conv, &settings, &tools, &ctx, &sink, &mut ledger));
        let mut beat = tokio::time::interval(interval);
        loop {
            tokio::select! {
                end = &mut run => break end,
                _ = beat.tick() => d.board.beat(job.agent),
            }
        }
    };
    sink.phase(Phase::Idle);
    let tokens = ledger.session_in.saturating_add(ledger.session_out);
    let per_agent = d.board.get(job.agent).map_or(tokens, |h| h.tokens);

    let result = match (sup.take_report(job.agent), end) {
        (Some(report), _) => AttemptEnd::Reported(report),
        (None, TurnEnd::Cancelled) => AttemptEnd::Cancelled,
        (None, TurnEnd::Failed(e) | TurnEnd::BudgetExceeded(e)) => AttemptEnd::Error(e),
        (None, TurnEnd::Completed) => {
            let last = conv
                .messages()
                .iter()
                .rev()
                .find_map(|m| match m {
                    Message::Assistant { content, .. } if !content.is_empty() => {
                        Some(content.clone())
                    }
                    _ => None,
                })
                .unwrap_or_default();
            AttemptEnd::NoReport(last)
        }
    };
    (result, per_agent)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: String::new(),
            parameters: json!({}),
        }
    }

    #[test]
    fn effective_tools_intersect_role_task_and_mode() {
        let specs: Vec<ToolSpec> = [
            "read_file",
            "search_files",
            "edit_file",
            "shell",
            "plan_create",
            "report_result",
            "ask_orchestrator",
        ]
        .map(spec)
        .to_vec();
        let risk = |n: &str| match n {
            "read_file" | "search_files" => Some(RiskClass::Read),
            "edit_file" => Some(RiskClass::Write),
            "shell" => Some(RiskClass::Exec),
            _ => Some(RiskClass::Internal),
        };
        let names = |v: Vec<ToolSpec>| v.into_iter().map(|s| s.name).collect::<Vec<_>>();

        let worker = names(effective_tools(
            Role::Worker,
            None,
            None,
            Mode::Cowork,
            &specs,
            risk,
        ));
        assert_eq!(
            worker,
            [
                "read_file",
                "search_files",
                "edit_file",
                "shell",
                "report_result",
                "ask_orchestrator"
            ]
        );

        let narrowed = names(effective_tools(
            Role::Worker,
            Some(&["read_file".into(), "plan_create".into()]),
            None,
            Mode::Auto,
            &specs,
            risk,
        ));
        assert_eq!(
            narrowed,
            ["read_file", "report_result", "ask_orchestrator"],
            "allowlist cannot add orchestrator tools"
        );

        let research = names(effective_tools(
            Role::Research,
            Some(&["shell".into(), "read_file".into()]),
            None,
            Mode::Auto,
            &specs,
            risk,
        ));
        assert_eq!(research, ["read_file", "report_result", "ask_orchestrator"]);

        // A skill listing shell cannot bring it back in plan mode.
        let skilled = names(effective_tools(
            Role::Worker,
            None,
            Some(&["read_file".into(), "shell".into()]),
            Mode::Plan,
            &specs,
            risk,
        ));
        assert_eq!(skilled, ["read_file", "report_result", "ask_orchestrator"]);

        let plan_mode = names(effective_tools(
            Role::Worker,
            None,
            None,
            Mode::Plan,
            &specs,
            risk,
        ));
        assert_eq!(
            plan_mode,
            [
                "read_file",
                "search_files",
                "report_result",
                "ask_orchestrator"
            ]
        );
    }

    #[test]
    fn custom_agent_tools_replace_defaults_but_never_escalate() {
        let specs: Vec<ToolSpec> = [
            "read_file",
            "search_files",
            "edit_file",
            "shell",
            "web_fetch",
            "plan_create",
            "memory_recall",
            "mcp__db__query",
            "report_result",
            "ask_orchestrator",
        ]
        .map(spec)
        .to_vec();
        let risk = |n: &str| match n {
            "read_file" | "search_files" => Some(RiskClass::Read),
            "edit_file" => Some(RiskClass::Write),
            "shell" => Some(RiskClass::Exec),
            "web_fetch" => Some(RiskClass::Network),
            n if n.starts_with("mcp__") => Some(RiskClass::External),
            _ => Some(RiskClass::Internal),
        };
        let names = |v: Vec<ToolSpec>| v.into_iter().map(|s| s.name).collect::<Vec<_>>();
        let custom: Vec<String> = [
            "read_file",
            "web_fetch",
            "shell",
            "plan_create",
            "memory_recall",
            "mcp__db__query",
        ]
        .map(String::from)
        .to_vec();

        let all = names(agent_tools(
            Role::Worker,
            Some(&custom),
            None,
            None,
            Mode::Auto,
            &specs,
            risk,
        ));
        assert_eq!(
            all,
            [
                "read_file",
                "shell",
                "web_fetch",
                "mcp__db__query",
                "report_result",
                "ask_orchestrator"
            ],
            "web_fetch joins a worker-based agent; control and memory tools never do"
        );

        let narrowed = names(agent_tools(
            Role::Worker,
            Some(&custom),
            Some(&[
                "read_file".into(),
                "edit_file".into(),
                "mcp__other__x".into(),
            ]),
            None,
            Mode::Auto,
            &specs,
            risk,
        ));
        assert_eq!(narrowed, ["read_file", "report_result", "ask_orchestrator"]);

        let plan_mode = names(agent_tools(
            Role::Worker,
            Some(&custom),
            None,
            None,
            Mode::Plan,
            &specs,
            risk,
        ));
        assert_eq!(
            plan_mode,
            ["read_file", "report_result", "ask_orchestrator"]
        );
        assert!(
            delegable("browser_click") && !delegable("plan_create") && !delegable("agent_create")
        );
    }

    #[test]
    fn browser_tools_need_the_allowlist() {
        let specs: Vec<ToolSpec> = [
            "read_file",
            "web_search",
            "web_fetch",
            "browser_open",
            "browser_snapshot",
            "browser_fill_form",
        ]
        .map(spec)
        .to_vec();
        let risk = |n: &str| {
            Some(if n.starts_with("browser") {
                RiskClass::Browser
            } else if n.starts_with("web") {
                RiskClass::Network
            } else {
                RiskClass::Read
            })
        };
        let names = |v: &[ToolSpec]| v.iter().map(|s| s.name.clone()).collect::<Vec<_>>();

        let all = effective_tools(Role::Research, None, None, Mode::Auto, &specs, risk);
        assert_eq!(
            names(&all),
            [
                "read_file",
                "web_search",
                "web_fetch",
                "browser_open",
                "browser_snapshot"
            ],
            "research never fills forms"
        );

        let narrowed = effective_tools(
            Role::Research,
            Some(&["read_file".into(), "web_fetch".into()]),
            None,
            Mode::Auto,
            &specs,
            risk,
        );
        let set = ActiveToolSet::new(narrowed, 3);
        assert!(set.search("browser", 5).is_empty());
        assert_eq!(names(&set.search("fetch", 5)), ["web_fetch"]);

        let plan = effective_tools(Role::Research, None, None, Mode::Plan, &specs, risk);
        assert_eq!(names(&plan), ["read_file"]);
    }

    #[test]
    fn mcp_tools_need_an_explicit_allowlist_entry() {
        let specs: Vec<ToolSpec> = [
            "read_file",
            "mcp__github__list_issues",
            "mcp__github__merge",
        ]
        .map(spec)
        .to_vec();
        let risk = |n: &str| {
            Some(if n.starts_with("mcp__") {
                RiskClass::External
            } else {
                RiskClass::Read
            })
        };
        let names = |v: Vec<ToolSpec>| v.into_iter().map(|s| s.name).collect::<Vec<_>>();
        let worker = |allowed: Option<&[String]>, mode| {
            names(effective_tools(
                Role::Worker,
                allowed,
                None,
                mode,
                &specs,
                risk,
            ))
        };
        assert_eq!(worker(None, Mode::Auto), ["read_file"]);
        let allow = [
            "read_file".to_owned(),
            "mcp__github__list_issues".to_owned(),
        ];
        assert_eq!(
            worker(Some(&allow), Mode::Auto),
            ["read_file", "mcp__github__list_issues"]
        );
        assert_eq!(worker(Some(&allow), Mode::Plan), ["read_file"]);
    }
}
