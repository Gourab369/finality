//! Orchestrator control tools (DESIGN §6.1.3) and the sub-agent reporting tools.

use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use h_agent_core::RiskClass;
use h_agent_safety::Location;
use h_agent_tools::tool::parse_args;
use h_agent_tools::{ActionPreview, Tool, ToolCtx, ToolError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::plan::{Outcome, StepSpec};
use crate::questions::UiQuestions;
use crate::supervisor::{Report, Supervisor};

/// Filled in once the supervisor exists (it needs the executor that holds these tools).
pub type SupervisorSlot = Arc<OnceLock<Weak<Supervisor>>>;

pub const ORCHESTRATOR_CONTROL: &[&str] = &[
    "plan_create",
    "plan_revise",
    "tasks_dispatch",
    "tasks_await",
    "task_answer",
    "task_set_outcome",
    "task_retry",
    "task_cancel",
    "ask_user",
];

const MAX_AWAIT_SECS: u64 = 600;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    PlanCreate,
    PlanRevise,
    TasksDispatch,
    TasksAwait,
    TaskAnswer,
    TaskSetOutcome,
    TaskRetry,
    TaskCancel,
    AskUser,
    ReportResult,
    AskOrchestrator,
}

struct ControlTool {
    kind: Kind,
    sup: SupervisorSlot,
    questions: Arc<UiQuestions>,
}

pub fn control_tools(sup: SupervisorSlot, questions: Arc<UiQuestions>) -> Vec<Arc<dyn Tool>> {
    [
        Kind::PlanCreate,
        Kind::PlanRevise,
        Kind::TasksDispatch,
        Kind::TasksAwait,
        Kind::TaskAnswer,
        Kind::TaskSetOutcome,
        Kind::TaskRetry,
        Kind::TaskCancel,
        Kind::AskUser,
        Kind::ReportResult,
        Kind::AskOrchestrator,
    ]
    .into_iter()
    .map(|kind| {
        Arc::new(ControlTool {
            kind,
            sup: sup.clone(),
            questions: questions.clone(),
        }) as Arc<dyn Tool>
    })
    .collect()
}

fn step_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "instructions": { "type": "string", "description": "Everything the sub-agent needs; it cannot see this conversation" },
            "role": { "type": "string", "enum": ["worker", "research", "self"], "description": "worker edits and runs commands, research only reads, self means you do it" },
            "allowed_tools": { "type": "array", "items": { "type": "string" }, "description": "Optional: narrow the role's tools" },
            "depends_on": { "type": "array", "items": { "type": "integer" }, "description": "Numbers of earlier steps" },
            "success_criteria": { "type": "string" },
            "skills": { "type": "array", "items": { "type": "string" }, "description": "Optional: skills from the catalog for the sub-agent to follow" },
            "agent": { "type": "string", "description": "Optional: a custom agent from the catalog runs this step (its base role replaces role)" }
        },
        "required": ["title", "instructions", "role"]
    })
}

fn task_arg() -> Value {
    json!({ "type": "integer", "minimum": 1, "description": "Task number" })
}

#[derive(Deserialize)]
struct PlanCreateArgs {
    goal: String,
    #[serde(default)]
    steps: Vec<StepSpec>,
    #[serde(default)]
    template_id: Option<String>,
}
#[derive(Deserialize)]
struct PlanReviseArgs {
    #[serde(default)]
    add: Vec<StepSpec>,
    #[serde(default)]
    remove: Vec<u32>,
}
#[derive(Deserialize)]
struct DispatchArgs {
    tasks: Vec<u32>,
    strategy: Option<String>,
}
#[derive(Deserialize)]
struct AwaitArgs {
    timeout_secs: Option<u64>,
}
#[derive(Deserialize)]
struct AnswerArgs {
    task: u32,
    answer: String,
}
#[derive(Deserialize)]
struct OutcomeArgs {
    task: u32,
    outcome: Outcome,
    #[serde(default)]
    reason: String,
}
#[derive(Deserialize)]
struct RetryArgs {
    task: u32,
    strategy: String,
}
#[derive(Deserialize)]
struct TaskArgs {
    task: u32,
}
#[derive(Deserialize)]
struct AskUserArgs {
    question: String,
    #[serde(default)]
    options: Vec<String>,
}
#[derive(Deserialize)]
struct ReportArgs {
    outcome: Outcome,
    summary: String,
    #[serde(default)]
    evidence: Vec<String>,
}
#[derive(Deserialize)]
struct AskOrchestratorArgs {
    question: String,
}

fn result(r: Result<String, String>) -> ToolOutput {
    match r {
        Ok(text) => ToolOutput::ok(text),
        Err(e) => ToolOutput::error(e),
    }
}

impl ControlTool {
    fn supervisor(&self) -> Result<Arc<Supervisor>, ToolError> {
        self.sup
            .get()
            .and_then(Weak::upgrade)
            .ok_or_else(|| ToolError::Failed("the task supervisor is not available".into()))
    }

    fn validate(&self, args: &Value) -> Result<(), ToolError> {
        match self.kind {
            Kind::PlanCreate => parse_args::<PlanCreateArgs>(args).map(drop),
            Kind::PlanRevise => parse_args::<PlanReviseArgs>(args).map(drop),
            Kind::TasksDispatch => parse_args::<DispatchArgs>(args).map(drop),
            Kind::TasksAwait => parse_args::<AwaitArgs>(args).map(drop),
            Kind::TaskAnswer => parse_args::<AnswerArgs>(args).map(drop),
            Kind::TaskSetOutcome => parse_args::<OutcomeArgs>(args).map(drop),
            Kind::TaskRetry => parse_args::<RetryArgs>(args).map(drop),
            Kind::TaskCancel => parse_args::<TaskArgs>(args).map(drop),
            Kind::AskUser => parse_args::<AskUserArgs>(args).map(drop),
            Kind::ReportResult => parse_args::<ReportArgs>(args).map(drop),
            Kind::AskOrchestrator => parse_args::<AskOrchestratorArgs>(args).map(drop),
        }
    }
}

#[async_trait]
impl Tool for ControlTool {
    fn name(&self) -> &str {
        match self.kind {
            Kind::PlanCreate => "plan_create",
            Kind::PlanRevise => "plan_revise",
            Kind::TasksDispatch => "tasks_dispatch",
            Kind::TasksAwait => "tasks_await",
            Kind::TaskAnswer => "task_answer",
            Kind::TaskSetOutcome => "task_set_outcome",
            Kind::TaskRetry => "task_retry",
            Kind::TaskCancel => "task_cancel",
            Kind::AskUser => "ask_user",
            Kind::ReportResult => "report_result",
            Kind::AskOrchestrator => "ask_orchestrator",
        }
    }

    fn description(&self) -> &str {
        match self.kind {
            Kind::PlanCreate => {
                "Create a plan of numbered steps for multi-step work. Steps can depend on earlier steps. \
                 With template_id (from lookup_table_query) the steps may be omitted to reuse the template's steps."
            }
            Kind::PlanRevise => "Add steps to the plan or remove steps that have not started.",
            Kind::TasksDispatch => {
                "Start plan tasks in sub-agents. Tasks whose dependencies are not met are rejected."
            }
            Kind::TasksAwait => {
                "Wait until a task finishes, a sub-agent asks a question, or an alert fires; returns the events and the plan."
            }
            Kind::TaskAnswer => "Answer a question a sub-agent asked.",
            Kind::TaskSetOutcome => {
                "Record or override a task's outcome (success, partial, failed) with a reason."
            }
            Kind::TaskRetry => {
                "Retry a failed or partial task with a different strategy than every earlier attempt."
            }
            Kind::TaskCancel => "Cancel a task; tasks that depend on it become blocked.",
            Kind::AskUser => "Ask the user a clarifying question and wait for the answer.",
            Kind::ReportResult => {
                "Report your task's outcome to the orchestrator. Call exactly once when done."
            }
            Kind::AskOrchestrator => {
                "Ask the orchestrator for information you need to continue; waits for the answer."
            }
        }
    }

    fn schema(&self) -> Value {
        let (props, required): (Value, Vec<&str>) = match self.kind {
            Kind::PlanCreate => (
                json!({
                    "goal": { "type": "string" },
                    "steps": { "type": "array", "items": step_schema() },
                    "template_id": { "type": "string", "description": "Lookup entry the plan is based on" }
                }),
                vec!["goal"],
            ),
            Kind::PlanRevise => (
                json!({ "add": { "type": "array", "items": step_schema() }, "remove": { "type": "array", "items": { "type": "integer" } } }),
                vec![],
            ),
            Kind::TasksDispatch => (
                json!({ "tasks": { "type": "array", "items": { "type": "integer" }, "minItems": 1 }, "strategy": { "type": "string", "description": "Approach for this first attempt" } }),
                vec!["tasks"],
            ),
            Kind::TasksAwait => (
                json!({ "timeout_secs": { "type": "integer", "minimum": 1, "maximum": MAX_AWAIT_SECS, "default": 60 } }),
                vec![],
            ),
            Kind::TaskAnswer => (
                json!({ "task": task_arg(), "answer": { "type": "string" } }),
                vec!["task", "answer"],
            ),
            Kind::TaskSetOutcome => (
                json!({ "task": task_arg(), "outcome": { "type": "string", "enum": ["success", "partial", "failed"] }, "reason": { "type": "string" } }),
                vec!["task", "outcome"],
            ),
            Kind::TaskRetry => (
                json!({ "task": task_arg(), "strategy": { "type": "string" } }),
                vec!["task", "strategy"],
            ),
            Kind::TaskCancel => (json!({ "task": task_arg() }), vec!["task"]),
            Kind::AskUser => (
                json!({ "question": { "type": "string" }, "options": { "type": "array", "items": { "type": "string" } } }),
                vec!["question"],
            ),
            Kind::ReportResult => (
                json!({ "outcome": { "type": "string", "enum": ["success", "partial", "failed"] }, "summary": { "type": "string" }, "evidence": { "type": "array", "items": { "type": "string" } } }),
                vec!["outcome", "summary"],
            ),
            Kind::AskOrchestrator => (
                json!({ "question": { "type": "string" } }),
                vec!["question"],
            ),
        };
        json!({ "type": "object", "properties": props, "required": required })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        self.validate(args)?;
        Ok(ActionPreview::new(self.name().to_owned(), Location::Inside))
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        if self.kind == Kind::AskUser {
            let a: AskUserArgs = parse_args(&args)?;
            let answer = self
                .questions
                .ask("orchestrator", &a.question, a.options, &ctx.cancel)
                .await;
            return Ok(ToolOutput::ok(answer.map_or_else(
                || "(the user did not answer)".to_owned(),
                |a| {
                    if a.trim().is_empty() {
                        "(the user skipped the question)".to_owned()
                    } else {
                        a
                    }
                },
            )));
        }
        let sup = self.supervisor()?;
        let out =
            match self.kind {
                Kind::PlanCreate => {
                    let a: PlanCreateArgs = parse_args(&args)?;
                    result(sup.create_plan(&a.goal, a.steps, a.template_id.as_deref()))
                }
                Kind::PlanRevise => {
                    let a: PlanReviseArgs = parse_args(&args)?;
                    result(sup.revise(a.add, &a.remove))
                }
                Kind::TasksDispatch => {
                    let a: DispatchArgs = parse_args(&args)?;
                    result(sup.dispatch(&a.tasks, a.strategy.as_deref()))
                }
                Kind::TasksAwait => {
                    let a: AwaitArgs = parse_args(&args)?;
                    let secs = a.timeout_secs.unwrap_or(60).clamp(1, MAX_AWAIT_SECS);
                    ToolOutput::ok(
                        sup.await_events(Duration::from_secs(secs), &ctx.cancel)
                            .await,
                    )
                }
                Kind::TaskAnswer => {
                    let a: AnswerArgs = parse_args(&args)?;
                    result(sup.answer(a.task, &a.answer))
                }
                Kind::TaskSetOutcome => {
                    let a: OutcomeArgs = parse_args(&args)?;
                    result(sup.set_outcome(a.task, a.outcome, &a.reason))
                }
                Kind::TaskRetry => {
                    let a: RetryArgs = parse_args(&args)?;
                    result(sup.retry(a.task, &a.strategy))
                }
                Kind::TaskCancel => {
                    let a: TaskArgs = parse_args(&args)?;
                    result(sup.cancel(a.task))
                }
                Kind::ReportResult => {
                    let a: ReportArgs = parse_args(&args)?;
                    let report = Report {
                        outcome: a.outcome,
                        summary: a.summary,
                        evidence: a.evidence,
                    };
                    result(sup.report(ctx.agent, report).map(|()| {
                        "Result recorded. Reply with a one-line final message.".to_owned()
                    }))
                }
                Kind::AskOrchestrator => {
                    let a: AskOrchestratorArgs = parse_args(&args)?;
                    result(sup.ask(ctx.agent, &a.question, &ctx.cancel).await)
                }
                Kind::AskUser => unreachable!("handled above"),
            };
        Ok(out)
    }
}
