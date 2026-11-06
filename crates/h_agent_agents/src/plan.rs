//! Plan and task model (DESIGN §6.1.4). Tasks are numbered from 1 for the model's convenience.

use chrono::{DateTime, Utc};
use h_agent_core::events::{PlanView, TaskLine};
use h_agent_core::{AgentId, PlanId, TaskId, TaskStatus};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Success,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskRole {
    #[serde(rename = "worker")]
    Worker,
    #[serde(rename = "research")]
    Research,
    /// Done by the Orchestrator itself; never dispatched.
    #[serde(rename = "self")]
    Orchestrator,
}

impl TaskRole {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskRole::Worker => "worker",
            TaskRole::Research => "research",
            TaskRole::Orchestrator => "self",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepSpec {
    pub title: String,
    pub instructions: String,
    pub role: TaskRole,
    /// Further narrows the role's default tools.
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    #[serde(default)]
    pub depends_on: Vec<u32>,
    #[serde(default)]
    pub success_criteria: String,
    /// Skills whose bodies go into the sub-agent's system prompt.
    #[serde(default)]
    pub skills: Vec<String>,
    /// Custom agent (DESIGN §6.5) that runs this step; its base role replaces `role`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

impl StepSpec {
    /// The custom agent's name, else the role.
    pub fn assignee(&self) -> &str {
        self.agent.as_deref().unwrap_or(self.role.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub n: u32,
    pub strategy: String,
    pub agent: Option<AgentId>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub outcome: Option<Outcome>,
    pub summary: String,
    pub reason: String,
    pub evidence: Vec<String>,
    pub tokens: u64,
    /// Tools the sub-agent called in this attempt.
    #[serde(default)]
    pub tools_used: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub number: u32,
    pub spec: StepSpec,
    pub status: TaskStatus,
    pub attempts: Vec<Attempt>,
    pub question: Option<String>,
}

impl Task {
    pub fn last_attempt(&self) -> Option<&Attempt> {
        self.attempts.last()
    }

    pub fn last_attempt_mut(&mut self) -> Option<&mut Attempt> {
        self.attempts.last_mut()
    }

    pub fn retries_used(&self) -> u32 {
        u32::try_from(self.attempts.len())
            .unwrap_or(u32::MAX)
            .saturating_sub(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub id: PlanId,
    pub goal: String,
    pub created_at: DateTime<Utc>,
    pub tasks: Vec<Task>,
    /// Triage classification of the turn that created the plan.
    #[serde(default)]
    pub query_type: String,
    #[serde(default)]
    pub topic: String,
    /// Lookup entry the plan was built from, credited by the Scorer.
    #[serde(default)]
    pub template_id: Option<String>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
}

pub fn is_terminal(s: TaskStatus) -> bool {
    matches!(
        s,
        TaskStatus::Success
            | TaskStatus::Partial
            | TaskStatus::Failed
            | TaskStatus::Cancelled
            | TaskStatus::Blocked
    )
}

pub fn is_active(s: TaskStatus) -> bool {
    matches!(
        s,
        TaskStatus::Queued | TaskStatus::Running | TaskStatus::AwaitingInput
    )
}

/// Lowercase alphanumeric words, so trivially reworded strategies count as the same.
pub fn normalize_strategy(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

impl Plan {
    pub fn new(goal: &str, steps: Vec<StepSpec>) -> Result<Self, String> {
        if goal.trim().is_empty() {
            return Err("goal must not be empty".into());
        }
        let mut plan = Self {
            id: PlanId::new(),
            goal: goal.trim().to_owned(),
            created_at: Utc::now(),
            tasks: Vec::new(),
            query_type: String::new(),
            topic: String::new(),
            template_id: None,
            finished_at: None,
        };
        if steps.is_empty() {
            return Err("a plan needs at least one step".into());
        }
        plan.add(steps)?;
        Ok(plan)
    }

    /// Appends steps. A step may only depend on tasks numbered before it, which rules out cycles.
    pub fn add(&mut self, steps: Vec<StepSpec>) -> Result<Vec<u32>, String> {
        let first = u32::try_from(self.tasks.len()).unwrap_or(u32::MAX) + 1;
        for (i, s) in steps.iter().enumerate() {
            let n = first + u32::try_from(i).unwrap_or(u32::MAX);
            if s.title.trim().is_empty() {
                return Err(format!("step {n}: title must not be empty"));
            }
            if let Some(d) = s.depends_on.iter().find(|d| **d == 0 || **d >= n) {
                return Err(format!(
                    "step {n}: depends_on {d} must refer to an earlier step"
                ));
            }
            if let Some(d) = s.depends_on.iter().find(|d| {
                self.task(**d)
                    .is_some_and(|t| t.status == TaskStatus::Cancelled)
            }) {
                return Err(format!("step {n}: depends on cancelled task {d}"));
            }
        }
        let mut added = Vec::new();
        for (i, spec) in steps.into_iter().enumerate() {
            let number = first + u32::try_from(i).unwrap_or(u32::MAX);
            self.tasks.push(Task {
                id: TaskId::new(),
                number,
                spec,
                status: TaskStatus::Pending,
                attempts: Vec::new(),
                question: None,
            });
            added.push(number);
        }
        Ok(added)
    }

    /// Cancels tasks that have not started and that no remaining task depends on.
    pub fn remove(&mut self, numbers: &[u32]) -> Result<(), String> {
        for n in numbers {
            let t = self.task(*n).ok_or_else(|| format!("no task {n}"))?;
            if !matches!(
                t.status,
                TaskStatus::Pending | TaskStatus::Queued | TaskStatus::Blocked
            ) {
                return Err(format!(
                    "task {n} is {}; only tasks that have not started can be removed",
                    t.status
                ));
            }
            let dependent = self.tasks.iter().find(|o| {
                !numbers.contains(&o.number)
                    && o.spec.depends_on.contains(n)
                    && o.status != TaskStatus::Cancelled
            });
            if let Some(o) = dependent {
                return Err(format!(
                    "task {} depends on task {n}; remove or revise it too",
                    o.number
                ));
            }
        }
        for n in numbers {
            if let Some(t) = self.task_mut(*n) {
                t.status = TaskStatus::Cancelled;
                if t.last_attempt().is_some_and(|a| a.started_at.is_none()) {
                    t.attempts.pop();
                }
            }
        }
        Ok(())
    }

    pub fn task(&self, n: u32) -> Option<&Task> {
        self.tasks.iter().find(|t| t.number == n)
    }

    pub fn task_mut(&mut self, n: u32) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.number == n)
    }

    /// Dependencies count as met when they succeeded or partially succeeded.
    pub fn deps_met(&self, n: u32) -> Result<(), String> {
        let t = self.task(n).ok_or_else(|| format!("no task {n}"))?;
        let waiting: Vec<String> = t
            .spec
            .depends_on
            .iter()
            .filter(|d| {
                !self.task(**d).is_some_and(|dt| {
                    matches!(dt.status, TaskStatus::Success | TaskStatus::Partial)
                })
            })
            .map(|d| format!("#{d}"))
            .collect();
        if waiting.is_empty() {
            Ok(())
        } else {
            Err(format!("task {n} is waiting on {}", waiting.join(", ")))
        }
    }

    /// Marks every task that (transitively) depends on `n` and has not started as Blocked.
    pub fn block_dependents(&mut self, n: u32) -> Vec<u32> {
        let mut blocked = Vec::new();
        let mut frontier = vec![n];
        while let Some(cur) = frontier.pop() {
            for t in &mut self.tasks {
                if t.spec.depends_on.contains(&cur)
                    && matches!(t.status, TaskStatus::Pending | TaskStatus::Queued)
                {
                    t.status = TaskStatus::Blocked;
                    blocked.push(t.number);
                    frontier.push(t.number);
                }
            }
        }
        blocked
    }

    /// Returns Blocked dependents to Pending once `n` is satisfied again.
    pub fn unblock_dependents(&mut self, n: u32) {
        let numbers: Vec<u32> = self
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Blocked && t.spec.depends_on.contains(&n))
            .map(|t| t.number)
            .collect();
        for m in numbers {
            if self.deps_met(m).is_ok()
                && let Some(t) = self.task_mut(m)
            {
                t.status = TaskStatus::Pending;
            }
        }
    }

    pub fn has_active(&self) -> bool {
        self.tasks.iter().any(|t| is_active(t.status))
    }

    pub fn is_finished(&self) -> bool {
        self.tasks.iter().all(|t| is_terminal(t.status))
    }

    pub fn view(&self, max_attempts: u32) -> PlanView {
        PlanView {
            goal: self.goal.clone(),
            tasks: self
                .tasks
                .iter()
                .map(|t| TaskLine {
                    number: t.number,
                    title: t.spec.title.clone(),
                    role: t.spec.assignee().to_owned(),
                    status: t.status,
                    attempt: u32::try_from(t.attempts.len()).unwrap_or(u32::MAX),
                    max_attempts,
                    note: t
                        .last_attempt()
                        .map(|a| {
                            if a.reason.is_empty() {
                                a.summary.clone()
                            } else {
                                a.reason.clone()
                            }
                        })
                        .unwrap_or_default(),
                })
                .collect(),
        }
    }

    /// Compact text view for the Orchestrator's context.
    pub fn digest(&self, max_attempts: u32) -> String {
        let mut out = format!("plan: {}\n", self.goal);
        for t in &self.tasks {
            let deps = if t.spec.depends_on.is_empty() {
                String::new()
            } else {
                format!(
                    " after {}",
                    t.spec
                        .depends_on
                        .iter()
                        .map(|d| format!("#{d}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            };
            out.push_str(&format!(
                "#{} [{}] {} ({}{deps}) attempts {}/{max_attempts}",
                t.number,
                t.status,
                t.spec.title,
                t.spec.assignee(),
                t.attempts.len()
            ));
            if let Some(a) = t.last_attempt().filter(|a| a.outcome.is_some()) {
                let text = if a.reason.is_empty() {
                    &a.summary
                } else {
                    &a.reason
                };
                out.push_str(&format!(
                    " — {}",
                    text.chars().take(200).collect::<String>()
                ));
            }
            if let Some(q) = &t.question {
                out.push_str(&format!(" — asks: {q}"));
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(title: &str, deps: &[u32]) -> StepSpec {
        StepSpec {
            title: title.into(),
            instructions: "do it".into(),
            role: TaskRole::Worker,
            allowed_tools: None,
            depends_on: deps.to_vec(),
            success_criteria: String::new(),
            skills: Vec::new(),
            agent: None,
        }
    }

    #[test]
    fn dependencies_must_point_backwards() {
        assert!(Plan::new("g", vec![step("a", &[]), step("b", &[2])]).is_err());
        assert!(Plan::new("g", vec![step("a", &[0])]).is_err());
        assert!(Plan::new("g", vec![]).is_err());
        let p = Plan::new(
            "g",
            vec![step("a", &[]), step("b", &[1]), step("c", &[1, 2])],
        )
        .unwrap();
        assert_eq!(p.deps_met(3).unwrap_err(), "task 3 is waiting on #1, #2");
        assert!(p.deps_met(1).is_ok());
    }

    #[test]
    fn blocking_is_transitive_and_reversible() {
        let mut p = Plan::new(
            "g",
            vec![
                step("a", &[]),
                step("b", &[1]),
                step("c", &[2]),
                step("d", &[]),
            ],
        )
        .unwrap();
        assert_eq!(p.block_dependents(1), vec![2, 3]);
        assert_eq!(p.task(4).unwrap().status, TaskStatus::Pending);
        p.task_mut(1).unwrap().status = TaskStatus::Success;
        p.unblock_dependents(1);
        assert_eq!(p.task(2).unwrap().status, TaskStatus::Pending);
        assert_eq!(
            p.task(3).unwrap().status,
            TaskStatus::Blocked,
            "still waits on #2"
        );
    }

    #[test]
    fn remove_respects_dependents() {
        let mut p = Plan::new("g", vec![step("a", &[]), step("b", &[1])]).unwrap();
        assert!(
            p.remove(&[1])
                .unwrap_err()
                .contains("task 2 depends on task 1")
        );
        p.remove(&[1, 2]).unwrap();
        assert!(p.is_finished());
    }

    #[test]
    fn strategies_normalize() {
        assert_eq!(
            normalize_strategy("  Use GREP!  then   read."),
            "use grep then read"
        );
        assert_eq!(
            normalize_strategy("use-grep"),
            normalize_strategy("Use grep")
        );
    }
}
