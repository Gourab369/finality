//! Per-agent loaded tools, BM25 `tool_search` and offloading (DESIGN §9.3).

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use h_agent_core::{AgentId, RiskClass};
use h_agent_safety::Location;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::registry::ToolSource;
use crate::tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput, ToolSpec, parse_args,
};
pub use h_agent_core::text::bm25;

pub const TOOL_SEARCH: &str = "tool_search";
const SEARCH_LIMIT: usize = 5;

/// Deferred tools an agent may load, and which are loaded (with the turn they were last used).
#[derive(Debug, Clone, Default)]
pub struct ActiveToolSet {
    pool: Vec<ToolSpec>,
    loaded: BTreeMap<String, u32>,
    turn: u32,
    offload_after: u32,
    /// The pool is every deferred tool (the Orchestrator's), so discovered tools join it.
    open: bool,
}

impl ActiveToolSet {
    pub fn new(pool: Vec<ToolSpec>, offload_after: u32) -> Self {
        Self {
            pool,
            loaded: BTreeMap::new(),
            turn: 0,
            offload_after: offload_after.max(1),
            open: false,
        }
    }

    pub fn open(mut self) -> Self {
        self.open = true;
        self
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn pool(&self) -> &[ToolSpec] {
        &self.pool
    }

    /// Replaces the pool; loaded tools that are gone are dropped.
    pub fn sync_pool(&mut self, pool: Vec<ToolSpec>) {
        self.loaded
            .retain(|name, _| pool.iter().any(|t| &t.name == name));
        self.pool = pool;
    }

    /// Adds tools not in the pool yet.
    pub fn extend_pool(&mut self, specs: Vec<ToolSpec>) {
        for s in specs {
            if !self.pool.iter().any(|t| t.name == s.name) {
                self.pool.push(s);
            }
        }
    }

    /// Starts an Orchestrator turn and unloads tools unused for `offload_after` turns. Returns them.
    pub fn begin_turn(&mut self) -> Vec<String> {
        self.turn += 1;
        let (turn, limit) = (self.turn, self.offload_after);
        let stale: Vec<String> = self
            .loaded
            .iter()
            .filter(|(_, last)| turn - **last > limit)
            .map(|(n, _)| n.clone())
            .collect();
        for n in &stale {
            self.loaded.remove(n);
        }
        stale
    }

    pub fn load(&mut self, name: &str) -> bool {
        if self.pool.iter().any(|t| t.name == name) {
            self.loaded.insert(name.to_owned(), self.turn);
            true
        } else {
            false
        }
    }

    /// Marks a loaded tool as used this turn.
    pub fn touch(&mut self, name: &str) {
        if let Some(last) = self.loaded.get_mut(name) {
            *last = self.turn;
        }
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded.contains_key(name)
    }

    pub fn loaded_specs(&self) -> Vec<ToolSpec> {
        self.pool
            .iter()
            .filter(|t| self.loaded.contains_key(&t.name))
            .cloned()
            .collect()
    }

    /// (name, turns since last use) of loaded tools.
    pub fn loaded(&self) -> Vec<(String, u32)> {
        self.loaded
            .iter()
            .map(|(n, last)| (n.clone(), self.turn - last))
            .collect()
    }

    pub fn unloaded(&self) -> Vec<&ToolSpec> {
        self.pool
            .iter()
            .filter(|t| !self.loaded.contains_key(&t.name))
            .collect()
    }

    /// The best pool matches for `query`, without loading them.
    pub fn search(&self, query: &str, limit: usize) -> Vec<ToolSpec> {
        let docs: Vec<String> = self
            .pool
            .iter()
            .map(|t| {
                format!(
                    "{} {} {}",
                    t.name.replace(['_', '-'], " "),
                    t.name,
                    t.description
                )
            })
            .collect();
        bm25(&docs, query)
            .into_iter()
            .take(limit)
            .map(|(i, _)| self.pool[i].clone())
            .collect()
    }

    /// Loads tools named by triage hints: name matches first, then the best search hit.
    pub fn preload_hints(&mut self, hints: &[String]) -> Vec<String> {
        let mut loaded = Vec::new();
        for hint in hints {
            let h = hint.trim().to_lowercase();
            if h.is_empty() {
                continue;
            }
            let mut names: Vec<String> = self
                .pool
                .iter()
                .filter(|t| t.name.to_lowercase().contains(&h))
                .map(|t| t.name.clone())
                .collect();
            if names.is_empty() {
                names = self.search(&h, 1).into_iter().map(|t| t.name).collect();
            }
            for n in names {
                if !self.is_loaded(&n) && self.load(&n) {
                    loaded.push(n);
                }
            }
        }
        loaded
    }

    /// Unloads everything (new session).
    pub fn reset(&mut self) {
        self.loaded.clear();
        self.turn = 0;
    }
}

/// Every agent's tool set, so `tool_search` can find the caller's.
#[derive(Default)]
pub struct ToolSets {
    sets: Mutex<HashMap<AgentId, Arc<Mutex<ActiveToolSet>>>>,
}

impl ToolSets {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn register(&self, agent: AgentId, set: Arc<Mutex<ActiveToolSet>>) {
        if let Ok(mut s) = self.sets.lock() {
            s.insert(agent, set);
        }
    }

    pub fn remove(&self, agent: AgentId) {
        if let Ok(mut s) = self.sets.lock() {
            s.remove(&agent);
        }
    }

    pub fn get(&self, agent: AgentId) -> Option<Arc<Mutex<ActiveToolSet>>> {
        self.sets.lock().ok()?.get(&agent).cloned()
    }
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
}

/// `tool_search(query)`: loads up to 5 matching deferred tools into the caller's set.
/// For open sets it first lets tool sources discover their tools.
pub struct ToolSearch {
    sets: Arc<ToolSets>,
    sources: Vec<Arc<dyn ToolSource>>,
}

impl ToolSearch {
    pub fn new(sets: Arc<ToolSets>) -> Self {
        Self {
            sets,
            sources: Vec::new(),
        }
    }

    pub fn with_sources(mut self, sources: Vec<Arc<dyn ToolSource>>) -> Self {
        self.sources = sources;
        self
    }
}

#[async_trait]
impl Tool for ToolSearch {
    fn name(&self) -> &str {
        TOOL_SEARCH
    }

    fn description(&self) -> &str {
        "Find and load more tools by describing what you need (for example \"search the web\"). \
         Loaded tools become callable from your next step."
    }

    fn schema(&self) -> Value {
        json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: SearchArgs = parse_args(args)?;
        if a.query.trim().is_empty() {
            return Err(ToolError::InvalidArgs("query must not be empty".into()));
        }
        Ok(ActionPreview::new(
            format!("search tools for \"{}\"", a.query.trim()),
            Location::Inside,
        ))
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: SearchArgs = parse_args(&args)?;
        let set = self
            .sets
            .get(ctx.agent)
            .ok_or_else(|| ToolError::Failed("no loadable tools for this agent".into()))?;
        let open = set.lock().is_ok_and(|s| s.is_open());
        let mut problems = Vec::new();
        if open {
            for source in &self.sources {
                problems.extend(source.discover(&ctx.cancel).await);
            }
            if ctx.cancel.is_cancelled() {
                return Err(ToolError::Cancelled);
            }
        }
        let mut set = set
            .lock()
            .map_err(|_| ToolError::Failed("tool set unavailable".into()))?;
        if open {
            set.extend_pool(
                self.sources
                    .iter()
                    .flat_map(|s| s.tools())
                    .filter(|t| t.tier() == Tier::Deferred)
                    .map(|t| t.spec())
                    .collect(),
            );
        }
        let notes: String = problems.iter().map(|p| format!("\nnote: {p}")).collect();
        let hits = set.search(&a.query, SEARCH_LIMIT);
        if hits.is_empty() {
            let names: Vec<&str> = set.pool().iter().map(|t| t.name.as_str()).collect();
            return Ok(ToolOutput::ok(format!(
                "no matching tools. Loadable tools: {}{notes}",
                if names.is_empty() {
                    "(none)".into()
                } else {
                    names.join(", ")
                }
            )));
        }
        let mut out = String::from("Loaded (callable from your next step):");
        for t in &hits {
            set.load(&t.name);
            out.push_str(&format!("\n- {}: {}", t.name, t.description));
        }
        out.push_str(&notes);
        Ok(ToolOutput::ok(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: description.into(),
            parameters: json!({}),
        }
    }

    fn pool() -> Vec<ToolSpec> {
        vec![
            spec(
                "web_search",
                "Search the web and return result titles and links",
            ),
            spec("web_fetch", "Fetch a web page and convert it to text"),
            spec("browser_click", "Click an element in the browser page"),
            spec("weather_lookup", "Look up the weather forecast for a city"),
        ]
    }

    #[test]
    fn bm25_ranks_relevant_tools_first() {
        let set = ActiveToolSet::new(pool(), 3);
        let names = |q| {
            set.search(q, 5)
                .into_iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(names("weather forecast")[0], "weather_lookup");
        assert_eq!(names("fetch a page")[0], "web_fetch");
        assert!(names("database migration").is_empty());
    }

    #[test]
    fn loads_touches_and_offloads_after_idle_turns() {
        let mut set = ActiveToolSet::new(pool(), 2);
        set.begin_turn();
        assert!(set.load("weather_lookup"));
        assert!(!set.load("shell"), "only pool tools load");
        set.begin_turn();
        set.begin_turn();
        assert!(
            set.is_loaded("weather_lookup"),
            "two idle turns are allowed"
        );
        assert_eq!(set.begin_turn(), vec!["weather_lookup".to_owned()]);
        assert!(set.loaded_specs().is_empty());

        set.load("web_fetch");
        set.begin_turn();
        set.begin_turn();
        set.touch("web_fetch");
        set.begin_turn();
        set.begin_turn();
        assert!(set.is_loaded("web_fetch"), "use resets the idle count");
    }

    #[test]
    fn hints_preload_by_name_then_search() {
        let mut set = ActiveToolSet::new(pool(), 3);
        set.begin_turn();
        let mut loaded = set.preload_hints(&["web".into(), "forecast".into(), "".into()]);
        loaded.sort();
        assert_eq!(loaded, ["weather_lookup", "web_fetch", "web_search"]);
        set.reset();
        assert!(set.loaded().is_empty());
    }
}
