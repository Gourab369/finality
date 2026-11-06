use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::tool::{Tier, Tool, ToolSpec};

/// Tools that appear at runtime, such as those of MCP servers.
#[async_trait]
pub trait ToolSource: Send + Sync {
    /// The tools known now. May change between calls.
    fn tools(&self) -> Vec<Arc<dyn Tool>>;
    /// Learns tools not known yet (may start servers). Returns problems to report.
    async fn discover(&self, cancel: &CancellationToken) -> Vec<String>;
    /// Groups whose tools are not known until `discover` runs, for the system prompt.
    fn undiscovered(&self) -> Vec<String>;
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
    sources: Vec<Arc<dyn ToolSource>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry with every built-in tool from this crate.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r.register(Arc::new(crate::read::ReadFile));
        r.register(Arc::new(crate::search::SearchFiles));
        r.register(Arc::new(crate::edit::EditFile));
        r.register(Arc::new(crate::shell::Shell));
        r
    }

    /// Replaces any tool with the same name.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_owned(), tool);
    }

    /// Registers `web_search`, `web_fetch` and the `browser_*` tools (all deferred).
    pub fn register_web(
        &mut self,
        web: &finality_core::config::WebConfig,
        browser: &Arc<crate::browser::BrowserSession>,
    ) {
        self.register(Arc::new(crate::web::WebSearch::new(web.clone())));
        self.register(Arc::new(crate::web::WebFetch::new(web)));
        for tool in crate::browser::BrowserTool::all(browser) {
            self.register(Arc::new(tool));
        }
    }

    /// Built-in and registered tools win over source tools with the same name.
    pub fn add_source(&mut self, source: Arc<dyn ToolSource>) {
        self.sources.push(source);
    }

    pub fn sources(&self) -> &[Arc<dyn ToolSource>] {
        &self.sources
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned().or_else(|| {
            self.sources
                .iter()
                .flat_map(|s| s.tools())
                .find(|t| t.name() == name)
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    pub fn specs(&self, filter: impl Fn(&dyn Tool) -> bool) -> Vec<ToolSpec> {
        let mut specs: Vec<ToolSpec> = self
            .tools
            .values()
            .filter(|t| filter(t.as_ref()))
            .map(|t| t.spec())
            .collect();
        for t in self.sources.iter().flat_map(|s| s.tools()) {
            if filter(t.as_ref()) && !self.tools.contains_key(t.name()) {
                specs.push(t.spec());
            }
        }
        specs
    }

    pub fn core_specs(&self) -> Vec<ToolSpec> {
        self.specs(|t| t.tier() == Tier::Core)
    }
}
