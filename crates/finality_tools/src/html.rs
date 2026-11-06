//! HTML to readable text, and DuckDuckGo result parsing (DESIGN §9.2).

use scraper::{ElementRef, Html, Node, Selector};
use url::Url;

const SKIP: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "head", "iframe", "canvas", "object",
];
/// Separated from their surroundings by a blank line.
const PARAGRAPH: &[&str] = &[
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "blockquote",
    "table",
    "ul",
    "ol",
    "dl",
    "figure",
    "form",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "aside",
    "main",
    "details",
    "hr",
];
/// Start on their own line.
const LINE: &[&str] = &["div", "li", "tr", "br", "dt", "dd", "figcaption", "summary"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn walk(el: ElementRef<'_>, out: &mut String, in_pre: bool) {
    let name = el.value().name();
    if SKIP.contains(&name) {
        return;
    }
    let paragraph = PARAGRAPH.contains(&name);
    let line = LINE.contains(&name);
    let newline = |out: &mut String| {
        if paragraph {
            out.push_str("\n\n");
        } else if line && !out.ends_with('\n') {
            out.push('\n');
        }
    };
    newline(out);
    if name == "li" {
        out.push_str("- ");
    }
    let pre = in_pre || name == "pre";
    for child in el.children() {
        match child.value() {
            Node::Text(t) if pre => out.push_str(t),
            Node::Text(t) => {
                let text = collapse(t);
                let gap = !out.is_empty() && !out.ends_with([' ', '\n']);
                if text.is_empty() {
                    if gap && !t.is_empty() {
                        out.push(' ');
                    }
                    continue;
                }
                if gap && t.starts_with(char::is_whitespace) {
                    out.push(' ');
                }
                out.push_str(&text);
                if t.ends_with(char::is_whitespace) {
                    out.push(' ');
                }
            }
            Node::Element(_) => {
                if let Some(child) = ElementRef::wrap(child) {
                    walk(child, out, pre);
                }
            }
            _ => {}
        }
    }
    if name == "a"
        && let Some(href) = el.value().attr("href").filter(|h| h.starts_with("http"))
    {
        let label = collapse(&el.text().collect::<String>());
        if !label.is_empty() && label != href {
            out.push_str(&format!(" [{href}]"));
        }
    }
    newline(out);
}

/// The page title and its readable text: scripts and styles dropped, one block per line, blank lines collapsed.
pub fn html_to_text(html: &str) -> (Option<String>, String) {
    let doc = Html::parse_document(html);
    let title = Selector::parse("title")
        .ok()
        .and_then(|s| doc.select(&s).next())
        .map(|t| collapse(&t.text().collect::<String>()))
        .filter(|t| !t.is_empty());
    let mut raw = String::new();
    walk(doc.root_element(), &mut raw, false);
    let mut text = String::new();
    let mut blank = false;
    for line in raw.lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank = !text.is_empty();
            continue;
        }
        if blank {
            text.push('\n');
            blank = false;
        }
        text.push_str(line);
        text.push('\n');
    }
    (title, text.trim_end().to_owned())
}

/// Result links point at `/l/?uddg=<target>`; returns the target.
fn result_url(href: &str) -> Option<String> {
    let absolute = if href.starts_with("//") {
        format!("https:{href}")
    } else if href.starts_with('/') {
        format!("https://duckduckgo.com{href}")
    } else {
        href.to_owned()
    };
    let url = Url::parse(&absolute).ok()?;
    if url
        .host_str()
        .is_some_and(|h| h.ends_with("duckduckgo.com"))
        && let Some((_, target)) = url.query_pairs().find(|(k, _)| k == "uddg")
    {
        return Some(target.into_owned());
    }
    matches!(url.scheme(), "http" | "https").then(|| url.to_string())
}

/// Organic results of DuckDuckGo's HTML endpoint; ads are skipped.
pub fn parse_duckduckgo(html: &str, max: usize) -> Vec<SearchHit> {
    let doc = Html::parse_document(html);
    let (Ok(result), Ok(link), Ok(snippet)) = (
        Selector::parse("div.result, div.web-result"),
        Selector::parse("a.result__a"),
        Selector::parse(".result__snippet"),
    ) else {
        return Vec::new();
    };
    let mut hits: Vec<SearchHit> = Vec::new();
    for r in doc.select(&result) {
        if r.value().classes().any(|c| c == "result--ad") {
            continue;
        }
        let Some(a) = r.select(&link).next() else {
            continue;
        };
        let Some(url) = a.value().attr("href").and_then(result_url) else {
            continue;
        };
        if hits.iter().any(|h| h.url == url) {
            continue;
        }
        hits.push(SearchHit {
            title: collapse(&a.text().collect::<String>()),
            url,
            snippet: r
                .select(&snippet)
                .next()
                .map(|s| collapse(&s.text().collect::<String>()))
                .unwrap_or_default(),
        });
        if hits.len() >= max {
            break;
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_html_to_readable_text() {
        let html = r#"<html><head><title> Zig  build </title><style>p{color:red}</style></head>
            <body><script>alert('x')</script><h1>Build system</h1>
            <p>Run <code>zig build</code> to   compile.</p>
            <ul><li>one</li><li>two</li></ul>
            <pre>fn main() {
    run();
}</pre>
            <p>See <a href="https://ziglang.org/learn/">the docs</a>.</p></body></html>"#;
        let (title, text) = html_to_text(html);
        assert_eq!(title.as_deref(), Some("Zig build"));
        assert!(!text.contains("alert") && !text.contains("color:red"));
        assert!(text.contains("Build system\n"));
        assert!(text.contains("Run zig build to compile."), "{text}");
        assert!(text.contains("- one\n- two"));
        assert!(text.contains("fn main() {\n    run();\n}"));
        assert!(text.contains("the docs [https://ziglang.org/learn/]"));
    }

    #[test]
    fn parses_duckduckgo_results() {
        let html = r#"
          <div class="result results_links result--ad"><a class="result__a" href="https://ads.example/">Ad</a></div>
          <div class="result results_links web-result">
            <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fziglang.org%2Flearn%2F&amp;rut=abc">Learn <b>Zig</b></a></h2>
            <a class="result__snippet">The <b>Zig</b> build system.</a>
          </div>
          <div class="result"><a class="result__a" href="https://example.com/b">Second</a></div>"#;
        let hits = parse_duckduckgo(html, 8);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].title, "Learn Zig");
        assert_eq!(hits[0].url, "https://ziglang.org/learn/");
        assert_eq!(hits[0].snippet, "The Zig build system.");
        assert_eq!(hits[1].url, "https://example.com/b");
        assert_eq!(parse_duckduckgo(html, 1).len(), 1);
    }
}
