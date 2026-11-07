//! HTML to Markdown for `web_fetch`, and DuckDuckGo result parsing (DESIGN §9.2).

use scraper::{ElementRef, Html, Node, Selector};
use url::Url;

/// Never rendered: code, media, page chrome and form controls.
const SKIP: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "head", "iframe", "canvas", "object",
    "embed", "nav", "aside", "footer", "form", "button", "select", "input", "textarea", "dialog",
    "img", "picture", "video", "audio", "map",
];
/// ARIA roles of page chrome.
const SKIP_ROLES: &[&str] = &[
    "navigation",
    "banner",
    "contentinfo",
    "complementary",
    "search",
    "dialog",
    "menu",
    "menubar",
];
/// Start on a line of their own.
const BLOCK: &[&str] = &[
    "div",
    "section",
    "header",
    "figure",
    "figcaption",
    "details",
    "summary",
    "address",
    "dl",
    "dt",
    "dd",
    "center",
    "fieldset",
    "li",
    "tr",
];
/// A `main` or `article` shorter than this is probably not the page's content.
const MIN_CONTENT_CHARS: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Clone, Copy)]
struct Cx<'a> {
    base: Option<&'a Url>,
    /// Inside the page's main content, where a `header` is part of the text.
    content: bool,
}

fn hidden(el: &ElementRef<'_>) -> bool {
    let v = el.value();
    v.attr("hidden").is_some()
        || v.attr("aria-hidden") == Some("true")
        || v.attr("role").is_some_and(|r| SKIP_ROLES.contains(&r))
        || v.attr("style")
            .is_some_and(|s| s.replace(' ', "").contains("display:none"))
}

/// Appends text with whitespace runs collapsed, keeping one space at either edge.
fn push_inline(out: &mut String, t: &str) {
    let text = collapse(t);
    let gap = !out.is_empty() && !out.ends_with([' ', '\n']);
    if text.is_empty() {
        if gap && !t.is_empty() {
            out.push(' ');
        }
        return;
    }
    if gap && t.starts_with(char::is_whitespace) {
        out.push(' ');
    }
    out.push_str(&text);
    if t.ends_with(char::is_whitespace) {
        out.push(' ');
    }
}

fn children(el: ElementRef<'_>, cx: Cx<'_>) -> String {
    let mut out = String::new();
    for child in el.children() {
        match child.value() {
            Node::Text(t) => push_inline(&mut out, t),
            Node::Element(_) => {
                if let Some(c) = ElementRef::wrap(child) {
                    out.push_str(&render(c, cx));
                }
            }
            _ => {}
        }
    }
    out
}

fn block(body: &str) -> String {
    let b = body.trim();
    if b.is_empty() {
        String::new()
    } else {
        format!("\n\n{b}\n\n")
    }
}

fn line(body: &str) -> String {
    let b = body.trim();
    if b.is_empty() {
        String::new()
    } else {
        format!("\n{b}\n")
    }
}

fn code_lang(pre: ElementRef<'_>) -> String {
    let code = pre
        .children()
        .filter_map(ElementRef::wrap)
        .filter(|c| c.value().name() == "code");
    std::iter::once(pre)
        .chain(code)
        .flat_map(|e| e.value().classes().map(str::to_owned).collect::<Vec<_>>())
        .find_map(|c| {
            c.strip_prefix("language-")
                .or_else(|| c.strip_prefix("lang-"))
                .map(str::to_owned)
        })
        .unwrap_or_default()
}

fn link(el: ElementRef<'_>, cx: Cx<'_>) -> String {
    let label = collapse(&children(el, cx));
    let Some(href) = el.value().attr("href").map(str::trim) else {
        return label;
    };
    if label.is_empty() || href.starts_with('#') {
        return label;
    }
    let url = match cx.base {
        Some(base) => base.join(href).ok(),
        None => Url::parse(href).ok(),
    };
    match url.filter(|u| matches!(u.scheme(), "http" | "https")) {
        Some(u) if u.as_str() != label => format!("[{label}]({u})"),
        _ => label,
    }
}

fn list(el: ElementRef<'_>, cx: Cx<'_>, ordered: bool) -> String {
    let mut items: Vec<String> = Vec::new();
    for li in el
        .children()
        .filter_map(ElementRef::wrap)
        .filter(|c| c.value().name() == "li" && !hidden(c))
    {
        let body = tidy(&children(li, cx));
        if body.is_empty() {
            continue;
        }
        let marker = if ordered {
            format!("{}. ", items.len() + 1)
        } else {
            "- ".to_owned()
        };
        let pad = " ".repeat(marker.len());
        let mut item = String::new();
        for (i, l) in body.lines().enumerate() {
            if i == 0 {
                item.push_str(&marker);
            } else {
                item.push('\n');
                if !l.is_empty() {
                    item.push_str(&pad);
                }
            }
            item.push_str(l);
        }
        items.push(item);
    }
    block(&items.join("\n"))
}

fn table(el: ElementRef<'_>, cx: Cx<'_>) -> String {
    let (Ok(tr), Ok(cell), Ok(th)) = (
        Selector::parse("tr"),
        Selector::parse("th, td"),
        Selector::parse("th"),
    ) else {
        return String::new();
    };
    let mut rows: Vec<String> = Vec::new();
    for row in el.select(&tr) {
        let cells: Vec<String> = row
            .select(&cell)
            .map(|c| collapse(&children(c, cx)).replace('|', "\\|"))
            .collect();
        if cells.iter().all(String::is_empty) {
            continue;
        }
        rows.push(format!("| {} |", cells.join(" | ")));
        if rows.len() == 1 && row.select(&th).next().is_some() {
            rows.push(format!("|{}|", vec![" --- "; cells.len()].join("|")));
        }
    }
    block(&rows.join("\n"))
}

fn render(el: ElementRef<'_>, cx: Cx<'_>) -> String {
    let name = el.value().name();
    if SKIP.contains(&name) || hidden(&el) || (name == "header" && !cx.content) {
        return String::new();
    }
    match name {
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
            let level = name[1..].parse::<usize>().unwrap_or(1);
            let text = collapse(&children(el, cx));
            if text.is_empty() {
                String::new()
            } else {
                format!("\n\n{} {text}\n\n", "#".repeat(level))
            }
        }
        "p" => block(&children(el, cx)),
        "br" => "\n".into(),
        "hr" => "\n\n---\n\n".into(),
        "pre" => {
            let code: String = el.text().collect();
            let code = code.trim_matches('\n');
            if code.trim().is_empty() {
                String::new()
            } else {
                format!("\n\n```{}\n{code}\n```\n\n", code_lang(el))
            }
        }
        "code" | "kbd" | "samp" => {
            let t = collapse(&el.text().collect::<String>());
            if t.is_empty() {
                String::new()
            } else if t.contains('`') {
                format!("`` {t} ``")
            } else {
                format!("`{t}`")
            }
        }
        "a" => link(el, cx),
        "ul" | "ol" => list(el, cx, name == "ol"),
        "blockquote" => {
            let body = tidy(&children(el, cx));
            let quoted: Vec<String> = body
                .lines()
                .map(|l| {
                    if l.is_empty() {
                        ">".to_owned()
                    } else {
                        format!("> {l}")
                    }
                })
                .collect();
            block(&quoted.join("\n"))
        }
        "table" => table(el, cx),
        "main" | "article" => line(&children(
            el,
            Cx {
                content: true,
                ..cx
            },
        )),
        n if BLOCK.contains(&n) => line(&children(el, cx)),
        _ => children(el, cx),
    }
}

/// Trailing spaces and runs of blank lines removed, and the stray space that inline text leaves at a line start.
/// Code fences are kept as they are.
fn tidy(raw: &str) -> String {
    let mut out = String::new();
    let mut blank = false;
    let mut fence = false;
    for raw_line in raw.lines() {
        let mut line = raw_line.trim_end();
        let is_fence = line.trim_start().starts_with("```");
        if fence && !is_fence {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if is_fence {
            fence = !fence;
        }
        if line.starts_with(' ') && !line.starts_with("  ") {
            line = &line[1..];
        }
        if line.trim().is_empty() {
            blank = !out.is_empty();
            continue;
        }
        if blank {
            out.push('\n');
            blank = false;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_owned()
}

/// The page title and its content as Markdown: headings, paragraphs, lists, tables, code blocks and links (made
/// absolute against `base`). Navigation, headers, footers, sidebars, forms and media are left out, and when the
/// page marks its content with a single `main` or `article`, only that is rendered.
pub fn html_to_markdown(html: &str, base: Option<&Url>) -> (Option<String>, String) {
    let doc = Html::parse_document(html);
    let first = |sel: &str| {
        Selector::parse(sel)
            .ok()
            .and_then(|s| {
                doc.select(&s)
                    .next()
                    .map(|e| collapse(&e.text().collect::<String>()))
            })
            .filter(|t| !t.is_empty())
    };
    let title = first("title").or_else(|| first("h1"));
    let only = |sel: &str| {
        let s = Selector::parse(sel).ok()?;
        let mut found = doc.select(&s);
        let one = found.next()?;
        found.next().is_none().then_some(one)
    };
    let page = only("body").unwrap_or_else(|| doc.root_element());
    let whole = Cx {
        base,
        content: false,
    };
    if let Some(main) = only("main")
        .or_else(|| only("[role=main]"))
        .or_else(|| only("article"))
    {
        let text = tidy(&children(
            main,
            Cx {
                content: true,
                ..whole
            },
        ));
        if text.chars().count() >= MIN_CONTENT_CHARS {
            return (title, text);
        }
    }
    (title, tidy(&children(page, whole)))
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
    {
        // `/y.js` links are ads.
        return url
            .query_pairs()
            .find(|(k, _)| k == "uddg")
            .map(|(_, target)| target.into_owned());
    }
    matches!(url.scheme(), "http" | "https").then(|| url.to_string())
}

/// DuckDuckGo's bot check, which it serves instead of results when it throttles automated searches.
pub fn duckduckgo_blocked(html: &str) -> bool {
    html.contains("anomaly-modal")
        || html.contains("challenge-form")
        || html.contains("bots use DuckDuckGo too")
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

/// Results of DuckDuckGo Lite: a table where a row with the link is followed by a row with the snippet.
pub fn parse_duckduckgo_lite(html: &str, max: usize) -> Vec<SearchHit> {
    let doc = Html::parse_document(html);
    let (Ok(row), Ok(link), Ok(snippet)) = (
        Selector::parse("tr"),
        Selector::parse("a.result-link"),
        Selector::parse("td.result-snippet"),
    ) else {
        return Vec::new();
    };
    let mut hits: Vec<SearchHit> = Vec::new();
    // Set while the rows belong to an ad or a duplicate.
    let mut skipping = false;
    for r in doc.select(&row) {
        if let Some(a) = r.select(&link).next() {
            if hits.len() >= max {
                break;
            }
            let sponsored = r.value().classes().any(|c| c.contains("sponsored"));
            let url = a.value().attr("href").and_then(result_url);
            skipping = true;
            if let Some(url) = url.filter(|u| !sponsored && !hits.iter().any(|h| &h.url == u)) {
                hits.push(SearchHit {
                    title: collapse(&a.text().collect::<String>()),
                    url,
                    snippet: String::new(),
                });
                skipping = false;
            }
        } else if !skipping
            && let Some(s) = r.select(&snippet).next()
            && let Some(last) = hits.last_mut()
            && last.snippet.is_empty()
        {
            last.snippet = collapse(&s.text().collect::<String>());
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_html_to_markdown() {
        let html = r#"<html><head><title> Zig  build </title><style>p{color:red}</style></head>
            <body><nav><a href="/">Home</a> <a href="/blog">Blog</a></nav>
            <header><a href="/">Site logo</a></header>
            <script>alert('x')</script><h1>Build system</h1>
            <p>Run <code>zig build</code> to   compile.</p>
            <ul><li>one</li><li>two<ul><li>nested</li></ul></li></ul>
            <ol><li>first</li><li>second</li></ol>
            <pre class="language-zig">fn main() {

    run();
}</pre>
            <table><tr><th>Step</th><th>What</th></tr><tr><td>1</td><td>build | test</td></tr></table>
            <blockquote><p>Quoted</p></blockquote>
            <p>See <a href="/learn/">the docs</a> and <a href="javascript:void(0)">this</a>.</p>
            <div hidden>invisible</div><aside>Related posts</aside>
            <footer>Copyright</footer></body></html>"#;
        let base = Url::parse("https://ziglang.org/news/").unwrap();
        let (title, md) = html_to_markdown(html, Some(&base));
        assert_eq!(title.as_deref(), Some("Zig build"));
        for gone in [
            "alert",
            "color:red",
            "Home",
            "Site logo",
            "invisible",
            "Related",
            "Copyright",
        ] {
            assert!(!md.contains(gone), "{gone} in:\n{md}");
        }
        assert!(
            md.starts_with("# Build system\n\nRun `zig build` to compile."),
            "{md}"
        );
        assert!(md.contains("- one\n- two\n\n  - nested"), "{md}");
        assert!(md.contains("1. first\n2. second"), "{md}");
        assert!(
            md.contains("```zig\nfn main() {\n\n    run();\n}\n```"),
            "{md}"
        );
        assert!(
            md.contains("| Step | What |\n| --- | --- |\n| 1 | build \\| test |"),
            "{md}"
        );
        assert!(md.contains("> Quoted"), "{md}");
        assert!(
            md.contains("See [the docs](https://ziglang.org/learn/) and this."),
            "{md}"
        );
    }

    #[test]
    fn renders_only_the_main_content_when_marked() {
        let text = "Real content. ".repeat(20);
        let html = format!(
            "<body><div class=menu>Menu entries</div><main><header><h1>Post</h1></header><p>{text}</p></main></body>"
        );
        let (_, md) = html_to_markdown(&html, None);
        assert!(md.starts_with("# Post\n\nReal content."), "{md}");
        assert!(!md.contains("Menu entries"));

        let tiny = "<body><p>Body text that matters.</p><main>Loading…</main></body>";
        assert!(
            html_to_markdown(tiny, None)
                .1
                .contains("Body text that matters.")
        );
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
        assert!(!duckduckgo_blocked(html));
    }

    #[test]
    fn parses_duckduckgo_lite_results() {
        let html = r#"<table>
          <tr class="result-sponsored"><td>1.</td><td><a class="result-link" href="https://duckduckgo.com/y.js?ad=1">Ad</a></td></tr>
          <tr class="result-sponsored"><td></td><td class="result-snippet">Buy now</td></tr>
          <tr><td>1.</td><td><a rel="nofollow" class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fziglang.org%2F">Zig</a></td></tr>
          <tr><td></td><td class="result-snippet">A general-purpose language.</td></tr>
          <tr><td></td><td><span class="link-text">ziglang.org</span></td></tr>
          <tr><td>2.</td><td><a class="result-link" href="https://example.com/">Example</a></td></tr>
        </table>"#;
        let hits = parse_duckduckgo_lite(html, 8);
        assert_eq!(
            hits.iter().map(|h| h.url.as_str()).collect::<Vec<_>>(),
            ["https://ziglang.org/", "https://example.com/"]
        );
        assert_eq!(hits[0].snippet, "A general-purpose language.");
        assert_eq!(parse_duckduckgo_lite(html, 1).len(), 1);
    }

    #[test]
    fn recognizes_the_bot_check() {
        assert!(duckduckgo_blocked(
            r#"<div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>"#
        ));
    }
}
