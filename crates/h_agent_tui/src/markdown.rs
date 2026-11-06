//! Minimal markdown for chat: headings, fenced code, bullet lists, `inline code` and **bold**.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

fn code_style() -> Style {
    Style::default().fg(Color::LightYellow)
}

fn inline(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    let (mut code, mut bold) = (false, false);
    let style = |code: bool, bold: bool| {
        if code {
            code_style()
        } else if bold {
            base.add_modifier(Modifier::BOLD)
        } else {
            base
        }
    };
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let toggle = if c == '`' {
            Some(true)
        } else if c == '*' && !code && chars.peek() == Some(&'*') {
            chars.next();
            Some(false)
        } else {
            None
        };
        match toggle {
            Some(is_code) => {
                if !buf.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut buf), style(code, bold)));
                }
                if is_code {
                    code = !code;
                } else {
                    bold = !bold;
                }
            }
            None => buf.push(c),
        }
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, style(code, bold)));
    }
    spans
}

pub fn render(text: &str, base: Style) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code = false;
    for raw in text.split('\n') {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            lines.push(Line::from(Span::styled(
                raw.to_owned(),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }
        if in_code {
            lines.push(Line::from(Span::styled(raw.to_owned(), code_style())));
            continue;
        }
        let hashes = trimmed.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
            let style = base.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
            lines.push(Line::from(Span::styled(
                trimmed[hashes + 1..].to_owned(),
                style,
            )));
            continue;
        }
        if let Some(rest) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            let indent = " ".repeat(raw.len() - trimmed.len());
            let mut spans = vec![Span::styled(format!("{indent}• "), base)];
            spans.extend(inline(rest, base));
            lines.push(Line::from(spans));
            continue;
        }
        lines.push(Line::from(inline(raw, base)));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn renders_blocks_and_inline() {
        let lines = render(
            "# Title\n- item `x`\n```rust\nlet a = **b**;\n```\nplain **bold**",
            Style::default(),
        );
        assert_eq!(
            text(&lines),
            vec![
                "Title",
                "• item x",
                "```rust",
                "let a = **b**;",
                "```",
                "plain bold"
            ]
        );
        assert!(
            lines[0].spans[0]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert_eq!(lines[1].spans[2].style.fg, Some(Color::LightYellow));
        assert!(
            lines[5].spans[1]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
    }
}
