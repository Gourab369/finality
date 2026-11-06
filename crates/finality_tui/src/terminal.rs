use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, SetTitle, disable_raw_mode, enable_raw_mode,
    supports_keyboard_enhancement,
};
use finality_core::events::{CoreCommand, UiEvent};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use crate::app::App;

const TICK: Duration = Duration::from_millis(100);
/// Keys already in flight when an approval modal appears must not answer it.
const APPROVAL_GRACE: Duration = Duration::from_millis(300);
/// xterm title stack: save the title before setting ours, restore it on exit. Terminals without
/// a title stack ignore these.
#[cfg(not(windows))]
const PUSH_TITLE: &str = "\x1b[22;0t";
#[cfg(not(windows))]
const POP_TITLE: &str = "\x1b[23;0t";

/// Best-effort terminal restore; safe to call more than once (e.g. from a panic hook).
pub fn restore_terminal() {
    let mut out = io::stdout();
    let _ = execute!(out, PopKeyboardEnhancementFlags);
    let _ = execute!(
        out,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    #[cfg(not(windows))]
    let _ = execute!(out, crossterm::style::Print(POP_TITLE));
    let _ = disable_raw_mode();
}

fn is_ctrl_c(k: &KeyEvent) -> bool {
    k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL)
}

struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

fn setup() -> io::Result<(Terminal<CrosstermBackend<Stdout>>, Guard)> {
    enable_raw_mode()?;
    let guard = Guard;
    let mut out = io::stdout();
    #[cfg(not(windows))]
    let _ = execute!(out, crossterm::style::Print(PUSH_TITLE));
    let _ = execute!(out, SetTitle(format!("∞ {}", finality_core::APP_NAME)));
    execute!(
        out,
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture
    )?;
    // Lets terminals that support it report Shift+Enter distinctly; Alt+Enter works everywhere.
    if supports_keyboard_enhancement().unwrap_or(false) {
        let _ = execute!(
            out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
    }
    let terminal = Terminal::new(CrosstermBackend::new(out))?;
    Ok((terminal, guard))
}

/// Runs the UI until the user quits or the core closes its channel.
pub async fn run(
    mut app: App,
    mut ui_rx: mpsc::UnboundedReceiver<UiEvent>,
    cmd_tx: mpsc::UnboundedSender<CoreCommand>,
) -> io::Result<()> {
    let (mut terminal, _guard) = setup()?;
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(TICK);
    let mut keys_blocked_until: Option<Instant> = None;
    loop {
        terminal.draw(|f| app.render(f))?;
        tokio::select! {
            ev = events.next() => match ev {
                Some(Ok(ev)) => {
                    let now = Instant::now();
                    let typed_ahead = matches!(&ev, Event::Key(k) if !is_ctrl_c(k))
                        && app.approval_pending()
                        && keys_blocked_until.is_some_and(|t| now < t);
                    if typed_ahead {
                        continue;
                    }
                    for cmd in app.handle_terminal_event(ev, now) {
                        if cmd_tx.send(cmd).is_err() {
                            return Ok(());
                        }
                    }
                }
                Some(Err(e)) => return Err(e),
                None => return Ok(()),
            },
            ui = ui_rx.recv() => match ui {
                Some(ev) => {
                    let before = app.approval_pending();
                    app.handle_ui_event(ev);
                    while let Ok(more) = ui_rx.try_recv() {
                        app.handle_ui_event(more);
                    }
                    if !before && app.approval_pending() {
                        keys_blocked_until = Some(Instant::now() + APPROVAL_GRACE);
                    }
                }
                None => return Ok(()),
            },
            _ = tick.tick() => app.tick(),
        }
        if app.should_quit() {
            terminal.draw(|f| app.render(f))?;
            return Ok(());
        }
    }
}
