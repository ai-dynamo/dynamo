use dynamo_ctl::app_state::{AppState, SharedState};
use dynamo_ctl::etcd_watcher::watch_etcd;

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem},
    Frame, Terminal,
};
use std::io::stdout;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let state: SharedState = Arc::new(RwLock::new(AppState::default()));
    let watcher_state = state.clone();

    tokio::spawn(async move {
        if let Err(e) = watch_etcd(watcher_state).await {
            eprintln!("Watcher error: {e:?}");
        }
    });

    let res = run(&mut terminal, state).await;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("Error: {err:?}");
    }

    Ok(())
}

async fn run<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: SharedState,
) -> Result<()> {
    loop {
        let snapshot = state.read().await.clone();
        terminal.draw(|f| ui(f, &snapshot))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    _ => {}
                }
            }
        }
    }
}

fn ui(f: &mut Frame, state: &AppState) {
    let size = f.size();

    let items: Vec<ListItem> = state
        .entries
        .values()
        .map(|e| ListItem::new(format!("{} = {}", e.key, e.value)))
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .title(" /dynamo/** keys ")
                .borders(Borders::ALL)
                .border_style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
        )
        .highlight_style(Style::default().bg(Color::Blue));

    f.render_widget(list, size);
}
