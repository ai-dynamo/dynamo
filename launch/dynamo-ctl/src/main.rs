use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table, Wrap},
};
use std::io::stdout;
use std::time::Duration;

// ASCII Banner - converted slightly for easier use in Rust string literal
const DYNAMO_FIGLET: &str = r#"
██████╗ ██╗   ██╗███╗   ██╗ █████╗ ███╗   ███╗ ██████╗ 
██╔══██╗╚██╗ ██╔╝████╗  ██║██╔══██╗████╗ ████║██╔═══██╗
██║  ██║ ╚████╔╝ ██╔██╗ ██║███████║██╔████╔██║██║   ██║
██║  ██║  ╚██╔╝  ██║╚██╗██║██╔══██║██║╚██╔╝██║██║   ██║
██████╔╝   ██║   ██║ ╚████║██║  ██║██║ ╚═╝ ██║╚██████╔╝
╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝
"#;

// Placeholder color constants (adjust as needed for terminal compatibility)
// Using basic ANSI colors first, can refine to RGB later
const BASE_BG: Color = Color::Black;
const PRIMARY_ACCENT: Color = Color::Cyan;
const TEXT_COLOR: Color = Color::White;
const SECONDARY_ACCENT_1: Color = Color::Magenta;
const SECONDARY_ACCENT_2: Color = Color::Green;
const ERROR_COLOR: Color = Color::Red;

enum CurrentView {
    NamespaceInput,
    Dashboard,
}

struct App {
    current_view: CurrentView,
    namespace_input: String,
    selected_namespace: Option<String>,
    // Add more state fields later (e.g., loaded_state, status_message)
}

impl Default for App {
    fn default() -> Self {
        Self {
            current_view: CurrentView::NamespaceInput,
            namespace_input: String::new(),
            selected_namespace: None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run it
    let app = App::default();
    let res = run_app(&mut terminal, app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("Error: {err:?}");
        return Err(err);
    }

    Ok(())
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> Result<()> {
    loop {
        terminal.draw(|f| match app.current_view {
            CurrentView::NamespaceInput => ui_namespace_input(f, &app),
            CurrentView::Dashboard => ui_dashboard(f, &app),
        })?;

        // Basic event handling (non-blocking)
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match app.current_view {
                    CurrentView::NamespaceInput => {
                        // Namespace Input specific key handling
                        match key.code {
                            KeyCode::Enter => {
                                if !app.namespace_input.is_empty() {
                                    // TODO: Validate namespace properly (check if state file exists)
                                    app.selected_namespace = Some(app.namespace_input.clone());
                                    app.current_view = CurrentView::Dashboard;
                                    // TODO: Load actual state here
                                } else {
                                    // TODO: Show error message in UI
                                }
                            }
                            KeyCode::Char(c) => {
                                app.namespace_input.push(c);
                            }
                            KeyCode::Backspace => {
                                app.namespace_input.pop();
                            }
                            KeyCode::Esc => return Ok(()),
                            _ => {}
                        }
                    }
                    CurrentView::Dashboard => {
                        // Dashboard specific key handling
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                // TODO: Maybe add confirmation
                                return Ok(());
                            }
                            // TODO: Add keys for Add ('a'), Remove ('r'), etc.
                            _ => {}
                        }
                    }
                }
            }
        }
        // TODO: Add async task handling here if needed (e.g., for background loading)
    }
    // Note: The loop only exits via Esc/q or error currently.
    // The Enter key now transitions the view instead of breaking.
}

fn ui_namespace_input(f: &mut Frame, app: &App) {
    let size = f.size();

    // Base layer
    f.render_widget(Block::default().bg(BASE_BG), size);

    // Use a centered vertical layout for the initial screen
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(30), // Top margin
            Constraint::Min(10),        // Banner height (adjust as needed)
            Constraint::Length(3),      // Input box
            Constraint::Percentage(30), // Bottom margin
        ])
        .split(f.size());

    // --- Banner --- (Using Paragraph for now, FIGfont needs more setup)
    let banner = Paragraph::new(DYNAMO_FIGLET)
        .style(Style::default().fg(PRIMARY_ACCENT))
        .alignment(Alignment::Center);
    f.render_widget(banner, chunks[1]);

    // --- Namespace Input --- //
    let input_label = Span::styled("Enter Namespace: ", Style::default().fg(TEXT_COLOR));
    let input_text = Span::styled(
        &app.namespace_input,
        Style::default()
            .fg(PRIMARY_ACCENT)
            .add_modifier(Modifier::BOLD),
    );
    let input_paragraph = Paragraph::new(Line::from(vec![input_label, input_text]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Namespace")
                .border_style(Style::default().fg(PRIMARY_ACCENT)),
        )
        .alignment(Alignment::Center);

    // Render the input paragraph in the center chunk
    f.render_widget(input_paragraph, chunks[2]);

    // Show cursor in the input area (approximation)
    f.set_cursor(
        chunks[2].x + (chunks[2].width / 2) - (17 / 2)
            + 17
            + app.namespace_input.chars().count() as u16,
        chunks[2].y + 1,
    );
}

fn ui_dashboard(f: &mut Frame, app: &App) {
    // Overall layout (Header, Main Area, Footer)
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Main Area
            Constraint::Length(2), // Footer
        ])
        .split(f.size());

    // Split main area horizontally (Left: Components + GPU/Metrics, Right: Raw State)
    let main_area_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55), // Left side slightly wider now
            Constraint::Percentage(45), // Right side
        ])
        .split(main_chunks[1]);

    // Split left main area vertically (Components, GPU Topology & Metrics)
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50), // Components Table
            Constraint::Percentage(50), // GPU Topology & Metrics
        ])
        .split(main_area_chunks[0]);

    // --- Header ---
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " DYNAMO CTL ",
            Style::default().fg(BASE_BG).bg(PRIMARY_ACCENT).bold(),
        ),
        Span::styled(" :: ", Style::default().fg(PRIMARY_ACCENT)),
        Span::styled(
            app.selected_namespace.as_deref().unwrap_or("N/A"),
            Style::default().fg(TEXT_COLOR).bold(),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(PRIMARY_ACCENT)),
    )
    .alignment(Alignment::Left);
    f.render_widget(title, main_chunks[0]);

    // --- Components Panel (Mock Data - Enhanced) ---
    let component_rows = vec![
        // Scalable Components (Mocked initial state)
        Row::new(vec![
            Cell::from("VllmWorker").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("1").style(Style::default().fg(SECONDARY_ACCENT_2).bold()), // Count
            Cell::from("Running").style(Style::default().fg(SECONDARY_ACCENT_2)),  // Status
            Cell::from("1 GPU").style(Style::default().fg(SECONDARY_ACCENT_1)),    // Config
            Cell::from("[0]").style(Style::default().fg(SECONDARY_ACCENT_1).italic()), // Assigned GPUs (Mock)
        ]),
        Row::new(vec![
            Cell::from("PrefillWorker").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("1").style(Style::default().fg(SECONDARY_ACCENT_2).bold()),
            Cell::from("Running").style(Style::default().fg(SECONDARY_ACCENT_2)),
            Cell::from("1 GPU").style(Style::default().fg(SECONDARY_ACCENT_1)),
            Cell::from("[1]").style(Style::default().fg(SECONDARY_ACCENT_1).italic()), // Assigned GPUs (Mock)
        ]),
        // Static/Other Components (from disagg_router.py graph)
        Row::new(vec![
            Cell::from("Frontend").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("-"),
            Cell::from("Active").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("-"),
            Cell::from("-"), // Add empty cell for GPU ID column
        ]),
        Row::new(vec![
            Cell::from("Processor").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("-"),
            Cell::from("Active").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Router").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("-"),
            Cell::from("Active").style(Style::default().fg(TEXT_COLOR)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
    ];

    let component_widths = [
        Constraint::Percentage(28), // Component Name
        Constraint::Percentage(10), // Count
        Constraint::Percentage(20), // Status
        Constraint::Percentage(20), // Config
        Constraint::Percentage(22), // Assigned GPUs
    ];

    let components_table = Table::new(component_rows, &component_widths)
        .header(
            Row::new(vec!["Component", "Count", "Status", "Config", "GPU IDs"]) // Added header
                .style(
                    Style::default()
                        .fg(PRIMARY_ACCENT)
                        .add_modifier(Modifier::BOLD),
                )
                .bottom_margin(1),
        )
        .block(
            Block::default()
                .title(" Components ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(PRIMARY_ACCENT)),
        )
        .column_spacing(1); // Reduced spacing slightly
    f.render_widget(components_table, left_chunks[0]);

    // --- GPU Topology & Metrics Panel (Mock Data) ---
    let gpu_panel_block = Block::default()
        .title(" GPU Topology & Metrics ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PRIMARY_ACCENT));
    let gpu_panel_inner_area = gpu_panel_block.inner(left_chunks[1]);
    f.render_widget(gpu_panel_block, left_chunks[1]);

    // *** CORRECTED Layout within the GPU panel ***
    let gpu_panel_chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(1), // GPU 0 Info          -> Index 0
            Constraint::Length(1), // GPU 0 Gauge         -> Index 1
            Constraint::Length(1), // GPU 1 Info          -> Index 2
            Constraint::Length(1), // GPU 1 Gauge         -> Index 3
            Constraint::Length(1), // Spacer              -> Index 4
            Constraint::Length(1), // KV Cache Title      -> Index 5
            Constraint::Length(1), // KV Cache Gauge      -> Index 6
            Constraint::Length(1), // Prefill Queue Title -> Index 7
            Constraint::Length(1), // Prefill Queue Gauge -> Index 8
            Constraint::Min(0),    // Filler              -> Index 9
        ])
        .split(gpu_panel_inner_area);

    // Mock GPU Data
    let gpus = vec![
        (0, Some("VllmWorker_0"), 75),
        (1, Some("PrefillWorker_0"), 25),
    ];

    // *** CORRECTED Indices for Rendering ***

    // Render GPU 0 Info & Gauge
    let gpu0_info = Line::from(vec![
        Span::styled("GPU 0:", Style::default().fg(PRIMARY_ACCENT).bold()),
        Span::styled(" Assign: ", Style::default().fg(TEXT_COLOR)),
        Span::styled(
            gpus[0].1.unwrap_or("None"),
            Style::default().fg(SECONDARY_ACCENT_1),
        ),
    ]);
    f.render_widget(Paragraph::new(gpu0_info), gpu_panel_chunks[0]); // Index 0
    let gpu0_gauge = Gauge::default()
        .percent(gpus[0].2)
        .gauge_style(Style::default().fg(SECONDARY_ACCENT_1).bg(Color::DarkGray))
        .label(format!("{}%", gpus[0].2));
    f.render_widget(gpu0_gauge, gpu_panel_chunks[1]); // Index 1

    // Render GPU 1 Info & Gauge
    let gpu1_info = Line::from(vec![
        Span::styled("GPU 1:", Style::default().fg(PRIMARY_ACCENT).bold()),
        Span::styled(" Assign: ", Style::default().fg(TEXT_COLOR)),
        Span::styled(
            gpus[1].1.unwrap_or("None"),
            Style::default().fg(SECONDARY_ACCENT_1),
        ),
    ]);
    f.render_widget(Paragraph::new(gpu1_info), gpu_panel_chunks[2]); // Index 2
    let gpu1_gauge = Gauge::default()
        .percent(gpus[1].2)
        .gauge_style(Style::default().fg(SECONDARY_ACCENT_1).bg(Color::DarkGray))
        .label(format!("{}%", gpus[1].2));
    f.render_widget(gpu1_gauge, gpu_panel_chunks[3]); // Index 3

    // Index 4 is the spacer, render nothing or `Paragraph::new("")` if needed

    // Render Metrics Gauges
    let kv_cache_perc = 65;
    let prefill_queue_size = 1;
    let max_prefill_queue_size = 2;
    let prefill_perc = ((prefill_queue_size as f32 / max_prefill_queue_size as f32) * 100.0) as u16;

    f.render_widget(
        Paragraph::new("Agg. KV Cache:").style(Style::default().fg(PRIMARY_ACCENT)),
        gpu_panel_chunks[5],
    ); // Index 5
    let kv_gauge = Gauge::default()
        .percent(kv_cache_perc)
        .gauge_style(Style::default().fg(SECONDARY_ACCENT_2).bg(Color::DarkGray))
        .label(format!("{}%", kv_cache_perc));
    f.render_widget(kv_gauge, gpu_panel_chunks[6]); // Index 6

    f.render_widget(
        Paragraph::new("Prefill Queue:").style(Style::default().fg(PRIMARY_ACCENT)),
        gpu_panel_chunks[7],
    ); // Index 7
    let prefill_gauge = Gauge::default()
        .percent(prefill_perc)
        .gauge_style(Style::default().fg(SECONDARY_ACCENT_2).bg(Color::DarkGray))
        .label(format!("{}/{}", prefill_queue_size, max_prefill_queue_size));
    f.render_widget(prefill_gauge, gpu_panel_chunks[8]); // Index 8

    // --- Raw State View (Mock Data - Adjusted for GPU list) ---
    let mock_state_json = format!(
        r#"{{
  "namespace": "{}",
  "components": {{
    "{}_VllmWorker_0": {{
      "component_type": "VllmWorker",
      "status": "Running",
      "allocated_gpus": [0] // List format
    }},
    "{}_PrefillWorker_0": {{
      "component_type": "PrefillWorker",
      "status": "Running",
      "allocated_gpus": [1] // List format
    }}
  }},
  "config": {{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "VllmWorker": {{ "tensor_parallel_size": 1 }},
    "PrefillWorker": {{}},
    "max_gpu_budget": null // Example, not in state file yet
  }}
}}"#,
        app.selected_namespace.as_deref().unwrap_or("mock_ns"),
        app.selected_namespace.as_deref().unwrap_or("mock_ns"),
        app.selected_namespace.as_deref().unwrap_or("mock_ns")
    );

    let state_paragraph = Paragraph::new(mock_state_json)
        .style(Style::default().fg(TEXT_COLOR))
        .block(
            Block::default()
                .title(" Raw State (~/.dynamo/state/...) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(PRIMARY_ACCENT)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(state_paragraph, main_area_chunks[1]);

    // --- Footer ---
    let keybinds = Line::from(vec![
        Span::styled(
            "[Q]",
            Style::default().fg(BASE_BG).bg(PRIMARY_ACCENT).bold(),
        ),
        Span::raw("uit "),
        Span::styled(
            "[A]",
            Style::default().fg(BASE_BG).bg(PRIMARY_ACCENT).bold(),
        ),
        Span::raw("dd "),
        Span::styled(
            "[R]",
            Style::default().fg(BASE_BG).bg(PRIMARY_ACCENT).bold(),
        ),
        Span::raw("emove "),
        Span::styled(
            "[N]",
            Style::default().fg(BASE_BG).bg(PRIMARY_ACCENT).bold(),
        ),
        Span::raw("amespace "),
    ]);
    let status_msg = Line::from(vec![
        Span::styled(
            " OK ",
            Style::default().fg(BASE_BG).bg(SECONDARY_ACCENT_2).bold(),
        ),
        Span::raw(" Loaded dashboard."),
    ]);

    let footer_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(main_chunks[2]);

    let footer_keys = Paragraph::new(keybinds)
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(PRIMARY_ACCENT)),
        )
        .alignment(Alignment::Left);
    let footer_status = Paragraph::new(status_msg)
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(PRIMARY_ACCENT)),
        )
        .alignment(Alignment::Right);

    f.render_widget(footer_keys, footer_chunks[0]);
    f.render_widget(footer_status, footer_chunks[1]);
}
