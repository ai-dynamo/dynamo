// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TUI rendering using Ratatui.
//!
//! Renders the application state into a terminal frame with:
//! - Three-column hierarchical view (Namespaces | Components | Endpoints)
//! - NATS status bar
//! - Optional metrics display
//! - Keyboard shortcut hints

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, PaneFocus};
use crate::model::HealthStatus;

/// Render the full UI.
pub fn render(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),       // Main content area
            Constraint::Length(3),     // NATS status bar
            Constraint::Length(3),     // Metrics bar
            Constraint::Length(1),     // Help bar
        ])
        .split(frame.area());

    if app.loading {
        render_loading(frame, chunks[0]);
    } else {
        render_discovery_panes(frame, app, chunks[0]);
    }
    render_nats_bar(frame, app, chunks[1]);
    render_metrics_bar(frame, app, chunks[2]);
    render_help_bar(frame, chunks[3]);
}

fn render_loading(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Dynamo TUI ");
    let text = Paragraph::new("Connecting to ETCD...")
        .style(Style::default().fg(Color::Yellow))
        .block(block);
    frame.render_widget(text, area);
}

fn render_discovery_panes(frame: &mut Frame, app: &App, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(37),
            Constraint::Percentage(38),
        ])
        .split(area);

    render_namespace_list(frame, app, columns[0]);
    render_component_list(frame, app, columns[1]);
    render_endpoint_list(frame, app, columns[2]);
}

fn render_namespace_list(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == PaneFocus::Namespaces;
    let border_style = if focused {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let items: Vec<ListItem> = app
        .namespaces
        .iter()
        .map(|ns| {
            let comp_count = ns.components.len();
            ListItem::new(format!("{} ({} components)", ns.name, comp_count))
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Namespaces ");

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");

    let mut state = ListState::default();
    if !app.namespaces.is_empty() {
        state.select(Some(app.ns_index));
    }
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_component_list(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == PaneFocus::Components;
    let border_style = if focused {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let components = app
        .selected_namespace()
        .map(|ns| &ns.components[..])
        .unwrap_or(&[]);

    let items: Vec<ListItem> = components
        .iter()
        .map(|comp| {
            let status_color = health_color(comp.status);
            let line = Line::from(vec![
                Span::styled(
                    format!("{} ", comp.status.symbol()),
                    Style::default().fg(status_color),
                ),
                Span::raw(&comp.name),
                Span::styled(
                    format!(" [{}/{}ep]", comp.instance_count, comp.endpoints.len()),
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            ListItem::new(line)
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Components ");

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");

    let mut state = ListState::default();
    if !components.is_empty() {
        state.select(Some(app.comp_index));
    }
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_endpoint_list(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == PaneFocus::Endpoints;
    let border_style = if focused {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let endpoints = app
        .selected_component()
        .map(|c| &c.endpoints[..])
        .unwrap_or(&[]);

    let items: Vec<ListItem> = endpoints
        .iter()
        .map(|ep| {
            let status_color = health_color(ep.status);
            let line = Line::from(vec![
                Span::styled(
                    format!("{} ", ep.status.symbol()),
                    Style::default().fg(status_color),
                ),
                Span::raw(&ep.name),
                Span::styled(
                    format!(" ({}x)", ep.instance_count),
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            ListItem::new(line)
        })
        .collect();

    // Show models in the title if available
    let title = if let Some(comp) = app.selected_component() {
        if !comp.models.is_empty() {
            format!(" Endpoints | Models: {} ", comp.models.join(", "))
        } else {
            " Endpoints ".to_string()
        }
    } else {
        " Endpoints ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");

    let mut state = ListState::default();
    if !endpoints.is_empty() {
        state.select(Some(app.ep_index));
    }
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_nats_bar(frame: &mut Frame, app: &App, area: Rect) {
    let stats = &app.nats_stats;
    let status = if stats.connected {
        Span::styled("Connected", Style::default().fg(Color::Green))
    } else {
        Span::styled("Disconnected", Style::default().fg(Color::Red))
    };

    let stream_info = if stats.streams.is_empty() {
        String::new()
    } else {
        let summaries: Vec<String> = stats
            .streams
            .iter()
            .map(|s| format!("{}({}c/{}m)", s.name, s.consumer_count, s.message_count))
            .collect();
        format!(" | Streams: {}", summaries.join(", "))
    };

    let line = Line::from(vec![
        Span::styled(" NATS: ", Style::default().add_modifier(Modifier::BOLD)),
        status,
        Span::raw(format!(
            " | Msgs {} / {}",
            format_count(stats.msgs_in),
            format_count(stats.msgs_out),
        )),
        Span::raw(format!(
            " | Bytes {} / {}",
            format_bytes(stats.bytes_in),
            format_bytes(stats.bytes_out),
        )),
        Span::raw(stream_info),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" NATS ");

    let para = Paragraph::new(line).block(block);
    frame.render_widget(para, area);
}

fn render_metrics_bar(frame: &mut Frame, app: &App, area: Rect) {
    let m = &app.metrics;
    let content = if !m.available {
        Line::from(vec![
            Span::styled(" Metrics: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled("Not configured", Style::default().fg(Color::DarkGray)),
        ])
    } else {
        let mut spans = vec![
            Span::styled(" Metrics: ", Style::default().add_modifier(Modifier::BOLD)),
        ];

        if let Some(ttft) = m.ttft_p50_ms {
            spans.push(Span::raw(format!("TTFT p50={:.0}ms", ttft)));
            if let Some(p99) = m.ttft_p99_ms {
                spans.push(Span::raw(format!(" p99={:.0}ms", p99)));
            }
            spans.push(Span::raw(" | "));
        }

        if let Some(tpot) = m.tpot_p50_ms {
            spans.push(Span::raw(format!("TPOT p50={:.0}ms", tpot)));
            if let Some(p99) = m.tpot_p99_ms {
                spans.push(Span::raw(format!(" p99={:.0}ms", p99)));
            }
            spans.push(Span::raw(" | "));
        }

        if let Some(inflight) = m.requests_inflight {
            spans.push(Span::raw(format!("Inflight: {}", inflight)));
            spans.push(Span::raw(" | "));
        }

        if let Some(queued) = m.requests_queued {
            spans.push(Span::raw(format!("Queued: {}", queued)));
            spans.push(Span::raw(" | "));
        }

        if let Some(tps) = m.tokens_per_sec {
            spans.push(Span::raw(format!("Tokens/s: {:.0}", tps)));
        }

        Line::from(spans)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Metrics ");

    let para = Paragraph::new(content).block(block).wrap(Wrap { trim: true });
    frame.render_widget(para, area);
}

fn render_help_bar(frame: &mut Frame, area: Rect) {
    let help = Line::from(vec![
        Span::styled(" q", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" Quit  "),
        Span::styled("hjkl/arrows", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" Navigate  "),
        Span::styled("Tab", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" Focus  "),
        Span::styled("r", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" Refresh"),
    ]);

    let para = Paragraph::new(help);
    frame.render_widget(para, area);
}

/// Map health status to a display color.
fn health_color(status: HealthStatus) -> Color {
    match status {
        HealthStatus::Ready => Color::Green,
        HealthStatus::Provisioning => Color::Yellow,
        HealthStatus::Offline => Color::Red,
    }
}

/// Format a byte count as human-readable (e.g., "12.3KB", "1.5MB").
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Format a message count (e.g., "1.2K", "3.5M").
fn format_count(count: u64) -> String {
    if count < 1000 {
        format!("{}", count)
    } else if count < 1_000_000 {
        format!("{:.1}K", count as f64 / 1000.0)
    } else {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(500), "500B");
        assert_eq!(format_bytes(1024), "1.0KB");
        assert_eq!(format_bytes(1536), "1.5KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0GB");
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1.0K");
        assert_eq!(format_count(12345), "12.3K");
        assert_eq!(format_count(1_234_567), "1.2M");
    }

    #[test]
    fn test_health_color() {
        assert_eq!(health_color(HealthStatus::Ready), Color::Green);
        assert_eq!(health_color(HealthStatus::Provisioning), Color::Yellow);
        assert_eq!(health_color(HealthStatus::Offline), Color::Red);
    }

    // Boundary value tests for format_bytes
    #[test]
    fn test_format_bytes_boundaries() {
        assert_eq!(format_bytes(1023), "1023B");
        assert_eq!(format_bytes(1024), "1.0KB");
        assert_eq!(format_bytes(1024 * 1024 - 1), "1024.0KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 - 1), "1024.0MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0GB");
    }

    #[test]
    fn test_format_bytes_large_values() {
        // 10 GB
        assert_eq!(format_bytes(10 * 1024 * 1024 * 1024), "10.0GB");
    }

    // Boundary value tests for format_count
    #[test]
    fn test_format_count_boundaries() {
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1.0K");
        assert_eq!(format_count(999_999), "1000.0K");
        assert_eq!(format_count(1_000_000), "1.0M");
    }

    #[test]
    fn test_format_count_large() {
        assert_eq!(format_count(999_999_999), "1000.0M");
    }

    // Render tests using ratatui's TestBackend
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use crate::app::App;
    use crate::model::*;
    use crate::sources::AppEvent;

    fn make_test_app() -> App {
        let mut app = App::new();
        let ns = vec![
            Namespace {
                name: "dynamo".into(),
                components: vec![
                    Component {
                        name: "backend".into(),
                        endpoints: vec![
                            Endpoint {
                                name: "generate".into(),
                                instance_count: 2,
                                status: HealthStatus::Ready,
                                last_seen: None,
                            },
                        ],
                        instance_count: 2,
                        status: HealthStatus::Ready,
                        models: vec!["llama-7b".into()],
                    },
                ],
            },
        ];
        app.handle_event(AppEvent::DiscoveryUpdate(ns));
        app
    }

    #[test]
    fn test_render_does_not_panic_loading() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new(); // loading = true
        terminal.draw(|frame| render(frame, &app)).unwrap();
    }

    #[test]
    fn test_render_does_not_panic_with_data() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = make_test_app();
        terminal.draw(|frame| render(frame, &app)).unwrap();
    }

    #[test]
    fn test_render_does_not_panic_empty_namespaces() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(vec![]));
        terminal.draw(|frame| render(frame, &app)).unwrap();
    }

    #[test]
    fn test_render_does_not_panic_small_terminal() {
        let backend = TestBackend::new(20, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = make_test_app();
        terminal.draw(|frame| render(frame, &app)).unwrap();
    }

    #[test]
    fn test_render_with_nats_stats() {
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = make_test_app();
        app.nats_stats = NatsStats {
            connected: true,
            server_id: "test".into(),
            msgs_in: 12345,
            msgs_out: 8901,
            bytes_in: 1024 * 1024 * 50,
            bytes_out: 1024 * 1024 * 12,
            streams: vec![StreamInfo {
                name: "audit".into(),
                consumer_count: 2,
                message_count: 500,
            }],
        };
        terminal.draw(|frame| render(frame, &app)).unwrap();
    }

    #[test]
    fn test_render_with_metrics() {
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = make_test_app();
        app.metrics = PrometheusMetrics {
            available: true,
            ttft_p50_ms: Some(42.0),
            ttft_p99_ms: Some(180.0),
            tpot_p50_ms: Some(8.0),
            tpot_p99_ms: Some(25.0),
            requests_inflight: Some(12),
            requests_queued: Some(3),
            tokens_per_sec: Some(450.5),
        };
        terminal.draw(|frame| render(frame, &app)).unwrap();
    }

    #[test]
    fn test_render_with_nats_disconnected() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = make_test_app();
        app.nats_stats = NatsStats::default(); // connected = false
        terminal.draw(|frame| render(frame, &app)).unwrap();
    }

    #[test]
    fn test_render_all_focus_states() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = make_test_app();

        for focus in [PaneFocus::Namespaces, PaneFocus::Components, PaneFocus::Endpoints] {
            app.focus = focus;
            terminal.draw(|frame| render(frame, &app)).unwrap();
        }
    }

    #[test]
    fn test_render_endpoint_pane_shows_models_in_title() {
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = make_test_app();
        // The app has models: ["llama-7b"] on the backend component
        // Render and check it doesn't panic
        terminal.draw(|frame| render(frame, &app)).unwrap();
        // We can verify the buffer contains the model name
        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer.content().iter().map(|c| c.symbol().to_string()).collect();
        assert!(content.contains("llama-7b"), "Buffer should contain model name");
    }
}
