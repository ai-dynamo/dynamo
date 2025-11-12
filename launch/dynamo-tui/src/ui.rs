// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{borrow::Cow, time::Instant};

use humantime::format_duration;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
};

use crate::app::{App, FocusColumn, InstanceState};

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.size();
    if area.height < 6 || area.width < 20 {
        frame.render_widget(
            Paragraph::new("Terminal too small for Dynamo TUI").block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Dynamo TUI")
                    .title_alignment(ratatui::layout::Alignment::Center),
            ),
            area,
        );
        return;
    }

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),
            Constraint::Length(7),
            Constraint::Length(1),
        ])
        .split(area);

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(24),
            Constraint::Percentage(32),
            Constraint::Percentage(44),
        ])
        .split(vertical[0]);

    render_namespaces(frame, columns[0], app);
    render_components(frame, columns[1], app);
    render_endpoints(frame, columns[2], app);

    render_stats(frame, vertical[1], app);
    render_status_bar(frame, vertical[2], app);
}

fn render_namespaces(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .namespaces_iter()
        .enumerate()
        .map(|(idx, (name, namespace))| {
            let summary = format!("{} components", namespace.components.len());
            let selected = idx == app.selection().namespace_index;
            let focus = matches!(app.selection().focus, FocusColumn::Namespaces);
            ListItem::new(Line::from(vec![
                Span::styled(
                    name.clone(),
                    styles_for_selection(selected, focus, Style::default().fg(Color::Cyan)),
                ),
                Span::raw(" "),
                Span::styled(
                    summary,
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                ),
            ]))
        })
        .collect();

    let list = List::new(items).block(Block::default().title("Namespaces").borders(Borders::ALL));

    frame.render_widget(list, area);
}

fn render_components(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = (0..app.component_count())
        .filter_map(|idx| {
            let (_, namespace) = app.namespace_at(app.selection().namespace_index)?;
            let (name, component) = namespace.components.iter().nth(idx)?;
            let summary = app.component_health_summary(component);
            Some((idx, name.clone(), summary))
        })
        .map(|(idx, name, summary)| {
            let selected = idx == app.selection().component_index;
            let focus = matches!(app.selection().focus, FocusColumn::Components);
            let status_label = format!(
                "{} · {}/{} endpoints · {} instances",
                summary.status.display_name(),
                summary.active_endpoints,
                summary.total_endpoints,
                summary.instances
            );

            ListItem::new(Line::from(vec![
                Span::styled(
                    name,
                    styles_for_selection(
                        selected,
                        focus,
                        Style::default()
                            .fg(Color::LightMagenta)
                            .add_modifier(Modifier::BOLD),
                    ),
                ),
                Span::raw(" "),
                Span::styled(
                    status_label,
                    Style::default()
                        .fg(summary.status.as_color())
                        .add_modifier(Modifier::ITALIC),
                ),
            ]))
        })
        .collect();

    let list = List::new(items).block(Block::default().title("Components").borders(Borders::ALL));

    frame.render_widget(list, area);
}

fn render_endpoints(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let endpoint_items: Vec<ListItem> = (0..app.endpoint_count())
        .filter_map(|idx| {
            let (name, endpoint) = app.endpoint_at(
                app.selection().namespace_index,
                app.selection().component_index,
                idx,
            )?;
            let summary = app.endpoint_health_summary(endpoint);
            Some((idx, name.clone(), summary))
        })
        .map(|(idx, name, summary)| {
            let selected = idx == app.selection().endpoint_index;
            let focus = matches!(app.selection().focus, FocusColumn::Endpoints);
            let status_label = format!(
                "{} · {} instance{}",
                summary.status.display_name(),
                summary.instances,
                if summary.instances == 1 { "" } else { "s" }
            );

            ListItem::new(Line::from(vec![
                Span::styled(
                    name,
                    styles_for_selection(
                        selected,
                        focus,
                        Style::default()
                            .fg(Color::LightYellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ),
                Span::raw(" "),
                Span::styled(
                    status_label,
                    Style::default()
                        .fg(summary.status.as_color())
                        .add_modifier(Modifier::ITALIC),
                ),
            ]))
        })
        .collect();

    let list =
        List::new(endpoint_items).block(Block::default().title("Endpoints").borders(Borders::ALL));

    frame.render_widget(list, chunks[0]);

    render_endpoint_detail(frame, chunks[1], app);
}

fn render_endpoint_detail(frame: &mut Frame, area: Rect, app: &App) {
    let Some((_, endpoint_state)) = app.endpoint_at(
        app.selection().namespace_index,
        app.selection().component_index,
        app.selection().endpoint_index,
    ) else {
        let block = Block::default()
            .title("Endpoint Details")
            .borders(Borders::ALL);
        frame.render_widget(block, area);
        return;
    };

    let summary = app.endpoint_health_summary(endpoint_state);
    let mut lines = Vec::new();

    lines.push(Line::from(vec![
        Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            summary.status.display_name(),
            Style::default().fg(summary.status.as_color()),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Instances: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(summary.instances.to_string()),
    ]));

    if let Some(last_seen) = summary.last_seen {
        if let Some(elapsed) = Instant::now().checked_duration_since(last_seen) {
            lines.push(Line::from(vec![
                Span::styled(
                    "Last activity: ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format_duration(elapsed).to_string()),
            ]));
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::styled(
        "Instances",
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Cyan),
    ));

    if endpoint_state.instances.is_empty() {
        lines.push(Line::styled(
            "No active instances",
            Style::default().fg(Color::DarkGray),
        ));
    } else {
        for instance in endpoint_state.instances.values() {
            lines.extend(instance_lines(instance));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .title("Endpoint Details")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

fn instance_lines(instance: &InstanceState) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(Line::from(vec![Span::styled(
        format!("• Instance {}", instance.instance.instance_id),
        Style::default().fg(Color::LightMagenta),
    )]));

    let transport = match &instance.instance.transport {
        dynamo_runtime::component::TransportType::NatsTcp(subject) => {
            Cow::Owned(format!("NATS subject: {subject}"))
        }
    };

    lines.push(Line::from(vec![
        Span::raw("   "),
        Span::styled(transport, Style::default().fg(Color::Gray)),
    ]));

    let elapsed = instance.last_seen.elapsed();
    lines.push(Line::from(vec![
        Span::raw("   "),
        Span::styled(
            format!("Last seen {}", format_duration(elapsed)),
            Style::default().fg(Color::DarkGray),
        ),
    ]));

    lines
}

fn render_stats(frame: &mut Frame, area: Rect, app: &App) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    render_nats_stats(frame, columns[0], app);
    render_metrics(frame, columns[1], app);
}

fn render_nats_stats(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default().title("NATS").borders(Borders::ALL);
    if let Some(stats) = app.nats_snapshot() {
        let text = vec![
            Line::from(format!(
                "State: {}",
                match stats.connection_state {
                    crate::app::NatsConnectionState::Connected => "Connected",
                    crate::app::NatsConnectionState::Disconnected => "Disconnected",
                }
            )),
            Line::from(format!(
                "In: {} bytes / {} messages",
                human_bytes(stats.in_bytes),
                stats.in_messages
            )),
            Line::from(format!(
                "Out: {} bytes / {} messages",
                human_bytes(stats.out_bytes),
                stats.out_messages
            )),
            Line::from(format!("Connections: {}", stats.connects)),
            Line::from(format!(
                "Updated: {} ago",
                format_duration(stats.last_updated.elapsed())
            )),
        ];
        frame.render_widget(
            Paragraph::new(text).block(block).wrap(Wrap { trim: true }),
            area,
        );
    } else {
        frame.render_widget(
            Paragraph::new("NATS statistics unavailable").block(block),
            area,
        );
    }
}

fn render_metrics(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default().title("Metrics").borders(Borders::ALL);

    if let Some(metrics) = app.metrics_snapshot() {
        let text = vec![
            format_metric_line("TTFT", metrics.ttft_ms, "ms"),
            format_metric_line("TPOT", metrics.tpot_ms, "ms"),
            format_metric_line("Requests/s", metrics.request_rate, "req/s"),
            format_metric_line("Output tokens/s", metrics.tokens_per_sec, "tok/s"),
            format_metric_line("Inflight", metrics.inflight, ""),
            format_metric_line("Queued", metrics.queued, ""),
            format!(
                "Totals: {:.0} requests · {:.0} tokens",
                metrics.total_requests, metrics.total_output_tokens
            ),
            format!(
                "Collected: {} ago",
                format_duration(metrics.collected_at.elapsed())
            )
            .to_string(),
        ];
        let paragraph = Paragraph::new(text.join("\n"))
            .block(block)
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, area);
    } else {
        frame.render_widget(Paragraph::new("Metrics not available").block(block), area);
    }
}

fn render_status_bar(frame: &mut Frame, area: Rect, app: &App) {
    let (mut message, style) = if let Some(err) = app.last_error() {
        (
            format!("Error: {err}"),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    } else if let Some(info) = app.last_info() {
        (info.clone(), Style::default().fg(Color::Cyan))
    } else {
        ("Ready".to_string(), Style::default().fg(Color::Gray))
    };

    if let Some(updated) = app.last_update() {
        let elapsed = format_duration(updated.elapsed()).to_string();
        message.push_str(&format!(" · last update {}", elapsed));
    }

    let paragraph =
        Paragraph::new(Span::styled(message, style)).block(Block::default().borders(Borders::ALL));

    frame.render_widget(paragraph, area);
}

fn styles_for_selection(selected: bool, focused: bool, base: Style) -> Style {
    if selected {
        if focused {
            base.add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                .fg(Color::Yellow)
        } else {
            base.add_modifier(Modifier::BOLD)
        }
    } else {
        base
    }
}

fn format_metric_line(label: &str, value: Option<f64>, unit: &str) -> String {
    match value {
        Some(v) => {
            if unit.is_empty() {
                format!("{label}: {:.2}", v)
            } else {
                format!("{label}: {:.2} {unit}", v)
            }
        }
        None => format!("{label}: —"),
    }
}

fn human_bytes(value: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = value as f64;
    let mut idx = 0;
    while size >= 1024.0 && idx < UNITS.len() - 1 {
        size /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{value} {}", UNITS[idx])
    } else {
        format!("{size:.1} {}", UNITS[idx])
    }
}
