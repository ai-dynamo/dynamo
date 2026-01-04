// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod app;
mod input;
mod metrics;
mod sources;
mod ui;

use std::sync::Arc;

use anyhow::Context;
use app::{App, AppEvent};
use clap::Parser;
use crossterm::{
    event::EnableMouseCapture,
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use dynamo_runtime::{DistributedRuntime, Runtime};
use input::InputEvent;
use ratatui::{Terminal, prelude::CrosstermBackend};
use sources::{spawn_discovery_pipeline, spawn_metrics_pipeline, spawn_nats_pipeline};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use ui::render;

/// Command line arguments for the Dynamo TUI.
#[derive(Debug, Parser)]
#[command(name = "dynamo-tui")]
#[command(about = "Terminal dashboard for monitoring Dynamo deployments")]
struct Args {
    /// Optional Prometheus /metrics endpoint to scrape for frontend metrics
    #[arg(long, env = "DYNAMO_TUI_METRICS_URL")]
    metrics_url: Option<String>,

    /// How often to refresh Prometheus metrics (e.g. "2s", "500ms", "1m")
    #[arg(long, default_value = "3s", value_parser = humantime::parse_duration)]
    metrics_interval: std::time::Duration,

    /// How often to poll NATS statistics (e.g. "1s")
    #[arg(long, default_value = "2s", value_parser = humantime::parse_duration)]
    nats_interval: std::time::Duration,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dynamo_runtime::logging::init();

    let args = Args::parse();

    let runtime = Runtime::from_settings().context("failed to create runtime from settings")?;
    let distributed_runtime = Arc::new(DistributedRuntime::from_settings(runtime.clone()).await?);

    let cancel_token = CancellationToken::new();

    let (app_event_tx, app_event_rx) = mpsc::channel::<AppEvent>(256);
    let (input_tx, input_rx) = mpsc::channel::<InputEvent>(32);

    let mut tasks = Vec::new();

    tasks.push(
        spawn_discovery_pipeline(
            distributed_runtime.clone(),
            app_event_tx.clone(),
            cancel_token.clone(),
        )
        .await?,
    );

    if let Some(nats_handle) = spawn_nats_pipeline(
        distributed_runtime.clone(),
        app_event_tx.clone(),
        cancel_token.clone(),
        args.nats_interval,
    )
    .await?
    {
        tasks.push(nats_handle);
    }

    if let Some(url) = args.metrics_url.clone() {
        if let Some(metrics_handle) = spawn_metrics_pipeline(
            url,
            args.metrics_interval,
            app_event_tx.clone(),
            cancel_token.clone(),
        )
        .await?
        {
            tasks.push(metrics_handle);
        }
    }

    tasks.push(input::spawn_input_listener(input_tx, cancel_token.clone()));

    let backend = CrosstermBackend::new(std::io::stdout());
    let terminal = Terminal::new(backend)?;

    execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    crossterm::terminal::enable_raw_mode()?;

    let res = run_app(terminal, app_event_rx, input_rx, cancel_token.clone()).await;

    cancel_token.cancel();

    // Ensure background tasks exit
    for handle in tasks {
        handle.abort();
    }

    restore_terminal()?;

    if let Err(err) = res {
        tracing::error!(error = %err, "Dynamo TUI exited with error");
        return Err(err);
    }

    // Shutdown the runtime gracefully
    distributed_runtime.shutdown();
    runtime.shutdown();

    Ok(())
}

/// Returns the terminal to its previous state.
fn restore_terminal() -> anyhow::Result<()> {
    crossterm::terminal::disable_raw_mode()?;
    execute!(std::io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}

async fn run_app(
    mut terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
    mut app_events: mpsc::Receiver<AppEvent>,
    mut input_events: mpsc::Receiver<InputEvent>,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    let mut app = App::default();
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(100));

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => break,
            _ = tick.tick() => {
                terminal.draw(|frame| render(frame, &app))?;
            }
            Some(event) = app_events.recv() => {
                app.apply_event(event);
            }
            Some(input) = input_events.recv() => {
                if app.handle_input(input) {
                    break;
                }
            }
            else => break,
        }
    }

    Ok(())
}
