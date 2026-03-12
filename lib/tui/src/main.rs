// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo TUI — Terminal UI for monitoring Dynamo deployments.
//!
//! A K9s-like interface that watches ETCD for service discovery,
//! monitors NATS message flows, and optionally scrapes Prometheus metrics.
//!
//! # Usage
//!
//! ```bash
//! # Basic usage (ETCD + NATS monitoring)
//! cargo run -p dynamo-tui
//!
//! # With metrics endpoint
//! cargo run -p dynamo-tui -- --metrics-url http://localhost:9100/metrics
//! ```

mod app;
mod input;
mod model;
mod sources;
mod ui;

use std::io;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use crossterm::event::{Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio_util::sync::CancellationToken;
use tracing as log;

use app::App;
use sources::etcd::{EtcdConfig, EtcdSource};
use sources::metrics::{MetricsConfig, MetricsSource};
use sources::nats::{NatsConfig, NatsSource};
use sources::{AppEvent, Source};

/// Dynamo TUI — Terminal UI for monitoring Dynamo deployments.
#[derive(Parser, Debug)]
#[command(
    name = "dynamo-tui",
    about = "Terminal UI for monitoring Dynamo deployments"
)]
struct Cli {
    /// ETCD endpoint(s), comma-separated. Overrides ETCD_ENDPOINTS env var.
    #[arg(long)]
    etcd_endpoints: Option<String>,

    /// NATS server URL. Overrides NATS_SERVER env var.
    #[arg(long)]
    nats_server: Option<String>,

    /// Prometheus metrics endpoint URL to scrape (optional).
    #[arg(long)]
    metrics_url: Option<String>,

    /// NATS polling interval.
    #[arg(long, default_value = "2s", value_parser = parse_duration)]
    nats_interval: Duration,

    /// Prometheus metrics scrape interval.
    #[arg(long, default_value = "3s", value_parser = parse_duration)]
    metrics_interval: Duration,
}

fn parse_duration(s: &str) -> Result<Duration, humantime::DurationError> {
    humantime::parse_duration(s)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing (to file, not terminal — we own the terminal)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "dynamo_tui=info,warn".into()),
        )
        .with_writer(io::stderr)
        .init();

    let cli = Cli::parse();
    let cancel = CancellationToken::new();

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Install panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    // Run the app
    let result = run_app(&mut terminal, cli, cancel.clone()).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cli: Cli,
    cancel: CancellationToken,
) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<AppEvent>(256);
    let mut app = App::new();

    // Spawn ETCD source
    let etcd_config = EtcdConfig::from_env(cli.etcd_endpoints.as_deref());
    let etcd_source = Box::new(EtcdSource::new(etcd_config));
    let etcd_tx = tx.clone();
    let etcd_cancel = cancel.clone();
    tokio::spawn(async move {
        etcd_source.run(etcd_tx, etcd_cancel).await;
    });

    // Spawn NATS source
    let nats_config = NatsConfig::from_env(cli.nats_server.as_deref(), Some(cli.nats_interval));
    let nats_source = Box::new(NatsSource::new(nats_config));
    let nats_tx = tx.clone();
    let nats_cancel = cancel.clone();
    tokio::spawn(async move {
        nats_source.run(nats_tx, nats_cancel).await;
    });

    // Spawn metrics source (if configured)
    if let Some(metrics_url) = cli.metrics_url {
        let metrics_config = MetricsConfig {
            url: metrics_url,
            scrape_interval: cli.metrics_interval,
        };
        let metrics_source = Box::new(MetricsSource::new(metrics_config));
        let metrics_tx = tx.clone();
        let metrics_cancel = cancel.clone();
        tokio::spawn(async move {
            metrics_source.run(metrics_tx, metrics_cancel).await;
        });
    }

    // Spawn terminal event reader
    let input_tx = tx.clone();
    let input_cancel = cancel.clone();
    tokio::spawn(async move {
        let mut reader = EventStream::new();
        loop {
            tokio::select! {
                _ = input_cancel.cancelled() => break,
                event = reader.next() => {
                    match event {
                        Some(Ok(Event::Key(key))) => {
                            let _ = input_tx.send(AppEvent::Input(key)).await;
                        }
                        Some(Err(e)) => {
                            log::error!("Terminal event error: {}", e);
                            break;
                        }
                        None => break,
                        _ => {} // Ignore mouse/resize for now
                    }
                }
            }
        }
    });

    // Spawn tick timer
    let tick_tx = tx;
    let tick_cancel = cancel.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(250));
        loop {
            tokio::select! {
                _ = tick_cancel.cancelled() => break,
                _ = interval.tick() => {
                    let _ = tick_tx.send(AppEvent::Tick).await;
                }
            }
        }
    });

    // Main event loop
    loop {
        // Render
        terminal.draw(|frame| ui::render(frame, &app))?;

        // Wait for next event
        if let Some(event) = rx.recv().await {
            app.handle_event(event);
            if app.should_quit {
                cancel.cancel();
                break;
            }
        } else {
            break;
        }
    }

    Ok(())
}
