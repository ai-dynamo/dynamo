// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dynamo-ft-controller`: one decider per vLLM fault-tolerance world.
//!
//! Logs go to stderr. Stdout carries one line per outcome:
//! `FT_CONTROLLER_RECOVERED ...` after each recovery,
//! `FT_CONTROLLER_ESCALATED ...` before exiting after an escalation, and, in a
//! dry run, `FT_CONTROLLER_WOULD_ESCALATE ...` where it would have escalated.
//!
//! Exit codes: 0 when stopped by SIGINT or SIGTERM, 1 when the controller
//! itself failed, 2 for bad arguments or a world that contradicts them, 3
//! after an escalation. A dry run never exits with code 3.

use std::collections::BTreeMap;
use std::future::Future;
use std::io::Write as _;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use dynamo_ft_controller::cli::{Cli, LogFormat, Member};
use dynamo_ft_controller::controller::{
    ConfigError, Controller, ESCALATED_EXIT_CODE, Event, Outcome, Rank, escalation_line,
    would_escalate_line,
};
use dynamo_ft_controller::engine::{HttpEngine, http_client};
use dynamo_ft_controller::escalate::CommandEscalator;
use dynamo_ft_controller::round::{FileRoundStore, MemoryRoundStore, RoundStore};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

/// Exit code for arguments that are wrong, as clap uses for its own checks.
const BAD_ARGUMENTS_EXIT_CODE: u8 = 2;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    // Checks clap cannot make alone (duplicate ranks, loopback store hosts, ...).
    let members = match cli.members() {
        Ok(members) => members,
        Err(error) => {
            eprintln!("error: {error:#}");
            return ExitCode::from(BAD_ARGUMENTS_EXIT_CODE);
        }
    };
    if let Err(error) = init_tracing(cli.log_format) {
        eprintln!("error: {error:#}");
        return ExitCode::from(BAD_ARGUMENTS_EXIT_CODE);
    }
    let world = cli.world.clone();
    match run(cli, members).await {
        Ok(Outcome::Stopped) => ExitCode::SUCCESS,
        Ok(Outcome::Escalated(reason)) => {
            print_line(&escalation_line(&world, &reason));
            ExitCode::from(ESCALATED_EXIT_CODE)
        }
        Err(error) if error.is::<ConfigError>() => {
            tracing::error!(error = %format!("{error:#}"), "configuration error; fix the arguments");
            ExitCode::from(BAD_ARGUMENTS_EXIT_CODE)
        }
        Err(error) => {
            tracing::error!(error = %format!("{error:#}"), "controller failed");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli, members: BTreeMap<u32, Member>) -> Result<Outcome> {
    let client = http_client().context("cannot build the HTTP client")?;
    let mut ranks = BTreeMap::new();
    for (rank, member) in members {
        let engine = HttpEngine::new(
            client.clone(),
            &member.url,
            cli.call_timeout,
            cli.wake_timeout,
        )?;
        ranks.insert(
            rank,
            Rank {
                engine: Arc::new(engine),
                store_host: member.store_host,
            },
        );
    }
    // Without --state-dir (allowed with --dry-run only) nothing is persisted.
    let rounds: Arc<dyn RoundStore> = match &cli.state_dir {
        Some(dir) => Arc::new(FileRoundStore::new(dir, &cli.world)),
        None => Arc::new(MemoryRoundStore::default()),
    };
    let escalator = Arc::new(CommandEscalator::new(
        cli.escalate_cmd.clone(),
        cli.escalate_timeout,
    ));
    let controller = Controller::new(cli.settings(), ranks, rounds, escalator)?;
    let shutdown = shutdown_signal()?;

    let (events, mut received) = mpsc::unbounded_channel();
    let world = cli.world;
    let printer = tokio::spawn(async move {
        while let Some(event) = received.recv().await {
            match event {
                Event::Recovered(recovery) => print_line(&recovery.to_string()),
                Event::WouldEscalate(reason) => print_line(&would_escalate_line(&world, &reason)),
                _ => {}
            }
        }
    });
    let outcome = controller.with_events(events).run(shutdown).await;
    // The controller is gone, so the printer drains and ends.
    printer.await.context("the summary printer failed")?;
    outcome
}

/// One summary line on stdout.
fn print_line(line: &str) {
    let mut stdout = std::io::stdout().lock();
    if let Err(error) = writeln!(stdout, "{line}").and_then(|()| stdout.flush()) {
        tracing::error!(%error, %line, "cannot write the summary line to stdout");
    }
}

fn init_tracing(format: LogFormat) -> Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env()
        .context("invalid RUST_LOG")?;
    let logs = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr);
    match format {
        LogFormat::Text => logs.try_init(),
        LogFormat::Json => logs.json().try_init(),
    }
    .map_err(|error| anyhow!("cannot set up logging: {error}"))
}

#[cfg(unix)]
fn shutdown_signal() -> Result<impl Future<Output = ()>> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut interrupt = signal(SignalKind::interrupt()).context("cannot listen for SIGINT")?;
    let mut terminate = signal(SignalKind::terminate()).context("cannot listen for SIGTERM")?;
    Ok(async move {
        tokio::select! {
            _ = interrupt.recv() => {}
            _ = terminate.recv() => {}
        }
    })
}

#[cfg(not(unix))]
fn shutdown_signal() -> Result<impl Future<Output = ()>> {
    Ok(async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "cannot listen for Ctrl-C; stop the controller by killing it");
            std::future::pending::<()>().await;
        }
    })
}
