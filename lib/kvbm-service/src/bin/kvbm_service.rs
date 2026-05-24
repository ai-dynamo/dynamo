// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbm_service` binary entry point.
//!
//! Layered configuration: clap CLI > `KVBM_SERVICE_*` env > optional TOML
//! (`--config <path>`) > defaults. The merged [`ServiceConfig`] is fed into
//! [`KvbmService::start`], which binds the HTTP sidecar and the gRPC UDS
//! listener.
//!
//! Shutdown: the first SIGINT or SIGTERM kicks off
//! [`KvbmService::shutdown_graceful`] with the configured grace period.
//! Subsequent signals are logged and **ignored** — operators are not allowed
//! to abbreviate the grace window because in-flight host→GPU writes can
//! corrupt client state. SIGKILL remains the escape hatch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;
use kvbm_service::config::ServiceFlags;
use kvbm_service::{KvbmService, NoopContainer, ServiceConfig};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "kvbm_service", about = "KVBM service shell")]
struct Cli {
    #[command(flatten)]
    flags: ServiceFlags,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let cfg = ServiceConfig::load(cli.flags.config_path(), cli.flags.to_cli())?;
    tracing::info!(?cfg, "loaded service config");

    let grace = cfg.shutdown_grace();
    // Always build the host-memory pool — the binary is the production
    // entry point and the `/v1/pool` snapshot is part of the contract.
    // Tests that don't want a pool call `KvbmService::start` directly.
    let service = KvbmService::start_with_pool(cfg, Arc::new(NoopContainer)).await?;
    tracing::info!(
        http_addr = %service.http_addr,
        uds_path = %service.uds_path.display(),
        grace = ?grace,
        pool_slabs = service.pool.as_ref().map(|p| p.slabs().len()).unwrap_or(0),
        "kvbm-service ready"
    );

    // Signal pump: forward the first signal through a oneshot. Later signals
    // are logged and dropped — the grace period must complete to avoid
    // data corruption.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let shutdown_started = Arc::new(AtomicBool::new(false));
    {
        let shutdown_started = shutdown_started.clone();
        tokio::spawn(async move {
            let mut shutdown_tx = Some(shutdown_tx);
            loop {
                wait_for_signal().await;
                if shutdown_started.swap(true, Ordering::SeqCst) {
                    tracing::warn!(
                        "shutdown signal received during graceful shutdown — ignoring \
                         (grace period must complete to avoid data corruption; \
                         use SIGKILL if you really need to abort)"
                    );
                    continue;
                }
                tracing::info!("shutdown signal received; beginning graceful shutdown");
                if let Some(tx) = shutdown_tx.take() {
                    let _ = tx.send(());
                }
            }
        });
    }

    // Wait for the first signal, then drive the graceful sequence.
    let _ = shutdown_rx.await;
    service.shutdown_graceful(grace).await;
    tracing::info!("kvbm-service shutdown complete");
    Ok(())
}

#[cfg(unix)]
async fn wait_for_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = sigint.recv() => {},
        _ = sigterm.recv() => {},
    }
}

#[cfg(not(unix))]
async fn wait_for_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
