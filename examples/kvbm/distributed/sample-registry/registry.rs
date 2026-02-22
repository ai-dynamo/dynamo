// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry Hub Binary
//!
//! Runs the distributed object registry hub for KV cache block coordination.
//!
//! # Usage
//!
//! ```bash
//! # Build and run from this directory
//! cd examples/kvbm/distributed/sample-registry
//! cargo run --release
//!
//! # Or build from workspace root
//! cargo build --manifest-path examples/kvbm/distributed/sample-registry/Cargo.toml --release
//!
//! # Run with custom settings via environment variables
//! DYN_REGISTRY_HUB_CAPACITY=10000000 \
//! DYN_REGISTRY_HUB_QUERY_ADDR=tcp://*:6000 \
//! DYN_REGISTRY_HUB_REGISTER_ADDR=tcp://*:6001 \
//! cargo run --manifest-path examples/kvbm/distributed/sample-registry/Cargo.toml --release
//! ```
//!
//! # Environment Variables
//!
//! - `DYN_REGISTRY_HUB_CAPACITY`: Registry capacity (default: 1000000)
//! - `DYN_REGISTRY_HUB_QUERY_ADDR`: Query address (default: tcp://*:5555)
//! - `DYN_REGISTRY_HUB_REGISTER_ADDR`: Register address (default: tcp://*:5556)
//! - `DYN_REGISTRY_HUB_METRICS_ADDR`: Metrics HTTP address (default: 0.0.0.0:9108)

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::EnvFilter;

use dynamo_llm::block_manager::block::transfer::remote::RemoteKey;
use dynamo_llm::block_manager::distributed::registry::{
    BinaryCodec, NoMetadata, PositionalEviction, PositionalKey, RegistryHubConfig, ZmqHub,
    ZmqHubServerConfig,
};

/// Type alias for the concrete hub type we use.
///
/// Uses `PositionalKey` for position-aware storage and `PositionalEviction`
/// which evicts from highest positions first (tail-first), optimizing for
/// prefix reuse in KV cache scenarios.
type G4RegistryHub = ZmqHub<
    PositionalKey,
    RemoteKey,
    NoMetadata,
    PositionalEviction<PositionalKey, RemoteKey>,
    BinaryCodec<PositionalKey, RemoteKey, NoMetadata>,
>;

struct MetricsState {
    started_at: Instant,
    capacity: u64,
}

async fn run_metrics_server(
    addr: String,
    state: Arc<MetricsState>,
    cancel: CancellationToken,
) -> Result<()> {
    let listener = TcpListener::bind(&addr).await?;
    info!("Registry metrics endpoint listening on http://{addr}/metrics");

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                return Ok(());
            }
            accepted = listener.accept() => {
                let (mut socket, _) = match accepted {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, "registry metrics accept failed");
                        continue;
                    }
                };

                let state = state.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = match socket.read(&mut buf).await {
                        Ok(n) => n,
                        Err(e) => {
                            tracing::warn!(error = %e, "registry metrics read failed");
                            return;
                        }
                    };
                    if n == 0 {
                        return;
                    }

                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first_line = req.lines().next().unwrap_or_default();

                    let (status, body, content_type) = if first_line.starts_with("GET /metrics") {
                        let uptime = state.started_at.elapsed().as_secs_f64();
                        let body = format!(
                            "# HELP registry_up Registry process liveness.\n\
                             # TYPE registry_up gauge\n\
                             registry_up 1\n\
                             # HELP registry_capacity_entries Configured registry capacity.\n\
                             # TYPE registry_capacity_entries gauge\n\
                             registry_capacity_entries {}\n\
                             # HELP registry_uptime_seconds Registry uptime in seconds.\n\
                             # TYPE registry_uptime_seconds gauge\n\
                             registry_uptime_seconds {:.3}\n",
                            state.capacity, uptime
                        );
                        ("200 OK", body, "text/plain; version=0.0.4")
                    } else if first_line.starts_with("GET /healthz") {
                        ("200 OK", "ok\n".to_string(), "text/plain")
                    } else {
                        ("404 Not Found", "not found\n".to_string(), "text/plain")
                    };

                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    if let Err(e) = socket.write_all(resp.as_bytes()).await {
                        tracing::warn!(error = %e, "registry metrics write failed");
                    }
                });
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    // Load config from environment
    let config = RegistryHubConfig::from_env();

    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║           Distributed Object Registry                        ║");
    info!("╠══════════════════════════════════════════════════════════════╣");
    info!(
        "║  Capacity:        {:<43}║",
        format!("{} entries", config.capacity)
    );
    info!("║  Query Addr:      {:<43}║", config.query_addr);
    info!("║  Register Addr:   {:<43}║", config.register_addr);
    info!(
        "║  Lease Timeout:   {:<43}║",
        format!("{} secs", config.lease_timeout.as_secs())
    );
    let metrics_addr = std::env::var("DYN_REGISTRY_HUB_METRICS_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9108".to_string());
    info!("║  Metrics Addr:    {:<43}║", metrics_addr);
    info!("╚══════════════════════════════════════════════════════════════╝");

    // Convert to ZmqHubServerConfig
    let zmq_config = ZmqHubServerConfig {
        query_addr: config.query_addr,
        pull_addr: config.register_addr,
        capacity: config.capacity,
    };

    // Create hub with positional eviction storage and codec
    // PositionalEviction evicts from highest positions first (tail-first),
    // which is optimal for KV cache prefix reuse.
    let storage = PositionalEviction::with_capacity(config.capacity as usize);
    let codec = BinaryCodec::new();
    let hub: G4RegistryHub = ZmqHub::new(zmq_config, storage, codec);

    // Setup cancellation
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let metrics_state = Arc::new(MetricsState {
        started_at: Instant::now(),
        capacity: config.capacity,
    });

    let metrics_cancel = cancel.clone();
    tokio::spawn(async move {
        if let Err(e) = run_metrics_server(metrics_addr, metrics_state, metrics_cancel).await {
            tracing::error!(error = %e, "registry metrics server failed");
        }
    });

    // Handle Ctrl+C
    tokio::spawn(async move {
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received Ctrl+C, initiating shutdown...");
                cancel_clone.cancel();
            }
            Err(e) => {
                tracing::error!("Failed to listen for Ctrl+C: {}", e);
            }
        }
    });

    // Run hub
    info!("Starting registry hub... Press Ctrl+C to stop.");
    hub.serve(cancel).await?;

    info!("Registry hub stopped.");
    Ok(())
}
