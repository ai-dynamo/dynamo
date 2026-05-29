// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone KV-cache indexer.
//!
//! Two transport backends are available, each behind its own feature gate:
//!
//! - **`indexer-runtime`** (HTTP + ZMQ): hosts an Axum HTTP server with
//!   `/register`, `/unregister`, `/query`, `/query_by_hash`, and peer-discovery
//!   routes.  Each registered worker spawns a ZMQ listener that ingests its KV
//!   events.  Entry point: [`run_server`].
//!
//! - **`velo-runtime`**: sets up the Velo process shell (TCP + optional UDS
//!   transports, filesystem peer discovery).  Query and ingest handlers are
//!   placeholders that land in follow-up MRs.  Entry point: [`run_server_velo`].
//!
//! ## Response shapes
//!
//! **`indexer-runtime`** (HTTP) returns a `ScoreResponse` with token counts
//! scaled by `block_size`, aligned with the Mooncake KV-Store Indexer API
//! Standardization RFC (<https://github.com/kvcache-ai/Mooncake/issues/1403>).
//! Tier counts are cumulative — see [`server`] for the exact semantics.

pub mod indexer;
pub mod listener;
pub mod metrics;
pub mod recovery;
pub mod registry;
pub mod server;
mod zmq;

#[cfg(feature = "velo-runtime")]
pub mod runtime;

use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::config::min_initial_workers_from_env;
use registry::WorkerRegistry;
use server::{AppState, create_router};

pub struct IndexerConfig {
    pub block_size: Option<u32>,
    pub port: u16,
    pub threads: usize,
    pub workers: Option<String>,
    pub model_name: String,
    pub tenant_id: String,
    pub peers: Option<String>,
}

pub(super) fn validate_zmq_endpoint(endpoint: &str) -> anyhow::Result<()> {
    let (scheme, address) = endpoint
        .split_once("://")
        .ok_or_else(|| anyhow::anyhow!("invalid ZMQ endpoint `{endpoint}`: missing scheme"))?;

    if address.is_empty() {
        anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing address");
    }

    match scheme {
        "tcp" => {
            let (host, port) = address.rsplit_once(':').ok_or_else(|| {
                anyhow::anyhow!("invalid ZMQ endpoint `{endpoint}`: missing TCP port")
            })?;
            if host.is_empty() {
                anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing TCP host");
            }
            if host.starts_with('[') {
                if !host.ends_with(']') {
                    anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing closing `]`");
                }
            } else if host.contains(':') {
                anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing TCP port");
            }
            port.parse::<u16>().map_err(|error| {
                anyhow::anyhow!("invalid ZMQ endpoint `{endpoint}`: invalid TCP port: {error}")
            })?;
            Ok(())
        }
        "ipc" | "inproc" => Ok(()),
        other => Err(anyhow::anyhow!(
            "invalid ZMQ endpoint `{endpoint}`: unsupported scheme `{other}`"
        )),
    }
}

pub(super) fn validate_listener_endpoints(
    endpoint: &str,
    replay_endpoint: Option<&str>,
) -> anyhow::Result<()> {
    validate_zmq_endpoint(endpoint)?;
    if let Some(replay_endpoint) = replay_endpoint {
        validate_zmq_endpoint(replay_endpoint).map_err(|error| {
            anyhow::anyhow!("invalid replay endpoint `{replay_endpoint}`: {error}")
        })?;
    }
    Ok(())
}

pub fn parse_workers(s: &str) -> anyhow::Result<Vec<(u64, u32, String)>> {
    let mut workers = Vec::new();

    for entry in s.split(',').filter(|entry| !entry.trim().is_empty()) {
        let (id_part, addr) = entry.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("invalid worker entry `{entry}`; expected worker_id[:dp_rank]=endpoint")
        })?;
        let id_part = id_part.trim();
        let (instance_id, dp_rank) = if let Some((id_str, rank_str)) = id_part.split_once(':') {
            (
                id_str
                    .parse::<u64>()
                    .map_err(|error| anyhow::anyhow!("invalid worker id in `{entry}`: {error}"))?,
                rank_str
                    .parse::<u32>()
                    .map_err(|error| anyhow::anyhow!("invalid dp_rank in `{entry}`: {error}"))?,
            )
        } else {
            (
                id_part
                    .parse::<u64>()
                    .map_err(|error| anyhow::anyhow!("invalid worker id in `{entry}`: {error}"))?,
                0,
            )
        };

        let endpoint = addr.trim().to_string();
        validate_zmq_endpoint(&endpoint)?;
        workers.push((instance_id, dp_rank, endpoint));
    }

    Ok(workers)
}

pub async fn run_server(config: IndexerConfig) -> anyhow::Result<()> {
    let cancel_token = CancellationToken::new();
    let shutdown_token = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Received shutdown signal");
        shutdown_token.cancel();
    });

    let peers: Vec<String> = config
        .peers
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.trim().to_string())
                .collect()
        })
        .unwrap_or_default();

    tracing::info!(
        block_size = ?config.block_size,
        port = config.port,
        threads = config.threads,
        model_name = %config.model_name,
        tenant_id = %config.tenant_id,
        num_peers = peers.len(),
        "Starting standalone KV cache indexer (HTTP-only mode)"
    );

    let registry = Arc::new(WorkerRegistry::new(config.threads));
    run_common(&config, &registry, cancel_token).await
}

async fn wait_for_min_initial_workers(
    registry: &WorkerRegistry,
    cancel_token: &CancellationToken,
) -> anyhow::Result<()> {
    let min_initial_workers = min_initial_workers_from_env()?;
    if min_initial_workers == 0 {
        return Ok(());
    }

    loop {
        let registered_workers = registry.list().len();
        if registered_workers >= min_initial_workers {
            return Ok(());
        }

        tokio::select! {
            _ = cancel_token.cancelled() => {
                anyhow::bail!(
                    "shutdown triggered before {} indexer workers appeared",
                    min_initial_workers
                );
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
}

async fn run_common(
    config: &IndexerConfig,
    registry: &Arc<WorkerRegistry>,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    if let Some(ref workers_str) = config.workers {
        let block_size = config.block_size.ok_or_else(|| {
            anyhow::anyhow!("--block-size is required when --workers is specified")
        })?;
        for (instance_id, dp_rank, endpoint) in parse_workers(workers_str)? {
            tracing::info!(instance_id, dp_rank, endpoint, "Registering initial worker");
            registry
                .register(
                    instance_id,
                    endpoint,
                    dp_rank,
                    config.model_name.clone(),
                    config.tenant_id.clone(),
                    block_size,
                    None,
                )
                .await?;
        }
    }

    let peers: Vec<String> = config
        .peers
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.trim().to_string())
                .collect()
        })
        .unwrap_or_default();

    if !peers.is_empty() {
        match recovery::recover_from_peers(&peers, registry).await {
            Ok(true) => tracing::info!("P2P recovery completed"),
            Ok(false) => tracing::warn!("no reachable peers, starting with empty state"),
            Err(e) => tracing::warn!(error = %e, "P2P recovery failed, starting with empty state"),
        }
        for peer in &peers {
            registry.register_peer(peer.clone());
        }
    }

    wait_for_min_initial_workers(registry, &cancel_token).await?;
    registry.signal_ready();

    #[cfg(feature = "metrics")]
    let prom_registry = {
        let r = prometheus::Registry::new();
        metrics::register(&r).expect("failed to register indexer metrics");
        r
    };

    let state = Arc::new(AppState {
        registry: registry.clone(),
        #[cfg(feature = "metrics")]
        prom_registry,
    });

    let app = create_router(state);
    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    tracing::info!("HTTP server listening on 0.0.0.0:{}", config.port);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel_token.cancelled().await;
            tracing::info!("Received shutdown signal, stopping HTTP server");
        })
        .await?;

    Ok(())
}

// ── Velo-runtime entry point ──────────────────────────────────────────────

/// Configuration for the velo-based standalone indexer.
#[cfg(feature = "velo-runtime")]
pub struct VeloIndexerConfig {
    /// TCP port to bind.  0 = OS-assigned.
    pub tcp_port: u16,
    /// If set, publish a [`runtime::discovery::IndexerPeerSnapshot`] JSON file
    /// (`kv-indexer.json`) to this directory.  Workers call
    /// `discovery::connect_to_indexer(&messenger, dir)` to read the file and
    /// register the indexer peer.  This is **not** velo's built-in
    /// `FilesystemPeerDiscovery` — the on-disk format differs.
    pub discovery_dir: Option<std::path::PathBuf>,
    /// If set, also bind a Unix domain socket at this path (Linux only).
    /// Velo prefers UDS over TCP for same-host connections automatically,
    /// which is useful for benchmarking co-located workers.
    #[cfg(unix)]
    pub uds_path: Option<std::path::PathBuf>,
}

/// Run the standalone KV cache indexer using velo as the transport layer.
///
/// Binds TCP (and optionally UDS) transports, registers no-op placeholder
/// handlers for event ingest and prefix-match queries, then publishes the
/// indexer's peer info for workers and routers to discover via
/// [`runtime::discovery::connect_to_indexer`].
///
/// Full query and ingest handlers land in follow-up MRs.
/// Gracefully shuts down on `Ctrl-C`.
#[cfg(feature = "velo-runtime")]
pub async fn run_server_velo(config: VeloIndexerConfig) -> anyhow::Result<()> {
    use std::net::TcpListener as StdTcpListener;
    use std::sync::Arc as StdArc;

    use velo::Messenger;
    use velo::backend::tcp::TcpTransportBuilder;

    use runtime::{discovery::IndexerDiscovery, query_engine, subscriber};

    // ── Build TCP transport ──────────────────────────────────────────────
    let listener = StdTcpListener::bind(("0.0.0.0", config.tcp_port)).map_err(|e| {
        anyhow::anyhow!(
            "failed to bind TCP listener on port {}: {e}",
            config.tcp_port
        )
    })?;
    let actual_addr = listener.local_addr()?;

    let tcp_transport = TcpTransportBuilder::new()
        .from_listener(listener)?
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build TCP transport: {e}"))?;

    tracing::info!(addr = %actual_addr, "Velo TCP transport bound");

    // ── Optionally add UDS transport (Linux only) ────────────────────────
    // UDS is registered BEFORE TCP so that velo's insertion-order priority
    // selects it automatically for same-host peers, giving the documented
    // low-latency fast path without any explicit peer-side configuration.
    let mut builder = Messenger::builder();

    #[cfg(unix)]
    if let Some(ref uds_path) = config.uds_path {
        use velo::backend::uds::UdsTransportBuilder;

        let uds_transport = UdsTransportBuilder::new()
            .socket_path(uds_path)
            .build()
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to build UDS transport at {}: {e}",
                    uds_path.display()
                )
            })?;

        builder = builder.add_transport(StdArc::new(uds_transport));
        tracing::info!(path = %uds_path.display(), "Velo UDS transport bound");
    }

    let builder = builder.add_transport(StdArc::new(tcp_transport));

    let messenger: StdArc<Messenger> = builder
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("failed to build velo Messenger: {e}"))?;

    // ── Register velo handlers first ─────────────────────────────────────
    // Handlers must be registered before the peer info is published.  A worker
    // that discovers the indexer immediately can send an AM event while Velo
    // still has no registered handler for that endpoint, causing a silent drop.
    // These are no-op stubs; full implementations land in follow-up MRs.
    subscriber::register(&messenger)?;
    query_engine::register(&messenger)?;

    // ── Publish peer info for workers/routers to discover ────────────────
    let _discovery_handle = if let Some(ref discovery_dir) = config.discovery_dir {
        Some(
            IndexerDiscovery::publish(messenger.clone(), discovery_dir)
                .map_err(|e| anyhow::anyhow!("failed to publish peer info: {e}"))?,
        )
    } else {
        None
    };

    tracing::info!(
        instance_id = ?messenger.instance_id(),
        "Velo-based KV cache indexer running"
    );

    // ── Wait for Ctrl-C ───────────────────────────────────────────────────
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("Received shutdown signal, stopping velo indexer");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workers() {
        let input = "1=tcp://host:5557,2:1=tcp://host:5558";
        let result = parse_workers(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 0, "tcp://host:5557".to_string()));
        assert_eq!(result[1], (2, 1, "tcp://host:5558".to_string()));
    }

    #[test]
    fn test_parse_workers_empty() {
        assert!(parse_workers("").unwrap().is_empty());
    }

    #[test]
    fn test_parse_workers_invalid_entry() {
        let error = parse_workers("1").unwrap_err().to_string();
        assert!(error.contains("invalid worker entry"));
    }

    #[test]
    fn test_validate_zmq_endpoint_allows_wildcard_tcp_bind() {
        validate_zmq_endpoint("tcp://*:5558").unwrap();
        validate_zmq_endpoint("tcp://127.0.0.1:0").unwrap();
        validate_zmq_endpoint("inproc://listener").unwrap();
        validate_zmq_endpoint("ipc:///tmp/dynamo.sock").unwrap();
    }

    #[test]
    fn test_validate_zmq_endpoint_rejects_invalid_values() {
        assert!(validate_zmq_endpoint("tcp://host").is_err());
        assert!(validate_zmq_endpoint("tcp://:5558").is_err());
        assert!(validate_zmq_endpoint("udp://host:5558").is_err());
        assert!(validate_zmq_endpoint("not-an-endpoint").is_err());
    }
}
