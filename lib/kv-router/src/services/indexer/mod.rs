// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone HTTP KV-cache indexer.
//!
//! Hosts an Axum HTTP server with `/register`, `/unregister`, `/query`,
//! `/query_by_hash`, and peer-discovery routes that workers / gateways can
//! call to drive cache-aware routing decisions. Each registered worker spawns
//! a ZMQ listener that ingests its KV events into a per-(model, routing group)
//! [`backend::Indexer`].
//!
//! `IndexerKey { model_name, routing_group }` remains this service's sole registry authority.
//! Resolving it to `IndexerDomainId` is intentionally deferred until registration can carry
//! authoritative explicit identity material; hashing local defaults here would create a second
//! authority without enabling safe cross-service identity.
//!
//! ## Multi-tier responses
//!
//! `/query` and `/query_by_hash` return both:
//! - the legacy flat `scores`/`frequencies`/`tree_sizes` (device-tier overlap),
//!   for backward compatibility, and
//! - a per-instance `instances` map keyed by `instance_id` with `gpu`, `cpu`,
//!   `disk`, per-`dp_rank` device counts, and `longest_matched`.
//!
//! The `instances` shape is intended to align with Mooncake's
//! "[RFC]: KV-Store Indexer API Standardization"
//! (<https://github.com/kvcache-ai/Mooncake/issues/1403>).
//! Tier counts are CUMULATIVE through each tier's walk — see the doc on the
//! response struct in [`server`] for the exact semantics.

pub mod backend;
pub mod listener;
pub mod logging;
pub mod metrics;
pub mod recovery;
pub mod registry;
pub mod server;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::config::min_initial_workers_from_env;
use crate::services::common::zmq::validate_endpoint as validate_zmq_endpoint;
use axum::http::header::HeaderName;
use logging::AccessLogSink;
use registry::WorkerRegistry;
use server::{AppState, create_router};

pub struct IndexerConfig {
    pub block_size: Option<u32>,
    pub port: u16,
    pub threads: usize,
    pub workers: Option<String>,
    pub model_name: String,
    pub routing_group: String,
    pub peers: Option<String>,
    pub access_log: Option<PathBuf>,
    pub trace_id_header: HeaderName,
    pub access_log_local_time: bool,
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
        routing_group = %config.routing_group,
        num_peers = peers.len(),
        "Starting standalone KV cache indexer (HTTP-only mode)"
    );

    let mut state = AppState::new_with_cancel_token(config.threads, cancel_token.clone())?;
    state.access_log_sink = match config.access_log {
        Some(ref path) => {
            let s = AccessLogSink::new(
                path,
                config.trace_id_header.clone(),
                config.access_log_local_time,
            )
            .map_err(|e| anyhow::anyhow!("failed to open access log {}: {e}", path.display()))?;
            Some(Arc::new(s))
        }
        None => None,
    };
    let state = Arc::new(state);
    run_common(&config, state, cancel_token).await
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
    state: Arc<AppState>,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    let registry = &state.registry;

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
                    config.routing_group.clone(),
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
