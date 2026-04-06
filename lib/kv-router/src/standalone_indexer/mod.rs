// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod indexer;
pub mod listener;
pub mod metrics;
pub mod recovery;
pub mod registry;
#[cfg(feature = "indexer-runtime")]
pub mod runtime;
pub mod server;
mod zmq;

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

#[cfg(feature = "indexer-runtime")]
pub struct RuntimeConfig {
    pub namespace: String,
    pub component_name: String,
    pub worker_component: String,
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

#[cfg(feature = "indexer-runtime")]
pub async fn run_with_runtime(
    runtime: dynamo_runtime::Runtime,
    config: IndexerConfig,
    runtime_config: RuntimeConfig,
) -> anyhow::Result<()> {
    use dynamo_runtime::{
        DistributedRuntime,
        pipeline::{ManyOut, SingleIn, network::Ingress},
    };

    use crate::indexer::{IndexerQueryRequest, IndexerQueryResponse, KV_INDEXER_QUERY_ENDPOINT};

    let distributed_runtime = DistributedRuntime::from_settings(runtime).await?;
    let cancel_token = distributed_runtime.primary_token();
    let component = distributed_runtime
        .namespace(&runtime_config.namespace)?
        .component(&runtime_config.component_name)?;

    tracing::info!(
        namespace = %runtime_config.namespace,
        component = %runtime_config.component_name,
        block_size = ?config.block_size,
        port = config.port,
        threads = config.threads,
        model_name = %config.model_name,
        tenant_id = %config.tenant_id,
        worker_component = %runtime_config.worker_component,
        num_peers = config.peers.as_ref().map(|p| p.split(',').count()).unwrap_or(0),
        "Starting standalone KV cache indexer (Dynamo runtime mode)"
    );

    let registry = Arc::new(WorkerRegistry::new(config.threads));
    let engine = Arc::new(runtime::query_engine::IndexerQueryEngine {
        registry: registry.clone(),
    });
    let ingress =
        Ingress::<SingleIn<IndexerQueryRequest>, ManyOut<IndexerQueryResponse>>::for_engine(
            engine,
        )?;
    let query_endpoint = component
        .endpoint(KV_INDEXER_QUERY_ENDPOINT)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true);

    distributed_runtime.runtime().secondary().spawn(async move {
        if let Err(err) = query_endpoint.start().await {
            tracing::error!(error = %err, "Query endpoint failed");
        }
    });

    tracing::info!(
        endpoint = KV_INDEXER_QUERY_ENDPOINT,
        "Query endpoint registered"
    );

    runtime::discovery::spawn_discovery_watcher(
        &distributed_runtime,
        registry.clone(),
        cancel_token.clone(),
    )
    .await?;
    runtime::subscriber::spawn_event_subscriber(
        &distributed_runtime,
        &runtime_config.namespace,
        &runtime_config.worker_component,
        registry.clone(),
        cancel_token.clone(),
    )
    .await?;

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
