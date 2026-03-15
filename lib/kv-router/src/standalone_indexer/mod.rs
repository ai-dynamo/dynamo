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

use std::sync::Arc;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

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

pub fn parse_workers(s: &str) -> Vec<(u64, u32, String)> {
    s.split(',')
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| {
            let (id_part, addr) = entry.split_once('=')?;
            let id_part = id_part.trim();
            let (id, dp_rank) = if let Some((id_str, rank_str)) = id_part.split_once(':') {
                (id_str.parse::<u64>().ok()?, rank_str.parse::<u32>().ok()?)
            } else {
                (id_part.parse::<u64>().ok()?, 0)
            };
            Some((id, dp_rank, addr.trim().to_string()))
        })
        .collect()
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

async fn run_common(
    config: &IndexerConfig,
    registry: &Arc<WorkerRegistry>,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    if let Some(ref workers_str) = config.workers {
        let block_size = config.block_size.ok_or_else(|| {
            anyhow::anyhow!("--block-size is required when --workers is specified")
        })?;
        for (instance_id, dp_rank, endpoint) in parse_workers(workers_str) {
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
        let result = parse_workers(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 0, "tcp://host:5557".to_string()));
        assert_eq!(result[1], (2, 1, "tcp://host:5558".to_string()));
    }

    #[test]
    fn test_parse_workers_empty() {
        assert!(parse_workers("").is_empty());
    }
}
