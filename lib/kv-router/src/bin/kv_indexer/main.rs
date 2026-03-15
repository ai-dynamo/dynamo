// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::standalone_indexer::{
    self, recovery,
    registry::WorkerRegistry,
    server::{AppState, create_router},
};

#[cfg(feature = "indexer-runtime")]
mod runtime;

#[derive(Parser)]
#[command(name = "dynamo-kv-indexer", about = "Standalone KV cache indexer")]
struct Cli {
    /// KV cache block size for initial workers registered via --workers
    #[arg(long)]
    block_size: Option<u32>,

    /// HTTP server port
    #[arg(long, default_value_t = 8090)]
    port: u16,

    /// Number of indexer threads (1 = single-threaded KvIndexer, >1 = ThreadPoolIndexer)
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Initial workers as "worker_id[:dp_rank]=zmq_address,..." (e.g. "1=tcp://host:5557,1:1=tcp://host:5558")
    #[arg(long)]
    workers: Option<String>,

    /// Model name for initial workers registered via --workers
    #[arg(long, default_value = "default")]
    model_name: String,

    /// Tenant ID for initial workers registered via --workers
    #[arg(long, default_value = "default")]
    tenant_id: String,

    /// Comma-separated peer URLs for P2P recovery (e.g. "http://host1:8090,http://host2:8091")
    #[arg(long)]
    peers: Option<String>,

    /// Enable Dynamo runtime integration (discovery, event plane, request plane).
    /// When enabled, workers are discovered via MDC and events arrive via the event plane.
    /// Also enables router to configure a remote indexer via the request plane.
    #[cfg(feature = "indexer-runtime")]
    #[arg(long)]
    dynamo_runtime: bool,

    /// Dynamo namespace to register the indexer component under
    #[cfg(feature = "indexer-runtime")]
    #[arg(long, default_value = "default")]
    namespace: String,

    /// Component name for this indexer in the Dynamo runtime
    #[cfg(feature = "indexer-runtime")]
    #[arg(long, default_value = "kv-indexer")]
    component_name: String,

    /// Component name that workers register under (for event plane subscription)
    #[cfg(feature = "indexer-runtime")]
    #[arg(long, default_value = "backend")]
    worker_component: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    #[cfg(feature = "indexer-runtime")]
    if cli.dynamo_runtime {
        // Full Dynamo runtime mode: discovery, event plane, request plane
        dynamo_runtime::logging::init();
        let worker = dynamo_runtime::Worker::from_settings()?;
        return worker.execute(move |runtime| app_with_runtime(runtime, cli));
    }

    // Standalone HTTP-only mode: no runtime dependencies
    tracing_subscriber::fmt::init();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(app_standalone(cli))
}

async fn app_standalone(cli: Cli) -> anyhow::Result<()> {
    let cancel_token = CancellationToken::new();

    // Install signal handler for graceful shutdown
    let shutdown_token = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Received shutdown signal");
        shutdown_token.cancel();
    });

    tracing::info!(
        block_size = ?cli.block_size,
        port = cli.port,
        threads = cli.threads,
        model_name = %cli.model_name,
        tenant_id = %cli.tenant_id,
        num_peers = cli.peers.as_ref().map(|p| p.split(',').count()).unwrap_or(0),
        "Starting standalone KV cache indexer (HTTP-only mode)"
    );

    let registry = Arc::new(WorkerRegistry::new(cli.threads));

    run_common(&cli, &registry, cancel_token).await
}

#[cfg(feature = "indexer-runtime")]
async fn app_with_runtime(runtime: dynamo_runtime::Runtime, cli: Cli) -> anyhow::Result<()> {
    use dynamo_kv_router::indexer::{
        IndexerQueryRequest, IndexerQueryResponse, KV_INDEXER_QUERY_ENDPOINT,
    };
    use dynamo_runtime::{
        DistributedRuntime,
        pipeline::{ManyOut, SingleIn, network::Ingress},
    };

    let distributed_runtime = DistributedRuntime::from_settings(runtime).await?;
    let cancel_token = distributed_runtime.primary_token();

    let component = distributed_runtime
        .namespace(&cli.namespace)?
        .component(&cli.component_name)?;

    tracing::info!(
        namespace = %cli.namespace,
        component = %cli.component_name,
        block_size = ?cli.block_size,
        port = cli.port,
        threads = cli.threads,
        model_name = %cli.model_name,
        tenant_id = %cli.tenant_id,
        worker_component = %cli.worker_component,
        num_peers = cli.peers.as_ref().map(|p| p.split(',').count()).unwrap_or(0),
        "Starting standalone KV cache indexer (Dynamo runtime mode)"
    );

    let registry = Arc::new(WorkerRegistry::new(cli.threads));

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
        if let Err(e) = query_endpoint.start().await {
            tracing::error!(error = %e, "Query endpoint failed");
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
        &cli.namespace,
        &cli.worker_component,
        registry.clone(),
        cancel_token.clone(),
    )
    .await?;

    run_common(&cli, &registry, cancel_token).await
}

/// Shared logic for both standalone and runtime modes:
/// register CLI workers, P2P recovery, signal ready, start HTTP server.
async fn run_common(
    cli: &Cli,
    registry: &Arc<WorkerRegistry>,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    if let Some(ref workers_str) = cli.workers {
        let block_size = cli.block_size.ok_or_else(|| {
            anyhow::anyhow!("--block-size is required when --workers is specified")
        })?;
        for (instance_id, dp_rank, endpoint) in standalone_indexer::parse_workers(workers_str) {
            tracing::info!(instance_id, dp_rank, endpoint, "Registering initial worker");
            registry
                .register(
                    instance_id,
                    endpoint,
                    dp_rank,
                    cli.model_name.clone(),
                    cli.tenant_id.clone(),
                    block_size,
                    None,
                )
                .await?;
        }
    }

    let peers: Vec<String> = cli
        .peers
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.trim().to_string())
                .collect()
        })
        .unwrap_or_default();

    // P2P recovery: fetch dump from a peer before starting ZMQ listeners.
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

    // Signal ready — unblocks all ZMQ listeners to start draining buffered events
    registry.signal_ready();

    #[cfg(feature = "metrics")]
    let prom_registry = {
        let r = prometheus::Registry::new();
        dynamo_kv_router::standalone_indexer::metrics::register(&r)
            .expect("failed to register indexer metrics");
        r
    };

    let state = Arc::new(AppState {
        registry: registry.clone(),
        #[cfg(feature = "metrics")]
        prom_registry,
    });

    let app = create_router(state);
    let listener = TcpListener::bind(("0.0.0.0", cli.port)).await?;
    tracing::info!("HTTP server listening on 0.0.0.0:{}", cli.port);

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
    use dynamo_kv_router::standalone_indexer::parse_workers;

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
