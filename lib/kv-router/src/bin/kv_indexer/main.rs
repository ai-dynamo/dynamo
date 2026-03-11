// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

mod indexer;
mod listener;
mod recovery;
mod registry;
mod server;

use registry::WorkerRegistry;
use server::{AppState, create_router};

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
}

fn parse_workers(s: &str) -> Vec<(u64, u32, String)> {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

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

    tracing::info!(
        block_size = ?cli.block_size,
        port = cli.port,
        threads = cli.threads,
        model_name = %cli.model_name,
        tenant_id = %cli.tenant_id,
        num_peers = peers.len(),
        "Starting standalone KV cache indexer"
    );

    let registry = WorkerRegistry::new(cli.threads);

    // Register initial workers — connects ZMQ sockets but listeners wait
    // for the ready signal. This ensures ZMQ subscription handshakes begin
    // before P2P recovery fetches the dump from a peer.
    if let Some(ref workers_str) = cli.workers {
        let block_size = cli.block_size.ok_or_else(|| {
            anyhow::anyhow!("--block-size is required when --workers is specified")
        })?;
        for (instance_id, dp_rank, endpoint) in parse_workers(workers_str) {
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

    // P2P recovery: fetch dump from a peer before starting ZMQ listeners.
    // The 1s delay inside recover_from_peers ensures the peer's tree has
    // advanced past our ZMQ connection floor before we fetch the dump.
    if !peers.is_empty() {
        match recovery::recover_from_peers(&peers, &registry).await {
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

    let state = Arc::new(AppState { registry });

    let app = create_router(state);
    let listener = TcpListener::bind(("0.0.0.0", cli.port)).await?;
    tracing::info!("HTTP server listening on 0.0.0.0:{}", cli.port);
    axum::serve(listener, app).await?;

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
