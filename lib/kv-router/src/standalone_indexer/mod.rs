// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod indexer;
pub mod listener;
pub mod metrics;
pub mod recovery;
pub mod registry;
pub mod server;

use std::sync::Arc;

use tokio::net::TcpListener;

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
        "Starting standalone KV cache indexer"
    );

    let registry = WorkerRegistry::new(config.threads);

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

    registry.signal_ready();

    #[cfg(feature = "metrics")]
    let prom_registry = {
        let r = prometheus::Registry::new();
        metrics::register(&r).expect("failed to register indexer metrics");
        r
    };

    let state = Arc::new(AppState {
        registry: Arc::new(registry),
        #[cfg(feature = "metrics")]
        prom_registry,
    });

    let app = create_router(state);
    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    tracing::info!("HTTP server listening on 0.0.0.0:{}", config.port);
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
