// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

mod indexer;
mod listener;
mod registry;
mod server;

use indexer::create_indexer;
use registry::WorkerRegistry;
use server::{AppState, create_router};

#[derive(Parser)]
#[command(name = "dynamo-kv-indexer", about = "Standalone KV cache indexer")]
struct Cli {
    /// KV cache block size (must match the vLLM engine's block size)
    #[arg(long)]
    block_size: u32,

    /// HTTP server port
    #[arg(long, default_value_t = 8090)]
    port: u16,

    /// Number of indexer threads (1 = single-threaded KvIndexer, >1 = ThreadPoolIndexer)
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Initial workers as "worker_id=zmq_address,..." (e.g. "1=tcp://host:5557,2=tcp://host:5558")
    #[arg(long)]
    workers: Option<String>,
}

fn parse_workers(s: &str) -> Vec<(u64, String)> {
    s.split(',')
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| {
            let (id_str, addr) = entry.split_once('=')?;
            let id = id_str.trim().parse::<u64>().ok()?;
            Some((id, addr.trim().to_string()))
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

    tracing::info!(
        block_size = cli.block_size,
        port = cli.port,
        threads = cli.threads,
        "Starting standalone KV cache indexer"
    );

    let indexer = create_indexer(cli.block_size, cli.threads);
    let registry = WorkerRegistry::new(indexer, cli.block_size);

    if let Some(ref workers_str) = cli.workers {
        for (worker_id, zmq_address) in parse_workers(workers_str) {
            tracing::info!(worker_id, zmq_address, "Registering initial worker");
            let mut addrs = std::collections::HashMap::new();
            addrs.insert(0, zmq_address);
            registry.register(worker_id, addrs)?;
        }
    }

    let state = Arc::new(AppState {
        registry,
        block_size: cli.block_size,
    });

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
        let input = "1=tcp://host:5557,2=tcp://host:5558";
        let result = parse_workers(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, "tcp://host:5557".to_string()));
        assert_eq!(result[1], (2, "tcp://host:5558".to_string()));
    }

    #[test]
    fn test_parse_workers_empty() {
        assert!(parse_workers("").is_empty());
    }
}
