// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;

#[cfg(feature = "kv-indexer-runtime")]
use dynamo_kv_router::standalone_indexer::RuntimeConfig;
use dynamo_kv_router::standalone_indexer::{self, IndexerConfig};

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
    #[cfg(feature = "kv-indexer-runtime")]
    #[arg(long)]
    dynamo_runtime: bool,

    /// Dynamo namespace to register the indexer component under.
    #[cfg(feature = "kv-indexer-runtime")]
    #[arg(long, default_value = "default")]
    namespace: String,

    /// Component name for this indexer in the Dynamo runtime.
    #[cfg(feature = "kv-indexer-runtime")]
    #[arg(long, default_value = "kv-indexer")]
    component_name: String,

    /// Component name that workers register under.
    #[cfg(feature = "kv-indexer-runtime")]
    #[arg(long, default_value = "backend")]
    worker_component: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    #[cfg(feature = "kv-indexer-runtime")]
    if cli.dynamo_runtime {
        dynamo_runtime::logging::init();
        let worker = dynamo_runtime::Worker::from_settings()?;
        return worker.execute(move |runtime| {
            standalone_indexer::run_with_runtime(
                runtime,
                IndexerConfig {
                    block_size: cli.block_size,
                    port: cli.port,
                    threads: cli.threads,
                    workers: cli.workers,
                    model_name: cli.model_name,
                    tenant_id: cli.tenant_id,
                    peers: cli.peers,
                },
                RuntimeConfig {
                    namespace: cli.namespace,
                    component_name: cli.component_name,
                    worker_component: cli.worker_component,
                },
            )
        });
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(standalone_indexer::run_server(IndexerConfig {
        block_size: cli.block_size,
        port: cli.port,
        threads: cli.threads,
        workers: cli.workers,
        model_name: cli.model_name,
        tenant_id: cli.tenant_id,
        peers: cli.peers,
    }))
}
