// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone shard server process for the multi-process UDS benchmark.
//!
//! Binds a UDS socket at `--socket-path`, creates a
//! `ThreadPoolIndexer<ConcurrentRadixTreeCompressed>` with `--num-threads`
//! threads and a `--block-size`-token block size, and serves requests until
//! SIGTERM / SIGINT.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p dynamo-kv-router \
//!   --features uds-raw-bench \
//!   --bin kv_shard_server -- \
//!   --socket-path /tmp/shard0.sock \
//!   --num-threads 4 \
//!   --block-size 16
//! ```

use std::path::PathBuf;

use clap::Parser;
use dynamo_kv_router::indexer::ThreadPoolIndexer;
use dynamo_kv_router::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
use dynamo_kv_router::shard_router::RawUdsShardServer;

/// Shard server for the raw-UDS multi-process benchmark.
#[derive(Parser, Debug)]
#[command(name = "kv_shard_server")]
struct Args {
    /// Path to the Unix Domain Socket to bind.
    #[arg(long)]
    socket_path: PathBuf,

    /// Number of indexer worker threads.
    #[arg(long, default_value = "4")]
    num_threads: usize,

    /// Block size in tokens (must match the router's block size).
    #[arg(long, default_value = "16")]
    block_size: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    tracing::info!(
        socket_path = %args.socket_path.display(),
        num_threads = args.num_threads,
        block_size = args.block_size,
        "starting kv_shard_server"
    );

    let trie = ConcurrentRadixTreeCompressed::new();
    let shard = ThreadPoolIndexer::new(trie, args.num_threads, args.block_size);
    let _server = RawUdsShardServer::bind(args.socket_path, shard).await?;

    // Block until SIGTERM or SIGINT.
    tokio::signal::ctrl_c().await?;
    tracing::info!("received shutdown signal, exiting");
    Ok(())
}
