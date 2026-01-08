// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry Hub Binary
//!
//! Runs the distributed object registry hub for KV cache block coordination.
//!
//! # Usage
//!
//! ```bash
//! # Build and run from this directory
//! cd examples/kvbm/distributed/sample-registry
//! cargo run --release
//!
//! # Or build from workspace root
//! cargo build --manifest-path examples/kvbm/distributed/sample-registry/Cargo.toml --release
//!
//! # Run with custom settings via environment variables
//! DYN_REGISTRY_HUB_CAPACITY=10000000 \
//! DYN_REGISTRY_HUB_QUERY_ADDR=tcp://*:6000 \
//! DYN_REGISTRY_HUB_REGISTER_ADDR=tcp://*:6001 \
//! cargo run --manifest-path examples/kvbm/distributed/sample-registry/Cargo.toml --release
//! ```
//!
//! # Environment Variables
//!
//! - `DYN_REGISTRY_HUB_CAPACITY`: Registry capacity (default: 1000000)
//! - `DYN_REGISTRY_HUB_QUERY_ADDR`: Query address (default: tcp://*:5555)
//! - `DYN_REGISTRY_HUB_REGISTER_ADDR`: Register address (default: tcp://*:5556)

use anyhow::Result;
use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::EnvFilter;

use dynamo_llm::block_manager::block::transfer::remote::RemoteKey;
use dynamo_llm::block_manager::distributed::registry::{
    BinaryCodec, NoMetadata, PositionalEviction, PositionalKey, RegistryHubConfig, ZmqHub,
    ZmqHubServerConfig,
};

/// Type alias for the concrete hub type we use.
///
/// Uses `PositionalKey` for position-aware storage and `PositionalEviction`
/// which evicts from highest positions first (tail-first), optimizing for
/// prefix reuse in KV cache scenarios.
type G4RegistryHub = ZmqHub<
    PositionalKey,
    RemoteKey,
    NoMetadata,
    PositionalEviction<PositionalKey, RemoteKey>,
    BinaryCodec<PositionalKey, RemoteKey, NoMetadata>,
>;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    // Load config from environment
    let config = RegistryHubConfig::from_env();

    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║           Distributed Object Registry                        ║");
    info!("╠══════════════════════════════════════════════════════════════╣");
    info!(
        "║  Capacity:        {:<43}║",
        format!("{} entries", config.capacity)
    );
    info!("║  Query Addr:      {:<43}║", config.query_addr);
    info!("║  Register Addr:   {:<43}║", config.register_addr);
    info!(
        "║  Lease Timeout:   {:<43}║",
        format!("{} secs", config.lease_timeout.as_secs())
    );
    info!("╚══════════════════════════════════════════════════════════════╝");

    // Convert to ZmqHubServerConfig
    let zmq_config = ZmqHubServerConfig {
        query_addr: config.query_addr,
        pull_addr: config.register_addr,
        capacity: config.capacity,
    };

    // Create hub with positional eviction storage and codec
    // PositionalEviction evicts from highest positions first (tail-first),
    // which is optimal for KV cache prefix reuse.
    let storage = PositionalEviction::with_capacity(config.capacity as usize);
    let codec = BinaryCodec::new();
    let hub: G4RegistryHub = ZmqHub::new(zmq_config, storage, codec);

    // Setup cancellation
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    // Handle Ctrl+C
    tokio::spawn(async move {
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received Ctrl+C, initiating shutdown...");
                cancel_clone.cancel();
            }
            Err(e) => {
                tracing::error!("Failed to listen for Ctrl+C: {}", e);
            }
        }
    });

    // Run hub
    info!("Starting registry hub... Press Ctrl+C to stop.");
    hub.serve(cancel).await?;

    info!("Registry hub stopped.");
    Ok(())
}

