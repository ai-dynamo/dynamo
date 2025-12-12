// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry Hub Binary
//!
//! Runs the distributed object registry hub for KV cache block coordination.
//!
//! # Usage
//!
//! ```bash
//! # Build and run from this directory
//! cd examples/kvbm/distributed/object-registry
//! cargo run --release
//!
//! # Or build from workspace root
//! cargo build --release -p object-registry-example
//!
//! # Run with custom settings via environment variables
//! DYN_REGISTRY_HUB_CAPACITY=10000000 \
//! DYN_REGISTRY_HUB_QUERY_ADDR=tcp://*:6000 \
//! DYN_REGISTRY_HUB_REGISTER_ADDR=tcp://*:6001 \
//! cargo run --release
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
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use dynamo_llm::block_manager::distributed::registry::{
    RegistryHub, RegistryHubConfig, ZmqRegistryHub,
};

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Load config from environment
    let config = RegistryHubConfig::from_env();

    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║           Distributed Object Registry                        ║");
    info!("╠══════════════════════════════════════════════════════════════╣");
    info!("║  Capacity:        {:<43}║", format!("{} entries", config.capacity));
    info!("║  Query Addr:      {:<43}║", config.query_addr);
    info!("║  Register Addr:   {:<43}║", config.register_addr);
    info!("║  Lease Timeout:   {:<43}║", format!("{} secs", config.lease_timeout.as_secs()));
    info!("╚══════════════════════════════════════════════════════════════╝");

    // Create hub
    let hub = ZmqRegistryHub::new(config)?;

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

