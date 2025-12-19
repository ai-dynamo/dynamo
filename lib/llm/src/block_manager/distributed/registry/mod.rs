// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed object registry for KV cache block coordination.
//!
//! This module provides a distributed registry service for tracking which
//! KV cache blocks have been stored in object storage. It enables:
//!
//! - **Deduplication**: Avoid redundant object writes by checking what already exists
//! - **Matching**: Find which blocks can be loaded from object
//! - **Registration**: Track newly stored blocks
//!
//! # Architecture
//!
//! The registry uses a hub-and-spoke model:
//! - **Hub**: Single coordinator (typically on the leader) holding the registry
//! - **Clients**: Workers connect to the hub to query and register entries
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         REGISTRY SERVICE                                 │
//! │                                                                          │
//! │                      ┌─────────────────────┐                             │
//! │                      │   Registry Hub      │                             │
//! │                      │   (on Leader)       │                             │
//! │                      └──────────┬──────────┘                             │
//! │                                 │                                        │
//! │            ┌────────────────────┼────────────────────┐                   │
//! │            ▼                    ▼                    ▼                   │
//! │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
//! │   │   Worker 0      │  │   Worker 1      │  │   Worker N      │         │
//! │   │ (Registry       │  │ (Registry       │  │ (Registry       │         │
//! │   │  Client)        │  │  Client)        │  │  Client)        │         │
//! │   └─────────────────┘  └─────────────────┘  └─────────────────┘         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Traits
//!
//! The module is designed around traits for flexibility:
//!
//! - [`DistributedRegistry`]: Client-side trait for workers
//! - [`RegistryHub`]: Server-side trait for the coordinator
//!
//! Internal traits (not public):
//! - `RegistryStorage`: Abstracts storage backend (Moka, HashMap, etc.)
//! - `RegistryTransport`: Abstracts network transport (ZMQ, RDMA, etc.)
//!
//! # Implementations
//!
//! ## Hub (Server-side)
//! - [`ZmqRegistryHub`]: ZMQ-based hub using REP/SUB sockets
//!
//! ## Client (Worker-side)
//! - [`ZmqRegistryClient`]: ZMQ-based client using REQ/PUB sockets
//! - [`LocalRegistry`]: In-process registry for testing/single-node
//!
//! # Example
//!
//! ## Starting the Hub
//!
//! ```ignore
//! use dynamo_llm::block_manager::distributed::registry::{
//!     ZmqRegistryHub, RegistryHubConfig, RegistryHub,
//! };
//! use tokio_util::sync::CancellationToken;
//!
//! let config = RegistryHubConfig {
//!     capacity: 1_000_000,
//!     query_addr: "tcp://*:5555".to_string(),
//!     register_addr: "tcp://*:5556".to_string(),
//! };
//!
//! let hub = ZmqRegistryHub::new(config)?;
//! let cancel = CancellationToken::new();
//! hub.serve(cancel).await?;
//! ```
//!
//! ## Using the Client
//!
//! ```ignore
//! use dynamo_llm::block_manager::distributed::registry::{
//!     ZmqRegistryClient, RegistryClientConfig, DistributedRegistry,
//! };
//!
//! let config = RegistryClientConfig::connect_to("leader", 5555, 5556);
//! let client = ZmqRegistryClient::connect(config).await?;
//!
//! // Check what needs to be stored (deduplication)
//! let result = client.can_offload(&hashes).await?;
//! for hash in &result.can_offload {
//!     store_to_object(*hash).await?;
//! }
//! client.register(&result.can_offload).await?;
//!
//! // Find what can be loaded from object
//! let matched = client.match_sequence_hashes(&hashes).await?;
//! for (hash, key) in matched {
//!     load_from_object(key).await?;
//! }
//! ```
//!
//! ## Testing with LocalRegistry
//!
//! ```ignore
//! use dynamo_llm::block_manager::distributed::registry::{
//!     LocalRegistry, DistributedRegistry,
//! };
//!
//! let registry = LocalRegistry::new(100_000);
//!
//! registry.register(&[hash1, hash2]).await?;
//!
//! let result = registry.can_offload(&[hash1, hash2, hash3]).await?;
//! assert_eq!(result.can_offload, vec![hash3]);
//! ```

pub mod client;
pub mod config;
pub mod external;
pub mod hub;
pub mod key_builder;
pub mod object_registry;
pub mod protocol;
pub(crate) mod storage;
pub mod traits;
pub mod types;

// Re-export public types
pub use traits::{DistributedRegistry, RegistryHub};
pub use types::{HubStats, MatchResult, ObjectKey, OffloadResult};

// Re-export config
pub use config::{RegistryClientConfig, RegistryHubConfig};

// Re-export implementations
pub use client::{LocalRegistry, ZmqRegistryClient};
pub use hub::ZmqRegistryHub;

// Re-export protocol for implementations
pub use protocol::{MessageType, SequenceHash};

// Re-export external registry types
pub use external::{ExternalRegistry, RegistryKey, SequenceHashRegistry, SharedExternalRegistry};

// Re-export ObjectRegistry
pub use object_registry::ObjectRegistry;

// Re-export key builder
pub use key_builder::RemoteKeyBuilder;

use std::sync::Arc;

/// Create a distributed registry client from environment variables if enabled.
///
/// Returns `Some(Arc<dyn DistributedRegistry>)` if `DYN_REGISTRY_ENABLE=1` is set,
/// otherwise returns `None`.
///
/// # Environment Variables
///
/// - `DYN_REGISTRY_ENABLE`: Set to "1" or "true" to enable (required)
/// - `DYN_REGISTRY_CLIENT_QUERY_ADDR`: Hub query address (default: tcp://localhost:5555)
/// - `DYN_REGISTRY_CLIENT_REGISTER_ADDR`: Hub register address (default: tcp://localhost:5556)
///
/// # Example
///
/// ```bash
/// # Enable distributed registry
/// DYN_REGISTRY_ENABLE=1
/// DYN_REGISTRY_CLIENT_QUERY_ADDR=tcp://leader:5555
/// DYN_REGISTRY_CLIENT_REGISTER_ADDR=tcp://leader:5556
/// ```
///
/// # Usage
///
/// ```ignore
/// if let Some(registry) = create_registry_from_env().await {
///     // Use distributed registry for deduplication
/// }
/// ```
pub async fn create_registry_from_env() -> Option<Arc<dyn DistributedRegistry>> {
    if !RegistryClientConfig::is_enabled() {
        tracing::info!(
            "Distributed registry NOT enabled. Set DYN_REGISTRY_ENABLE=1 to enable cross-worker deduplication."
        );
        return None;
    }

    let config = RegistryClientConfig::from_env();

    tracing::info!(
        "Connecting to distributed registry hub: query={}, register={}",
        config.hub_query_addr,
        config.hub_register_addr
    );
    tracing::info!(
        "Make sure the registry-hub binary is running! Start with: cargo run -p dynamo-llm --features block-manager --bin registry-hub"
    );

    match ZmqRegistryClient::connect(config).await {
        Ok(client) => {
            tracing::info!("✓ Distributed registry client connected successfully");
            Some(Arc::new(client) as Arc<dyn DistributedRegistry>)
        }
        Err(e) => {
            tracing::error!(
                "✗ Failed to connect to distributed registry hub: {}. \
                 Continuing WITHOUT distributed deduplication. \
                 Make sure registry-hub is running on the configured addresses.",
                e
            );
            None
        }
    }
}
