// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer discovery for the Dynamo Active Message system.
//!
//! This crate provides peer discovery functionality for translating worker IDs
//! and instance IDs to network addresses. It supports multiple discovery backends:
//!
//! - **InMemory**: Fast in-memory discovery for testing and single-node deployments
//! - **HTTP**: Client for discovery HTTP service (feature: `http-service`)
//! - **Libp2p**: Decentralized DHT-based discovery (feature: `p2p`)
//! - **Etcd**: Centralized discovery with TTL and coordination (feature: `etcd`)
//!
//! # Architecture
//!
//! Discovery is transport-agnostic, storing addresses as opaque bytes. The active
//! message runtime deserializes and validates transport compatibility.
//!
//! # Example
//!
//! ```
//! use dynamo_am_discovery::{InstanceId, WorkerAddress, PeerInfo, InMemoryDiscovery, PeerDiscovery};
//! use bytes::Bytes;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! // Create discovery backend
//! let discovery = InMemoryDiscovery::new();
//!
//! // Register a peer
//! let instance_id = InstanceId::new_v4();
//! let address = WorkerAddress::from_bytes(Bytes::from_static(b"tcp://127.0.0.1:5555"));
//! discovery.register(instance_id, address.clone()).await?;
//!
//! // Look up by worker ID
//! let worker_id = instance_id.worker_id();
//! let peer_info = discovery.discover_by_worker_id(worker_id).await?;
//!
//! assert_eq!(peer_info.instance_id(), instance_id);
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use futures::future::BoxFuture;
use std::fmt;
use std::sync::Arc;

mod address;
mod identity;

// pub mod system;
pub mod manager;

// #[cfg(feature = "etcd")]
// pub mod etcd;

// #[cfg(feature = "p2p")]
// pub mod p2p;

// #[cfg(feature = "http-service")]
// pub mod http;

// #[cfg(feature = "http-service")]
// pub mod http_client;

// pub mod profiles;

pub use address::{PeerInfo, WorkerAddress};
pub use identity::{InstanceFactory, InstanceId, WorkerId};

// #[cfg(feature = "etcd")]
// pub use etcd::{EtcdConfig, EtcdConfigBuilder};

// #[cfg(feature = "p2p")]
// pub use p2p::{P2pConfig, P2pConfigBuilder};

// #[cfg(feature = "http-service")]
// pub use http::{
//     BootstrapPeer, BootstrapPeersResponse, ErrorResponse, HealthResponse, PeerInfoResponse,
//     PeerListResponse, RegisterRequest, RegisterResponse, endpoints,
// };

/// Error type for discovery operations.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// Worker ID collision detected - same worker_id registered to different instance
    #[error(
        "Worker ID collision: worker_id {0} already registered to instance {1}, attempted to register to {2}"
    )]
    WorkerIdCollision(WorkerId, InstanceId, InstanceId),

    /// Address checksum mismatch during re-registration
    #[error("Address checksum mismatch for instance {0}: existing=0x{1:016x}, new=0x{2:016x}")]
    ChecksumMismatch(InstanceId, u64, u64),

    /// Backend-specific error
    #[error("Backend error: {0}")]
    Backend(#[from] anyhow::Error),
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DiscoveryQueryError {
    #[error("Not found")]
    NotFound,

    #[error("Backend error: {0}")]
    Backend(Arc<anyhow::Error>),
}

/// Trait for discovering peers by worker_id or instance_id.
///
/// Implementations provide different strategies for peer discovery:
/// - In-memory: Fast local cache for testing
/// - Etcd: Centralized discovery with TTL and coordination
/// - Libp2p-kad: Decentralized DHT-based discovery
/// - HTTP: Client for discovery HTTP service
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to support concurrent access
/// from multiple event system threads.
///
/// # Transport Agnostic
///
/// Discovery returns [`PeerInfo`] with opaque byte addresses. The runtime
/// layer deserializes and validates transport compatibility.
pub trait PeerDiscovery: Send + Sync + fmt::Debug {
    /// Lookup peer by worker_id.
    ///
    /// # Arguments
    /// * `worker_id` - The 64-bit worker identifier
    ///
    /// # Returns
    /// * `Ok(PeerInfo)` - Discovered peer with address
    /// * `Err(DiscoveryError::NotFound)` - Worker ID not registered
    fn discover_by_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>>;

    /// Lookup peer by instance_id.
    ///
    /// # Arguments
    /// * `instance_id` - The UUID used for transport-level addressing
    ///
    /// # Returns
    /// * `Ok(PeerInfo)` - Discovered peer with address
    /// * `Err(DiscoveryError::NotFound)` - Instance ID not registered
    fn discover_by_instance_id(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>>;

    /// Register this peer in the discovery system.
    ///
    /// # Arguments
    /// * `instance_id` - The instance ID to register
    /// * `worker_address` - Worker address (transport-agnostic)
    ///
    /// # Returns
    /// * `Ok(())` - Successfully registered
    /// * `Err(DiscoveryError::WorkerIdCollision)` - Worker ID collision detected
    /// * `Err(DiscoveryError::ChecksumMismatch)` - Re-registration with different address
    ///
    /// # Checksum Validation
    ///
    /// When re-registering the same instance_id, the implementation validates that
    /// the address checksum matches the stored checksum. This prevents accidental
    /// overwrites with different addresses.
    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>>;

    /// Unregister this peer from the discovery system.
    ///
    /// Called during graceful shutdown to remove this peer from discovery.
    ///
    /// # Arguments
    /// * `instance_id` - The UUID of the peer to unregister
    ///
    /// # Returns
    /// * `Ok(())` - Successfully unregistered (or was not registered)
    fn unregister_instance(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>>;
}
