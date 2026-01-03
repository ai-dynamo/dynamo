// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "etcd")]
mod etcd;

#[cfg(feature = "etcd")]
pub use etcd::{EtcdConfig, EtcdConfigBuilder};

#[cfg(feature = "p2p")]
mod p2p;

#[cfg(feature = "p2p")]
pub use p2p::{P2pConfig, P2pConfigBuilder};

#[cfg(test)]
pub(crate) mod test_support;

mod validation;

use std::sync::Arc;

use futures::future::BoxFuture;

pub use crate::peer::PeerDiscovery;
use crate::peer::{
    DiscoveryError, DiscoveryQueryError, InstanceId, PeerInfo, WorkerAddress, WorkerId,
};

/// Validates cluster ID format.
///
/// Cluster IDs must contain only:
/// - Lowercase letters (a-z)
/// - Numbers (0-9)
/// - Hyphens (-)
/// - Underscores (_)
///
/// No uppercase, spaces, slashes, or special characters allowed.
///
/// # Errors
///
/// Returns a validation error if:
/// - The cluster_id is empty
/// - The cluster_id contains invalid characters
pub fn validate_cluster_id(cluster_id: &str) -> Result<(), validator::ValidationError> {
    if cluster_id.is_empty() {
        return Err(validator::ValidationError::new("cluster_id_empty"));
    }

    for ch in cluster_id.chars() {
        if !matches!(ch, 'a'..='z' | '0'..='9' | '-' | '_') {
            return Err(validator::ValidationError::new("cluster_id_invalid_chars"));
        }
    }

    Ok(())
}

/// A [`DiscoverySystem`] should provide one or more concrete implementations of discovery traits in this crate.
pub trait DiscoverySystem: Send + Sync + std::fmt::Debug {
    /// Returns a [`PeerDiscoveryExt`] implementation if available.
    fn peer_discovery(&self) -> Option<Arc<dyn PeerDiscovery>>;

    /// Gracefully shutdown the discovery system.
    ///
    /// This should stop background tasks (like keep-alive), close connections,
    /// and clean up resources. Implementations should make this idempotent.
    ///
    /// Default implementation does nothing (no-op).
    fn shutdown(&self) {
        // Default no-op for implementations that don't need explicit shutdown
    }
}

/// Attach a [`DiscoverySystem`] to its peer discovery implementation while keeping the system alive.
#[allow(dead_code)]
pub(crate) fn peer_discovery_handle(
    system: Arc<dyn DiscoverySystem>,
) -> Option<Arc<dyn PeerDiscovery>> {
    system.peer_discovery().map(|inner| {
        Arc::new(SystemBackedPeerDiscovery::new(system, inner)) as Arc<dyn PeerDiscovery>
    })
}

#[derive(Clone)]
#[allow(dead_code)]
struct SystemBackedPeerDiscovery {
    system: Arc<dyn DiscoverySystem>,
    inner: Arc<dyn PeerDiscovery>,
}

impl SystemBackedPeerDiscovery {
    fn new(system: Arc<dyn DiscoverySystem>, inner: Arc<dyn PeerDiscovery>) -> Self {
        Self { system, inner }
    }
}

impl std::fmt::Debug for SystemBackedPeerDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemBackedPeerDiscovery").finish()
    }
}

impl PeerDiscovery for SystemBackedPeerDiscovery {
    fn discover_by_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        self.inner.discover_by_worker_id(worker_id)
    }

    fn discover_by_instance_id(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        self.inner.discover_by_instance_id(instance_id)
    }

    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        self.inner.register_instance(instance_id, worker_address)
    }

    fn unregister_instance(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        self.inner.unregister_instance(instance_id)
    }
}
