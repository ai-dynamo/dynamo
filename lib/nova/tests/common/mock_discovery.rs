// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock discovery backend for testing.

use dashmap::DashMap;
use dynamo_nova_discovery::peer::{
    AwaitableQueryResult, AwaitableRegisterResult, DiscoveryError, DiscoveryQueryError, InstanceId,
    PeerDiscovery, PeerInfo, WorkerAddress, WorkerId,
};
use futures::FutureExt;
use std::sync::Arc;

/// Mock discovery backend that stores peers in memory.
///
/// This is useful for testing discovery-driven workflows without requiring
/// external services like etcd or a filesystem.
#[derive(Debug, Clone)]
pub struct MockDiscovery {
    /// Stores worker_id -> PeerInfo
    workers: Arc<DashMap<WorkerId, PeerInfo>>,
    /// Stores instance_id -> PeerInfo
    instances: Arc<DashMap<InstanceId, PeerInfo>>,
}

impl MockDiscovery {
    /// Create a new empty MockDiscovery backend.
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            instances: Arc::new(DashMap::new()),
        }
    }

    /// Populate the discovery with a peer.
    ///
    /// This is useful for test setup where you want to pre-populate the discovery
    /// system with known peers.
    pub fn populate(&self, peer_info: PeerInfo) {
        let instance_id = peer_info.instance_id();
        let worker_id = peer_info.worker_id();
        self.workers.insert(worker_id, peer_info.clone());
        self.instances.insert(instance_id, peer_info);
    }

    /// Clear all registered peers.
    #[expect(dead_code)]
    pub fn clear(&self) {
        self.workers.clear();
        self.instances.clear();
    }

    /// Get the number of registered peers.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Check if the discovery is empty.
    #[expect(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

impl Default for MockDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerDiscovery for MockDiscovery {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> AwaitableQueryResult {
        let workers = self.workers.clone();
        async move {
            workers
                .get(&worker_id)
                .map(|entry| entry.clone())
                .ok_or(DiscoveryQueryError::NotFound)
        }
        .boxed()
    }

    fn discover_by_instance_id(&self, instance_id: InstanceId) -> AwaitableQueryResult {
        let instances = self.instances.clone();
        async move {
            instances
                .get(&instance_id)
                .map(|entry| entry.clone())
                .ok_or(DiscoveryQueryError::NotFound)
        }
        .boxed()
    }

    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> AwaitableRegisterResult {
        let workers = self.workers.clone();
        let instances = self.instances.clone();

        async move {
            let worker_id = instance_id.worker_id();
            let peer_info = PeerInfo::new(instance_id, worker_address);

            // Check for instance collision
            if instances.contains_key(&instance_id) {
                return Err(DiscoveryError::InstanceAlreadyRegistered(instance_id));
            }

            // Check for worker_id collision (different instance with same worker_id)
            if let Some(existing) = workers.get(&worker_id) {
                let existing_instance = existing.instance_id();
                if existing_instance != instance_id {
                    return Err(DiscoveryError::WorkerIdCollision(
                        worker_id,
                        existing_instance,
                        instance_id,
                    ));
                }
            }

            // Register
            workers.insert(worker_id, peer_info.clone());
            instances.insert(instance_id, peer_info);

            Ok(())
        }
        .boxed()
    }

    fn unregister_instance(&self, instance_id: InstanceId) -> AwaitableRegisterResult {
        let workers = self.workers.clone();
        let instances = self.instances.clone();

        async move {
            let worker_id = instance_id.worker_id();

            workers.remove(&worker_id);
            instances.remove(&instance_id);

            Ok(())
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_address() -> WorkerAddress {
        // Create a simple test address using the builder
        let mut builder = WorkerAddress::builder();
        builder
            .add_entry("tcp", bytes::Bytes::from_static(b"tcp://127.0.0.1:12345"))
            .unwrap();
        builder.build().unwrap()
    }

    #[tokio::test]
    async fn test_mock_discovery_register_and_discover() {
        let mock = MockDiscovery::new();
        let instance_id = InstanceId::new_v4();
        let address = create_test_address();

        // Register
        mock.register_instance(instance_id, address.clone())
            .await
            .unwrap();

        // Discover by worker_id
        let worker_id = instance_id.worker_id();
        let found = mock.discover_by_worker_id(worker_id).await.unwrap();
        assert_eq!(found.instance_id(), instance_id);
        assert_eq!(&found.worker_address, &address);

        // Discover by instance_id
        let found = mock.discover_by_instance_id(instance_id).await.unwrap();
        assert_eq!(found.instance_id(), instance_id);
        assert_eq!(&found.worker_address, &address);
    }

    #[tokio::test]
    async fn test_mock_discovery_not_found() {
        let mock = MockDiscovery::new();
        let instance_id = InstanceId::new_v4();
        let worker_id = instance_id.worker_id();

        // Should not find unregistered peers
        let result = mock.discover_by_worker_id(worker_id).await;
        assert!(matches!(result, Err(DiscoveryQueryError::NotFound)));

        let result = mock.discover_by_instance_id(instance_id).await;
        assert!(matches!(result, Err(DiscoveryQueryError::NotFound)));
    }

    #[tokio::test]
    async fn test_mock_discovery_collision_detection() {
        let mock = MockDiscovery::new();
        let instance_id = InstanceId::new_v4();
        let address = create_test_address();

        // First registration succeeds
        mock.register_instance(instance_id, address.clone())
            .await
            .unwrap();

        // Second registration of same instance fails
        let result = mock.register_instance(instance_id, address).await;
        assert!(matches!(
            result,
            Err(DiscoveryError::InstanceAlreadyRegistered(_))
        ));
    }

    #[tokio::test]
    async fn test_mock_discovery_populate() {
        let mock = MockDiscovery::new();
        let instance_id = InstanceId::new_v4();
        let address = create_test_address();
        let peer_info = PeerInfo::new(instance_id, address.clone());

        // Populate directly (bypasses collision checks for test setup)
        mock.populate(peer_info);

        // Verify it's discoverable
        let worker_id = instance_id.worker_id();
        let found = mock.discover_by_worker_id(worker_id).await.unwrap();
        assert_eq!(found.instance_id(), instance_id);
    }

    #[tokio::test]
    async fn test_mock_discovery_unregister() {
        let mock = MockDiscovery::new();
        let instance_id = InstanceId::new_v4();
        let address = create_test_address();

        // Register
        mock.register_instance(instance_id, address).await.unwrap();
        assert_eq!(mock.len(), 1);

        // Unregister
        mock.unregister_instance(instance_id).await.unwrap();
        assert_eq!(mock.len(), 0);

        // Should not be discoverable
        let worker_id = instance_id.worker_id();
        let result = mock.discover_by_worker_id(worker_id).await;
        assert!(matches!(result, Err(DiscoveryQueryError::NotFound)));
    }
}
