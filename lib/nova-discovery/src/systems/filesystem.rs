// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Filesystem-based peer discovery.
//!
//! This backend stores peer information in a JSON file on disk, providing
//! simple persistence for development and testing scenarios. While suitable for
//! lightweight deployments and CI/CD environments, consider etcd or p2p backends
//! for production deployments requiring high availability or cross-host coordination.

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::peer::{
    DiscoveryError, DiscoveryQueryError, InstanceId, PeerDiscovery, PeerInfo, WorkerAddress,
    WorkerId,
};

/// Serializable peer registry for filesystem storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PeerRegistry {
    peers: Vec<SerializedPeerInfo>,
}

/// Serialized representation of PeerInfo for JSON storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedPeerInfo {
    instance_id: String,
    worker_id: u64,
    worker_address: WorkerAddress,
    address_checksum: u64,
}

impl SerializedPeerInfo {
    fn from_peer_info(peer_info: &PeerInfo) -> Self {
        Self {
            instance_id: peer_info.instance_id().to_string(),
            worker_id: peer_info.worker_id().as_u64(),
            worker_address: peer_info.worker_address().clone(),
            address_checksum: peer_info.address_checksum(),
        }
    }

    fn to_peer_info(&self) -> Result<PeerInfo> {
        let uuid =
            uuid::Uuid::parse_str(&self.instance_id).context("Failed to parse instance_id")?;
        let instance_id = InstanceId::from_uuid(uuid);
        Ok(PeerInfo::new(instance_id, self.worker_address.clone()))
    }
}

/// Filesystem-based peer discovery backend.
///
/// Stores peer information in a JSON file. Provides simple persistence for
/// testing scenarios where external dependencies (etcd, p2p) are not desired.
///
/// # File Format
///
/// The file contains a JSON object with a `peers` array:
/// ```json
/// {
///   "peers": [
///     {
///       "instance_id": "uuid-string",
///       "worker_id": 123,
///       "worker_address_bytes": [1, 2, 3, ...],
///       "address_checksum": 12345678
///     }
///   ]
/// }
/// ```
///
/// # Concurrency
///
/// Uses file-based locking for concurrent access across processes.
/// Within a single process, uses RwLock for thread safety.
#[derive(Debug, Clone)]
pub struct FilesystemPeerDiscovery {
    file_path: PathBuf,
    inner: Arc<RwLock<FilesystemPeerDiscoveryInner>>,
}

#[derive(Debug)]
struct FilesystemPeerDiscoveryInner {
    by_worker_id: HashMap<WorkerId, InstanceId>,
    by_instance_id: HashMap<InstanceId, PeerInfo>,
}

impl FilesystemPeerDiscoveryInner {
    fn new() -> Self {
        Self {
            by_worker_id: HashMap::new(),
            by_instance_id: HashMap::new(),
        }
    }
}

impl FilesystemPeerDiscovery {
    /// Create a new filesystem-based peer discovery at the specified path.
    ///
    /// If the file exists, it will be loaded. If it doesn't exist, it will be created.
    pub fn new(file_path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = file_path.into();
        let inner = Arc::new(RwLock::new(FilesystemPeerDiscoveryInner::new()));

        let discovery = Self { file_path, inner };

        // Load existing data if file exists
        if discovery.file_path.exists() {
            discovery.load_from_disk()?;
        }

        Ok(discovery)
    }

    /// Create a new filesystem-based peer discovery in a temporary directory.
    ///
    /// Useful for testing. The file will be automatically cleaned up when the
    /// temp directory is dropped (at process exit).
    pub fn new_temp() -> Result<Self> {
        let temp_dir = std::env::temp_dir();
        let file_name = format!("dynamo-discovery-{}.json", uuid::Uuid::new_v4());
        let file_path = temp_dir.join(file_name);
        Self::new(file_path)
    }

    /// Load peer registry from disk.
    fn load_from_disk(&self) -> Result<()> {
        // If file doesn't exist, that's okay - it will be created on first save
        if !self.file_path.exists() {
            return Ok(());
        }

        let content =
            std::fs::read_to_string(&self.file_path).context("Failed to read discovery file")?;

        let registry: PeerRegistry =
            serde_json::from_str(&content).context("Failed to parse discovery file")?;

        let mut inner = self.inner.write();
        inner.by_worker_id.clear();
        inner.by_instance_id.clear();

        for serialized in registry.peers {
            let peer_info = serialized.to_peer_info()?;
            let instance_id = peer_info.instance_id();
            let worker_id = peer_info.worker_id();

            inner.by_worker_id.insert(worker_id, instance_id);
            inner.by_instance_id.insert(instance_id, peer_info);
        }

        Ok(())
    }

    /// Save peer registry to disk.
    fn save_to_disk(&self) -> Result<()> {
        let inner = self.inner.read();

        let peers: Vec<SerializedPeerInfo> = inner
            .by_instance_id
            .values()
            .map(SerializedPeerInfo::from_peer_info)
            .collect();

        let registry = PeerRegistry { peers };

        let content =
            serde_json::to_string_pretty(&registry).context("Failed to serialize registry")?;

        // Write to temp file first, then rename for atomicity
        let temp_path = self.file_path.with_extension("tmp");
        std::fs::write(&temp_path, content).context("Failed to write temp file")?;
        std::fs::rename(&temp_path, &self.file_path).context("Failed to rename file")?;

        Ok(())
    }

    /// Synchronous lookup by worker_id (used internally).
    fn discover_by_worker_id_sync(
        &self,
        worker_id: WorkerId,
    ) -> Result<PeerInfo, DiscoveryQueryError> {
        // Reload from disk to get latest state
        if let Err(e) = self.load_from_disk() {
            return Err(DiscoveryQueryError::Backend(Arc::new(e)));
        }

        let state = self.inner.read();
        let by_worker_id = state.by_worker_id.get(&worker_id);
        if let Some(instance_id) = by_worker_id {
            let peer_info = state.by_instance_id.get(instance_id);
            if let Some(peer_info) = peer_info {
                return Ok(peer_info.clone());
            }
        }
        Err(DiscoveryQueryError::NotFound)
    }

    /// Synchronous lookup by instance_id (used internally).
    fn discover_by_instance_id_sync(
        &self,
        instance_id: InstanceId,
    ) -> Result<PeerInfo, DiscoveryQueryError> {
        // Reload from disk to get latest state
        if let Err(e) = self.load_from_disk() {
            return Err(DiscoveryQueryError::Backend(Arc::new(e)));
        }

        let state = self.inner.read();
        let by_instance_id = state.by_instance_id.get(&instance_id);
        if let Some(peer_info) = by_instance_id {
            return Ok(peer_info.clone());
        }
        Err(DiscoveryQueryError::NotFound)
    }

    /// Synchronous registration (used internally).
    fn register_instance_sync(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<(), DiscoveryError> {
        // Reload from disk to get latest state
        self.load_from_disk()
            .map_err(|e| DiscoveryError::Backend(e))?;

        let mut state = self.inner.write();

        // Validate no worker_id collision
        let worker_id = instance_id.worker_id();
        if let Some(existing_instance) = state.by_worker_id.get(&worker_id) {
            if *existing_instance != instance_id {
                return Err(DiscoveryError::WorkerIdCollision(
                    worker_id,
                    *existing_instance,
                    instance_id,
                ));
            }
        }

        // Check for duplicate registration
        if let Some(existing_peer_info) = state.by_instance_id.get(&instance_id) {
            if existing_peer_info.address_checksum() == worker_address.checksum() {
                return Err(DiscoveryError::InstanceAlreadyRegistered(instance_id));
            } else {
                return Err(DiscoveryError::ChecksumMismatch(
                    instance_id,
                    existing_peer_info.address_checksum(),
                    worker_address.checksum(),
                ));
            }
        }

        // Register peer
        let peer_info = PeerInfo::new(instance_id, worker_address);
        state.by_worker_id.insert(worker_id, instance_id);
        state.by_instance_id.insert(instance_id, peer_info);

        drop(state);

        // Save to disk
        self.save_to_disk()
            .map_err(|e| DiscoveryError::Backend(e))?;

        Ok(())
    }

    /// Synchronous unregistration (used internally).
    fn unregister_instance_sync(&self, instance_id: InstanceId) -> Result<(), DiscoveryError> {
        // Reload from disk to get latest state
        self.load_from_disk()
            .map_err(|e| DiscoveryError::Backend(e))?;

        let mut state = self.inner.write();
        state.by_worker_id.remove(&instance_id.worker_id());
        state.by_instance_id.remove(&instance_id);

        drop(state);

        // Save to disk
        self.save_to_disk()
            .map_err(|e| DiscoveryError::Backend(e))?;

        Ok(())
    }
}

impl PeerDiscovery for FilesystemPeerDiscovery {
    fn discover_by_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let this = self.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || this.discover_by_worker_id_sync(worker_id))
                .await
                .map_err(|e| DiscoveryQueryError::Backend(Arc::new(e.into())))?
        })
    }

    fn discover_by_instance_id(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let this = self.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || this.discover_by_instance_id_sync(instance_id))
                .await
                .map_err(|e| DiscoveryQueryError::Backend(Arc::new(e.into())))?
        })
    }

    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let this = self.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                this.register_instance_sync(instance_id, worker_address)
            })
            .await
            .map_err(|e| DiscoveryError::Backend(e.into()))?
        })
    }

    fn unregister_instance(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let this = self.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || this.unregister_instance_sync(instance_id))
                .await
                .map_err(|e| DiscoveryError::Backend(e.into()))?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn create_test_peer_info() -> (InstanceId, WorkerAddress) {
        let instance_id = InstanceId::new_v4();
        let mut builder = WorkerAddress::builder();
        builder
            .add_entry("endpoint", Bytes::from_static(b"test-address"))
            .unwrap();
        let worker_address = builder.build().unwrap();
        (instance_id, worker_address)
    }

    #[tokio::test]
    async fn test_filesystem_register_and_discover() {
        let discovery = FilesystemPeerDiscovery::new_temp().unwrap();
        let (instance_id, worker_address) = create_test_peer_info();
        let worker_id = instance_id.worker_id();

        // Register
        discovery
            .register_instance(instance_id, worker_address.clone())
            .await
            .unwrap();

        // Discover by worker_id
        let peer_info = discovery.discover_by_worker_id(worker_id).await.unwrap();
        assert_eq!(peer_info.instance_id(), instance_id);
        assert_eq!(peer_info.worker_id(), worker_id);

        // Discover by instance_id
        let peer_info = discovery
            .discover_by_instance_id(instance_id)
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id);
    }

    #[tokio::test]
    async fn test_filesystem_persistence() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test-discovery-{}.json", uuid::Uuid::new_v4()));

        let (instance_id, worker_address) = create_test_peer_info();
        let worker_id = instance_id.worker_id();

        // Create and register
        {
            let discovery = FilesystemPeerDiscovery::new(&file_path).unwrap();
            discovery
                .register_instance(instance_id, worker_address.clone())
                .await
                .unwrap();
        }

        // Drop and recreate - should load from disk
        {
            let discovery = FilesystemPeerDiscovery::new(&file_path).unwrap();
            let peer_info = discovery.discover_by_worker_id(worker_id).await.unwrap();
            assert_eq!(peer_info.instance_id(), instance_id);
        }

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
    }

    #[tokio::test]
    async fn test_filesystem_unregister() {
        let discovery = FilesystemPeerDiscovery::new_temp().unwrap();
        let (instance_id, worker_address) = create_test_peer_info();
        let worker_id = instance_id.worker_id();

        // Register
        discovery
            .register_instance(instance_id, worker_address)
            .await
            .unwrap();

        // Verify registered
        assert!(discovery.discover_by_worker_id(worker_id).await.is_ok());

        // Unregister
        discovery.unregister_instance(instance_id).await.unwrap();

        // Verify unregistered
        assert!(matches!(
            discovery.discover_by_worker_id(worker_id).await,
            Err(DiscoveryQueryError::NotFound)
        ));
    }

    #[tokio::test]
    async fn test_filesystem_duplicate_registration() {
        let discovery = FilesystemPeerDiscovery::new_temp().unwrap();

        let instance_id = InstanceId::new_v4();

        // Create two different addresses
        let mut builder1 = WorkerAddress::builder();
        builder1
            .add_entry("endpoint", Bytes::from_static(b"address1"))
            .unwrap();
        let address1 = builder1.build().unwrap();

        let mut builder2 = WorkerAddress::builder();
        builder2
            .add_entry("endpoint", Bytes::from_static(b"address2"))
            .unwrap();
        let address2 = builder2.build().unwrap();

        // Register first time - should succeed
        discovery
            .register_instance(instance_id, address1.clone())
            .await
            .unwrap();

        // Try to register same instance with same address - should fail with InstanceAlreadyRegistered
        let result = discovery.register_instance(instance_id, address1).await;
        assert!(matches!(
            result,
            Err(DiscoveryError::InstanceAlreadyRegistered(_))
        ));

        // Try to register same instance with different address - should fail with ChecksumMismatch
        let result = discovery.register_instance(instance_id, address2).await;
        assert!(matches!(
            result,
            Err(DiscoveryError::ChecksumMismatch(_, _, _))
        ));
    }
}
