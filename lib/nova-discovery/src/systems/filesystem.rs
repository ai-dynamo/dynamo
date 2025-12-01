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

#[cfg(feature = "filesystem")]
use fs4::fs_std::FileExt;
#[cfg(feature = "filesystem")]
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};

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
        let instance_id = InstanceId::from(uuid);
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
#[derive(Clone)]
pub struct FilesystemPeerDiscovery {
    file_path: PathBuf,
    inner: Arc<RwLock<FilesystemPeerDiscoveryInner>>,
    #[cfg(feature = "filesystem")]
    _watcher: Arc<Option<RecommendedWatcher>>,
}

impl std::fmt::Debug for FilesystemPeerDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilesystemPeerDiscovery")
            .field("file_path", &self.file_path)
            .finish()
    }
}

#[derive(Debug)]
struct FilesystemPeerDiscoveryInner {
    by_worker_id: HashMap<WorkerId, InstanceId>,
    by_instance_id: HashMap<InstanceId, PeerInfo>,
    cache_valid: bool,
}

impl FilesystemPeerDiscoveryInner {
    fn new() -> Self {
        Self {
            by_worker_id: HashMap::new(),
            by_instance_id: HashMap::new(),
            cache_valid: false,
        }
    }
}

impl FilesystemPeerDiscovery {
    /// Create a new filesystem-based peer discovery at the specified path.
    ///
    /// If the file exists, it will be loaded. If it doesn't exist, it will be created.
    #[cfg(feature = "filesystem")]
    pub fn new(file_path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = file_path.into();
        let inner = Arc::new(RwLock::new(FilesystemPeerDiscoveryInner::new()));

        // Setup file watcher for cache invalidation
        let inner_clone = Arc::clone(&inner);
        let watch_target = file_path.clone();

        let watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                // Check if the event is for our specific file
                let is_our_file = event.paths.iter().any(|p| p == &watch_target);
                if is_our_file && (event.kind.is_modify() || event.kind.is_create()) {
                    // Invalidate cache on file modification
                    inner_clone.write().cache_valid = false;
                    tracing::debug!("Cache invalidated due to file modification: {:?}", event);
                }
            }
        })
        .context("Failed to create file watcher")?;

        let mut watcher = watcher;

        // Watch the parent directory so we can detect file creation
        let watch_path = if file_path.exists() {
            file_path.clone()
        } else if let Some(parent) = file_path.parent() {
            // Ensure parent directory exists
            if !parent.exists() {
                std::fs::create_dir_all(parent).ok();
            }
            parent.to_path_buf()
        } else {
            file_path.clone()
        };

        watcher
            .watch(&watch_path, RecursiveMode::NonRecursive)
            .context("Failed to watch discovery path")?;

        let discovery = Self {
            file_path: file_path.clone(),
            inner,
            _watcher: Arc::new(Some(watcher)),
        };

        Ok(discovery)
    }

    /// Create a new filesystem-based peer discovery at the specified path (without watcher).
    ///
    /// This is used when the filesystem feature is not enabled.
    #[cfg(not(feature = "filesystem"))]
    pub fn new(file_path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = file_path.into();
        let inner = Arc::new(RwLock::new(FilesystemPeerDiscoveryInner::new()));

        let discovery = Self { file_path, inner };

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

    /// Load peer registry from disk with file locking.
    #[cfg(feature = "filesystem")]
    async fn load_from_disk(&self) -> Result<()> {
        // If file doesn't exist, that's okay - it will be created on first save
        if !self.file_path.exists() {
            return Ok(());
        }

        // Use spawn_blocking for file locking since fs4 locks are blocking
        let file_path = self.file_path.clone();
        let content = tokio::task::spawn_blocking(move || -> Result<String> {
            let file = std::fs::File::open(&file_path).context("Failed to open discovery file")?;

            file.lock_shared()
                .context("Failed to acquire shared lock")?;

            let content =
                std::fs::read_to_string(&file_path).context("Failed to read discovery file")?;

            // Lock is automatically released when file is dropped
            Ok(content)
        })
        .await
        .context("spawn_blocking failed")??;

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

        // Mark cache as valid after successful load
        inner.cache_valid = true;

        Ok(())
    }

    /// Load peer registry from disk (fallback without locking).
    #[cfg(not(feature = "filesystem"))]
    async fn load_from_disk(&self) -> Result<()> {
        // If file doesn't exist, that's okay - it will be created on first save
        if !self.file_path.exists() {
            return Ok(());
        }

        let content = tokio::fs::read_to_string(&self.file_path)
            .await
            .context("Failed to read discovery file")?;

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

        inner.cache_valid = true;

        Ok(())
    }

    /// Save peer registry to disk with file locking.
    #[cfg(feature = "filesystem")]
    async fn save_to_disk(&self) -> Result<()> {
        let content = {
            let inner = self.inner.read();

            let peers: Vec<SerializedPeerInfo> = inner
                .by_instance_id
                .values()
                .map(SerializedPeerInfo::from_peer_info)
                .collect();

            let registry = PeerRegistry { peers };

            serde_json::to_string_pretty(&registry).context("Failed to serialize registry")?
        }; // Lock is dropped here

        // Use spawn_blocking for file locking since fs4 locks are blocking
        let file_path = self.file_path.clone();

        // Use a unique temp file name to avoid collisions in concurrent saves
        let temp_file_name = format!(
            "{}.tmp.{}",
            self.file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            uuid::Uuid::new_v4()
        );
        let temp_path = self
            .file_path
            .parent()
            .map(|p| p.join(&temp_file_name))
            .unwrap_or_else(|| PathBuf::from(&temp_file_name));

        tokio::task::spawn_blocking(move || -> Result<()> {
            // Ensure parent directory exists
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }

            // Write to temp file first
            std::fs::write(&temp_path, &content).context("Failed to write temp file")?;

            // Open/create the target file and acquire exclusive lock
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&file_path)
                .context("Failed to open discovery file")?;

            file.lock_exclusive()
                .context("Failed to acquire exclusive lock")?;

            // Rename temp file to actual file (atomic operation)
            std::fs::rename(&temp_path, &file_path).context("Failed to rename file")?;

            // Lock is automatically released when file is dropped
            Ok(())
        })
        .await
        .context("spawn_blocking failed")??;

        Ok(())
    }

    /// Save peer registry to disk (fallback without locking).
    #[cfg(not(feature = "filesystem"))]
    async fn save_to_disk(&self) -> Result<()> {
        let content = {
            let inner = self.inner.read();

            let peers: Vec<SerializedPeerInfo> = inner
                .by_instance_id
                .values()
                .map(SerializedPeerInfo::from_peer_info)
                .collect();

            let registry = PeerRegistry { peers };

            serde_json::to_string_pretty(&registry).context("Failed to serialize registry")?
        }; // Lock is dropped here

        // Write to temp file first, then rename for atomicity
        let temp_path = self.file_path.with_extension("tmp");
        tokio::fs::write(&temp_path, content)
            .await
            .context("Failed to write temp file")?;
        tokio::fs::rename(&temp_path, &self.file_path)
            .await
            .context("Failed to rename file")?;

        Ok(())
    }

    /// Async lookup by worker_id (used internally).
    async fn discover_by_worker_id_async(
        &self,
        worker_id: WorkerId,
    ) -> Result<PeerInfo, DiscoveryQueryError> {
        // Lazy reload: only reload if cache is invalid
        let cache_valid = self.inner.read().cache_valid;
        if !cache_valid && let Err(e) = self.load_from_disk().await {
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

    /// Async lookup by instance_id (used internally).
    async fn discover_by_instance_id_async(
        &self,
        instance_id: InstanceId,
    ) -> Result<PeerInfo, DiscoveryQueryError> {
        // Lazy reload: only reload if cache is invalid
        let cache_valid = self.inner.read().cache_valid;
        if !cache_valid && let Err(e) = self.load_from_disk().await {
            return Err(DiscoveryQueryError::Backend(Arc::new(e)));
        }

        let state = self.inner.read();
        let by_instance_id = state.by_instance_id.get(&instance_id);
        if let Some(peer_info) = by_instance_id {
            return Ok(peer_info.clone());
        }
        Err(DiscoveryQueryError::NotFound)
    }

    /// Async registration (used internally).
    async fn register_instance_async(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<(), DiscoveryError> {
        // Always reload from disk before registration to ensure consistency
        self.load_from_disk()
            .await
            .map_err(DiscoveryError::Backend)?;

        // Scope the write lock to ensure it's dropped before await
        {
            let mut state = self.inner.write();

            // Validate no worker_id collision
            let worker_id = instance_id.worker_id();
            if let Some(existing_instance) = state.by_worker_id.get(&worker_id)
                && *existing_instance != instance_id
            {
                return Err(DiscoveryError::WorkerIdCollision(
                    worker_id,
                    *existing_instance,
                    instance_id,
                ));
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
        } // Lock is dropped here

        // Save to disk
        self.save_to_disk().await.map_err(DiscoveryError::Backend)?;

        Ok(())
    }

    /// Async unregistration (used internally).
    async fn unregister_instance_async(
        &self,
        instance_id: InstanceId,
    ) -> Result<(), DiscoveryError> {
        // Always reload from disk before unregistration to ensure consistency
        self.load_from_disk()
            .await
            .map_err(DiscoveryError::Backend)?;

        // Scope the write lock to ensure it's dropped before await
        {
            let mut state = self.inner.write();
            state.by_worker_id.remove(&instance_id.worker_id());
            state.by_instance_id.remove(&instance_id);
        } // Lock is dropped here

        // Save to disk
        self.save_to_disk().await.map_err(DiscoveryError::Backend)?;

        Ok(())
    }
}

impl PeerDiscovery for FilesystemPeerDiscovery {
    fn discover_by_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let this = self.clone();
        Box::pin(async move { this.discover_by_worker_id_async(worker_id).await })
    }

    fn discover_by_instance_id(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let this = self.clone();
        Box::pin(async move { this.discover_by_instance_id_async(instance_id).await })
    }

    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let this = self.clone();
        Box::pin(async move {
            this.register_instance_async(instance_id, worker_address)
                .await
        })
    }

    fn unregister_instance(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let this = self.clone();
        Box::pin(async move { this.unregister_instance_async(instance_id).await })
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

    #[tokio::test]
    async fn test_filesystem_concurrent_access_single_instance() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test-discovery-{}.json", uuid::Uuid::new_v4()));

        // Create a single discovery instance
        let discovery = FilesystemPeerDiscovery::new(&file_path).unwrap();

        // Register multiple peers concurrently from the same instance
        let mut handles = vec![];
        for _ in 0..5 {
            let disc = discovery.clone();
            let handle = tokio::spawn(async move {
                let (instance_id, worker_address) = create_test_peer_info();
                disc.register_instance(instance_id, worker_address.clone())
                    .await
                    .unwrap();
                (instance_id, worker_address)
            });
            handles.push(handle);
        }

        // Wait for all registrations to complete
        let mut peers = vec![];
        for handle in handles {
            let peer = handle.await.unwrap();
            peers.push(peer);
        }

        // Verify all peers are discoverable
        for (instance_id, _) in &peers {
            let worker_id = instance_id.worker_id();
            let peer_info = discovery.discover_by_worker_id(worker_id).await.unwrap();
            assert_eq!(peer_info.instance_id(), *instance_id);
        }

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
    }

    #[tokio::test]
    async fn test_filesystem_sequential_multi_instance() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test-discovery-{}.json", uuid::Uuid::new_v4()));

        // Create first discovery instance and register a peer
        let discovery1 = FilesystemPeerDiscovery::new(&file_path).unwrap();
        let (instance_id1, worker_address1) = create_test_peer_info();
        discovery1
            .register_instance(instance_id1, worker_address1.clone())
            .await
            .unwrap();

        // Create second discovery instance - it should see the first peer
        let discovery2 = FilesystemPeerDiscovery::new(&file_path).unwrap();
        let peer_info = discovery2
            .discover_by_worker_id(instance_id1.worker_id())
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id1);

        // Register another peer from second instance
        let (instance_id2, worker_address2) = create_test_peer_info();
        discovery2
            .register_instance(instance_id2, worker_address2.clone())
            .await
            .unwrap();

        // First instance should see the second peer after reload
        // Give watcher time to invalidate cache
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let peer_info = discovery1
            .discover_by_worker_id(instance_id2.worker_id())
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id2);

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
    }

    #[cfg(feature = "filesystem")]
    #[tokio::test]
    async fn test_filesystem_cache_invalidation() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test-discovery-{}.json", uuid::Uuid::new_v4()));

        // Create first discovery instance and register a peer
        let discovery1 = FilesystemPeerDiscovery::new(&file_path).unwrap();
        let (instance_id1, worker_address1) = create_test_peer_info();
        discovery1
            .register_instance(instance_id1, worker_address1.clone())
            .await
            .unwrap();

        // Create second discovery instance
        let discovery2 = FilesystemPeerDiscovery::new(&file_path).unwrap();

        // Give watcher time to setup
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Verify discovery2 can see the peer (after cache invalidation)
        let peer_info = discovery2
            .discover_by_worker_id(instance_id1.worker_id())
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id1);

        // Register another peer through discovery1
        let (instance_id2, worker_address2) = create_test_peer_info();
        discovery1
            .register_instance(instance_id2, worker_address2.clone())
            .await
            .unwrap();

        // Give watcher time to detect the change
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify discovery2's cache was invalidated and it can now see the new peer
        let peer_info2 = discovery2
            .discover_by_worker_id(instance_id2.worker_id())
            .await
            .unwrap();
        assert_eq!(peer_info2.instance_id(), instance_id2);

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
    }

    #[cfg(feature = "filesystem")]
    #[tokio::test]
    async fn test_filesystem_external_modification() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test-discovery-{}.json", uuid::Uuid::new_v4()));

        // Create discovery instance and register a peer
        let discovery = FilesystemPeerDiscovery::new(&file_path).unwrap();
        let (instance_id1, worker_address1) = create_test_peer_info();
        discovery
            .register_instance(instance_id1, worker_address1.clone())
            .await
            .unwrap();

        // Verify peer is discoverable
        assert!(
            discovery
                .discover_by_worker_id(instance_id1.worker_id())
                .await
                .is_ok()
        );

        // Simulate external modification - manually add another peer to the file
        let (instance_id2, worker_address2) = create_test_peer_info();
        let serialized2 = SerializedPeerInfo {
            instance_id: instance_id2.to_string(),
            worker_id: instance_id2.worker_id().as_u64(),
            worker_address: worker_address2.clone(),
            address_checksum: worker_address2.checksum(),
        };

        // Read current file
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        let mut registry: PeerRegistry = serde_json::from_str(&content).unwrap();
        registry.peers.push(serialized2);

        // Write modified registry back
        let new_content = serde_json::to_string_pretty(&registry).unwrap();
        tokio::fs::write(&file_path, new_content).await.unwrap();

        // Give watcher time to detect the change
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify the discovery instance picks up the externally added peer
        let peer_info2 = discovery
            .discover_by_worker_id(instance_id2.worker_id())
            .await
            .unwrap();
        assert_eq!(peer_info2.instance_id(), instance_id2);

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
    }
}
