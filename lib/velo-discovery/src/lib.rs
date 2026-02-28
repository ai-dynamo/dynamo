// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Filesystem-based peer discovery for Velo distributed systems.
//!
//! This crate provides [`FilesystemPeerDiscovery`], a [`velo_messenger::PeerDiscovery`]
//! backend that stores peer information in a JSON file on disk. It is suitable for
//! development, testing, and lightweight single-host deployments.
//!
//! # File Format
//!
//! The file contains a JSON object with a `peers` array:
//! ```json
//! {
//!   "peers": [
//!     {
//!       "instance_id": "uuid-string",
//!       "worker_id": 123,
//!       "worker_address": "<msgpack bytes>",
//!       "address_checksum": 12345678
//!     }
//!   ]
//! }
//! ```
//!
//! # Concurrency
//!
//! Uses file-based locking (via `fs4`) for safe concurrent access across processes.
//! Within a process, uses `RwLock` + `AsyncMutex` for thread safety.

use anyhow::{Context, Result};
use fs4::fs_std::FileExt;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use velo_common::{InstanceId, PeerInfo, WorkerAddress, WorkerId};

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
            address_checksum: peer_info.worker_address().checksum(),
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
/// Cache invalidation is done on-demand: the cache is marked invalid when writes
/// occur, and reloaded on the next query. For multi-process use, the cache is
/// always reloaded before writes to ensure consistency.
#[derive(Clone)]
pub struct FilesystemPeerDiscovery {
    file_path: PathBuf,
    inner: Arc<RwLock<FilesystemPeerDiscoveryInner>>,
    /// Mutex to serialize write operations (register/unregister) to prevent race conditions.
    write_mutex: Arc<AsyncMutex<()>>,
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
    /// If the file exists, it will be loaded lazily on first query. If it doesn't
    /// exist, it will be created on first registration.
    pub fn new(file_path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = file_path.into();
        let inner = Arc::new(RwLock::new(FilesystemPeerDiscoveryInner::new()));
        Ok(Self {
            file_path,
            inner,
            write_mutex: Arc::new(AsyncMutex::new(())),
        })
    }

    /// Create a new filesystem-based peer discovery in a temporary directory.
    ///
    /// Useful for testing.
    pub fn new_temp() -> Result<Self> {
        let temp_dir = std::env::temp_dir();
        let file_name = format!("velo-discovery-{}.json", uuid::Uuid::new_v4());
        let file_path = temp_dir.join(file_name);
        Self::new(file_path)
    }

    /// Register peer info directly.
    pub fn register_peer_info(&self, peer_info: &PeerInfo) -> Result<()> {
        let instance_id = peer_info.instance_id();
        let worker_address = peer_info.worker_address().clone();
        self.register_instance(instance_id, worker_address)
    }

    /// Register an instance with an address (synchronous convenience wrapper).
    pub fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<()> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.block_on(self.register_instance_async(instance_id, worker_address))
            }
            Err(_) => {
                let rt =
                    tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
                rt.block_on(self.register_instance_async(instance_id, worker_address))
            }
        }
    }

    /// Unregister an instance (synchronous convenience wrapper).
    pub fn unregister_instance(&self, instance_id: InstanceId) -> Result<()> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.unregister_instance_async(instance_id)),
            Err(_) => {
                let rt =
                    tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
                rt.block_on(self.unregister_instance_async(instance_id))
            }
        }
    }

    /// Load peer registry from disk with file locking.
    async fn load_from_disk(&self) -> Result<()> {
        if !self.file_path.exists() {
            return Ok(());
        }

        let file_path = self.file_path.clone();
        let content = tokio::task::spawn_blocking(move || -> Result<String> {
            let file = std::fs::File::open(&file_path).context("Failed to open discovery file")?;
            file.lock_shared()
                .context("Failed to acquire shared lock")?;
            let content =
                std::fs::read_to_string(&file_path).context("Failed to read discovery file")?;
            // Lock is released when file is dropped
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

        inner.cache_valid = true;
        Ok(())
    }

    /// Save peer registry to disk with file locking and atomic rename.
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
        };

        let file_path = self.file_path.clone();
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
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }

            std::fs::write(&temp_path, &content).context("Failed to write temp file")?;

            let file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&file_path)
                .context("Failed to open discovery file")?;
            file.lock_exclusive()
                .context("Failed to acquire exclusive lock")?;

            std::fs::rename(&temp_path, &file_path).context("Failed to rename file")?;
            Ok(())
        })
        .await
        .context("spawn_blocking failed")??;

        Ok(())
    }

    async fn discover_by_worker_id_async(&self, worker_id: WorkerId) -> Result<PeerInfo> {
        let cache_valid = self.inner.read().cache_valid;
        if !cache_valid {
            self.load_from_disk().await?;
        }

        let state = self.inner.read();
        if let Some(instance_id) = state.by_worker_id.get(&worker_id)
            && let Some(peer_info) = state.by_instance_id.get(instance_id)
        {
            return Ok(peer_info.clone());
        }
        anyhow::bail!("Worker ID {} not found", worker_id)
    }

    async fn discover_by_instance_id_async(&self, instance_id: InstanceId) -> Result<PeerInfo> {
        let cache_valid = self.inner.read().cache_valid;
        if !cache_valid {
            self.load_from_disk().await?;
        }

        let state = self.inner.read();
        if let Some(peer_info) = state.by_instance_id.get(&instance_id) {
            return Ok(peer_info.clone());
        }
        anyhow::bail!("Instance ID {} not found", instance_id)
    }

    async fn register_instance_async(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<()> {
        let _guard = self.write_mutex.lock().await;

        self.load_from_disk().await?;

        {
            let mut state = self.inner.write();

            let worker_id = instance_id.worker_id();
            if let Some(existing_instance) = state.by_worker_id.get(&worker_id)
                && *existing_instance != instance_id
            {
                anyhow::bail!(
                    "Worker ID collision: worker_id={} is already claimed by instance {} (new: {})",
                    worker_id,
                    existing_instance,
                    instance_id
                );
            }

            if let Some(existing_peer_info) = state.by_instance_id.get(&instance_id) {
                if existing_peer_info.worker_address().checksum() == worker_address.checksum() {
                    anyhow::bail!("Instance {} is already registered", instance_id);
                } else {
                    anyhow::bail!(
                        "Instance {} already registered with a different address (checksum mismatch: {} vs {})",
                        instance_id,
                        existing_peer_info.worker_address().checksum(),
                        worker_address.checksum()
                    );
                }
            }

            let peer_info = PeerInfo::new(instance_id, worker_address);
            state.by_worker_id.insert(worker_id, instance_id);
            state.by_instance_id.insert(instance_id, peer_info);
        }

        self.save_to_disk().await?;
        Ok(())
    }

    async fn unregister_instance_async(&self, instance_id: InstanceId) -> Result<()> {
        let _guard = self.write_mutex.lock().await;

        self.load_from_disk().await?;

        {
            let mut state = self.inner.write();
            state.by_worker_id.remove(&instance_id.worker_id());
            state.by_instance_id.remove(&instance_id);
        }

        self.save_to_disk().await?;
        Ok(())
    }
}

impl velo_messenger::PeerDiscovery for FilesystemPeerDiscovery {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(self.discover_by_worker_id_async(worker_id))
    }

    fn discover_by_instance_id(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(self.discover_by_instance_id_async(instance_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_test_address(data: &[u8]) -> WorkerAddress {
        let map: HashMap<String, Vec<u8>> = [("endpoint".to_string(), data.to_vec())]
            .into_iter()
            .collect();
        let encoded = rmp_serde::to_vec(&map).unwrap();
        WorkerAddress::from_encoded(encoded)
    }

    fn create_test_peer_info() -> (InstanceId, WorkerAddress) {
        let instance_id = InstanceId::new_v4();
        let worker_address = make_test_address(b"test-address");
        (instance_id, worker_address)
    }

    #[tokio::test]
    async fn test_register_and_discover() {
        let discovery = FilesystemPeerDiscovery::new_temp().unwrap();
        let (instance_id, worker_address) = create_test_peer_info();
        let worker_id = instance_id.worker_id();

        discovery
            .register_instance_async(instance_id, worker_address.clone())
            .await
            .unwrap();

        let peer_info = discovery
            .discover_by_worker_id_async(worker_id)
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id);
        assert_eq!(peer_info.worker_id(), worker_id);

        let peer_info = discovery
            .discover_by_instance_id_async(instance_id)
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id);
    }

    #[tokio::test]
    async fn test_persistence() {
        let temp_dir = std::env::temp_dir();
        let file_path =
            temp_dir.join(format!("test-velo-discovery-{}.json", uuid::Uuid::new_v4()));

        let (instance_id, worker_address) = create_test_peer_info();
        let worker_id = instance_id.worker_id();

        {
            let discovery = FilesystemPeerDiscovery::new(&file_path).unwrap();
            discovery
                .register_instance_async(instance_id, worker_address.clone())
                .await
                .unwrap();
        }

        {
            let discovery = FilesystemPeerDiscovery::new(&file_path).unwrap();
            let peer_info = discovery
                .discover_by_worker_id_async(worker_id)
                .await
                .unwrap();
            assert_eq!(peer_info.instance_id(), instance_id);
        }

        let _ = std::fs::remove_file(&file_path);
    }

    #[tokio::test]
    async fn test_unregister() {
        let discovery = FilesystemPeerDiscovery::new_temp().unwrap();
        let (instance_id, worker_address) = create_test_peer_info();
        let worker_id = instance_id.worker_id();

        discovery
            .register_instance_async(instance_id, worker_address)
            .await
            .unwrap();

        assert!(
            discovery
                .discover_by_worker_id_async(worker_id)
                .await
                .is_ok()
        );

        discovery
            .unregister_instance_async(instance_id)
            .await
            .unwrap();

        assert!(
            discovery
                .discover_by_worker_id_async(worker_id)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_duplicate_registration() {
        let discovery = FilesystemPeerDiscovery::new_temp().unwrap();
        let instance_id = InstanceId::new_v4();
        let address1 = make_test_address(b"address1");
        let address2 = make_test_address(b"address2");

        discovery
            .register_instance_async(instance_id, address1.clone())
            .await
            .unwrap();

        let result = discovery
            .register_instance_async(instance_id, address1.clone())
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("already registered"),
            "Expected 'already registered', got: {msg}"
        );

        let result = discovery
            .register_instance_async(instance_id, address2.clone())
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("checksum mismatch") || msg.contains("different address"),
            "Expected checksum mismatch error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let temp_dir = std::env::temp_dir();
        let file_path =
            temp_dir.join(format!("test-velo-discovery-{}.json", uuid::Uuid::new_v4()));

        let discovery = FilesystemPeerDiscovery::new(&file_path).unwrap();

        let mut handles = vec![];
        for _ in 0..5 {
            let disc = discovery.clone();
            let handle = tokio::spawn(async move {
                let (instance_id, worker_address) = create_test_peer_info();
                disc.register_instance_async(instance_id, worker_address.clone())
                    .await
                    .unwrap();
                (instance_id, worker_address)
            });
            handles.push(handle);
        }

        let mut peers = vec![];
        for handle in handles {
            let peer = handle.await.unwrap();
            peers.push(peer);
        }

        for (instance_id, _) in &peers {
            let worker_id = instance_id.worker_id();
            let peer_info = discovery
                .discover_by_worker_id_async(worker_id)
                .await
                .unwrap();
            assert_eq!(peer_info.instance_id(), *instance_id);
        }

        let _ = std::fs::remove_file(&file_path);
    }

    #[tokio::test]
    async fn test_sequential_multi_instance() {
        let temp_dir = std::env::temp_dir();
        let file_path =
            temp_dir.join(format!("test-velo-discovery-{}.json", uuid::Uuid::new_v4()));

        let discovery1 = FilesystemPeerDiscovery::new(&file_path).unwrap();
        let (instance_id1, worker_address1) = create_test_peer_info();
        discovery1
            .register_instance_async(instance_id1, worker_address1.clone())
            .await
            .unwrap();

        // Second instance starts with cache_valid=false, so it will load from disk
        let discovery2 = FilesystemPeerDiscovery::new(&file_path).unwrap();
        let peer_info = discovery2
            .discover_by_worker_id_async(instance_id1.worker_id())
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id1);

        let (instance_id2, worker_address2) = create_test_peer_info();
        discovery2
            .register_instance_async(instance_id2, worker_address2.clone())
            .await
            .unwrap();

        // Invalidate discovery1's cache to simulate detecting external change
        discovery1.inner.write().cache_valid = false;
        let peer_info = discovery1
            .discover_by_worker_id_async(instance_id2.worker_id())
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id2);

        let _ = std::fs::remove_file(&file_path);
    }

    #[tokio::test]
    async fn test_peer_discovery_trait() {
        use velo_messenger::PeerDiscovery;

        let discovery = FilesystemPeerDiscovery::new_temp().unwrap();
        let (instance_id, worker_address) = create_test_peer_info();
        let worker_id = instance_id.worker_id();

        discovery
            .register_instance_async(instance_id, worker_address.clone())
            .await
            .unwrap();

        let peer_info = discovery.discover_by_worker_id(worker_id).await.unwrap();
        assert_eq!(peer_info.instance_id(), instance_id);

        let peer_info = discovery
            .discover_by_instance_id(instance_id)
            .await
            .unwrap();
        assert_eq!(peer_info.instance_id(), instance_id);
    }
}
