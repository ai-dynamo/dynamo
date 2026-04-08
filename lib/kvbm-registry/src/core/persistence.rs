// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Persistence layer for the registry hub.
//!
//! Provides snapshot and WAL-based persistence for crash recovery.
//!

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use async_trait::async_trait;
use bincode::config;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

/// Persisted entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEntry<K, V, M> {
    pub key: K,
    pub value: V,
    pub metadata: Option<M>,
}

/// Access statistics for eviction tracking.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessStats {
    /// Last access timestamp (milliseconds since epoch)
    pub last_access_ms: u64,
    /// Total access count
    pub access_count: u64,
}

/// Full registry state for snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrySnapshot<K, V, M> {
    /// Protocol version for forward/backward compatibility
    pub version: u32,
    /// Snapshot sequence number
    pub sequence: u64,
    /// Timestamp when snapshot was taken
    pub timestamp_ms: u64,
    /// All entries in the registry
    pub entries: Vec<PersistedEntry<K, V, M>>,
    /// Access statistics for each key (for LRU/LFU)
    pub access_stats: Vec<(K, AccessStats)>,
}

impl<K, V, M> Default for RegistrySnapshot<K, V, M> {
    fn default() -> Self {
        Self {
            version: 1,
            sequence: 0,
            timestamp_ms: 0,
            entries: Vec::new(),
            access_stats: Vec::new(),
        }
    }
}

/// WAL entry types for incremental persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry<K, V, M> {
    /// Register new entries
    Register {
        seq: u64,
        entries: Vec<PersistedEntry<K, V, M>>,
    },
    /// Remove entries by key
    Remove { seq: u64, keys: Vec<K> },
    /// Touch entries (access notification)
    Touch { seq: u64, keys: Vec<K> },
    /// Checkpoint marker
    Checkpoint { seq: u64, snapshot_seq: u64 },
}

/// Trait for persistence storage backends.
#[async_trait]
pub trait PersistenceBackend: Send + Sync {
    /// Write data to a path (atomic if possible)
    async fn write(&self, path: &str, data: &[u8]) -> Result<()>;

    /// Read data from a path
    async fn read(&self, path: &str) -> Result<Vec<u8>>;

    /// Check if a path exists
    async fn exists(&self, path: &str) -> Result<bool>;

    /// Append data to a file (for WAL)
    async fn append(&self, path: &str, data: &[u8]) -> Result<()>;

    /// List files with prefix
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Delete a file
    async fn delete(&self, path: &str) -> Result<()>;

    /// Atomic rename (for safe snapshot writes)
    async fn rename(&self, from: &str, to: &str) -> Result<()>;
}

/// Local disk persistence backend.
pub struct LocalDiskBackend {
    base_path: PathBuf,
}

impl LocalDiskBackend {
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;
        Ok(Self { base_path })
    }

    fn full_path(&self, path: &str) -> PathBuf {
        self.base_path.join(path)
    }
}

#[async_trait]
impl PersistenceBackend for LocalDiskBackend {
    async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        let full_path = self.full_path(path);

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Write to temp file first, then rename (atomic on most filesystems)
        let temp_path = full_path.with_extension("tmp");
        tokio::fs::write(&temp_path, data).await?;
        tokio::fs::rename(&temp_path, &full_path).await?;

        // Sync to ensure durability
        if let Ok(file) = tokio::fs::File::open(&full_path).await {
            let _ = file.sync_all().await;
        }

        Ok(())
    }

    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        let full_path = self.full_path(path);
        Ok(tokio::fs::read(&full_path).await?)
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let full_path = self.full_path(path);
        Ok(full_path.exists())
    }

    async fn append(&self, path: &str, data: &[u8]) -> Result<()> {
        let full_path = self.full_path(path);

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&full_path)
            .await?;

        file.write_all(data).await?;
        file.sync_all().await?;

        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let search_path = self.full_path(prefix);

        let mut results = Vec::new();

        // If prefix is a directory, list its contents
        if search_path.is_dir() {
            let mut entries = tokio::fs::read_dir(&search_path).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if let Ok(relative) = path.strip_prefix(&self.base_path) {
                    results.push(relative.to_string_lossy().to_string());
                }
            }
        } else {
            // Otherwise, list files in parent that start with the prefix
            let search_dir = search_path
                .parent()
                .unwrap_or(&self.base_path)
                .to_path_buf();

            if search_dir.exists() {
                let mut entries = tokio::fs::read_dir(&search_dir).await?;
                while let Some(entry) = entries.next_entry().await? {
                    let path = entry.path();
                    if let Ok(relative) = path.strip_prefix(&self.base_path) {
                        let path_str = relative.to_string_lossy().to_string();
                        if path_str.starts_with(prefix) {
                            results.push(path_str);
                        }
                    }
                }
            }
        }

        results.sort();
        Ok(results)
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let full_path = self.full_path(path);
        if full_path.exists() {
            tokio::fs::remove_file(&full_path).await?;
        }
        Ok(())
    }

    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        let from_path = self.full_path(from);
        let to_path = self.full_path(to);
        tokio::fs::rename(&from_path, &to_path).await?;
        Ok(())
    }
}

/// Snapshot-based persistence.
pub struct SnapshotPersistence<K, V, M> {
    backend: Box<dyn PersistenceBackend>,
    snapshot_prefix: String,
    ops_since_snapshot: AtomicU64,
    snapshot_interval: u64,
    retention_count: usize,
    _phantom: std::marker::PhantomData<(K, V, M)>,
}

impl<K, V, M> SnapshotPersistence<K, V, M>
where
    K: Serialize + for<'de> Deserialize<'de> + Send + Sync,
    V: Serialize + for<'de> Deserialize<'de> + Send + Sync,
    M: Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    pub fn new(
        backend: Box<dyn PersistenceBackend>,
        snapshot_prefix: String,
        snapshot_interval: u64,
        retention_count: usize,
    ) -> Self {
        Self {
            backend,
            snapshot_prefix,
            ops_since_snapshot: AtomicU64::new(0),
            snapshot_interval,
            retention_count,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Take a snapshot of the registry state.
    pub async fn take_snapshot(&self, snapshot: &RegistrySnapshot<K, V, M>) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let filename = format!(
            "{}/snapshot-{:08}-{}.bin",
            self.snapshot_prefix, snapshot.sequence, timestamp
        );

        let data = bincode::serde::encode_to_vec(snapshot, config::standard())?;
        self.backend.write(&filename, &data).await?;
        self.ops_since_snapshot.store(0, Ordering::Release);

        tracing::info!(
            seq = snapshot.sequence,
            entries = snapshot.entries.len(),
            "Snapshot saved: {}",
            filename
        );

        // Clean up old snapshots
        self.cleanup_old_snapshots().await?;

        Ok(())
    }

    /// Load the latest snapshot.
    pub async fn load_latest(&self) -> Result<Option<RegistrySnapshot<K, V, M>>> {
        let files = self.backend.list(&self.snapshot_prefix).await?;

        let snapshot_files: Vec<_> = files
            .iter()
            .filter(|f| f.contains("snapshot-") && f.ends_with(".bin"))
            .collect();

        if snapshot_files.is_empty() {
            return Ok(None);
        }

        // Get the latest snapshot (highest sequence number)
        let latest = snapshot_files.iter().max().unwrap();
        let data = self.backend.read(latest).await?;
        let (snapshot, _): (RegistrySnapshot<K, V, M>, _) =
            bincode::serde::decode_from_slice(&data, config::standard())?;

        tracing::info!(
            seq = snapshot.sequence,
            entries = snapshot.entries.len(),
            "Loaded snapshot: {}",
            latest
        );

        Ok(Some(snapshot))
    }

    /// Record an operation (for snapshot interval tracking).
    pub fn record_operation(&self) {
        self.ops_since_snapshot.fetch_add(1, Ordering::Relaxed);
    }

    /// Check if we should take a snapshot.
    pub fn should_snapshot(&self) -> bool {
        self.ops_since_snapshot.load(Ordering::Relaxed) >= self.snapshot_interval
    }

    /// Clean up old snapshots, keeping only the most recent ones.
    async fn cleanup_old_snapshots(&self) -> Result<()> {
        let files = self.backend.list(&self.snapshot_prefix).await?;

        let mut snapshot_files: Vec<_> = files
            .iter()
            .filter(|f| f.contains("snapshot-") && f.ends_with(".bin"))
            .cloned()
            .collect();

        if snapshot_files.len() <= self.retention_count {
            return Ok(());
        }

        snapshot_files.sort();
        let to_delete = snapshot_files.len() - self.retention_count;

        for file in snapshot_files.iter().take(to_delete) {
            tracing::debug!("Deleting old snapshot: {}", file);
            self.backend.delete(file).await?;
        }

        Ok(())
    }
}

/// Write-Ahead Log persistence.
pub struct WalPersistence<K, V, M> {
    backend: Box<dyn PersistenceBackend>,
    wal_path: String,
    current_seq: AtomicU64,
    _phantom: std::marker::PhantomData<(K, V, M)>,
}

impl<K, V, M> WalPersistence<K, V, M>
where
    K: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync,
    V: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync,
    M: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync,
{
    pub fn new(backend: Box<dyn PersistenceBackend>, wal_path: String) -> Self {
        Self {
            backend,
            wal_path,
            current_seq: AtomicU64::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Log a WAL entry.
    pub async fn log(&self, entry: WalEntry<K, V, M>) -> Result<u64> {
        let seq = self.current_seq.fetch_add(1, Ordering::SeqCst);

        // Serialize with length prefix for safe recovery
        let data = bincode::serde::encode_to_vec(&entry, config::standard())?;
        let mut buf = Vec::with_capacity(4 + data.len());
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&data);

        self.backend.append(&self.wal_path, &buf).await?;

        Ok(seq)
    }

    /// Replay WAL entries from disk.
    pub async fn replay(&self) -> Result<Vec<WalEntry<K, V, M>>> {
        if !self.backend.exists(&self.wal_path).await? {
            return Ok(Vec::new());
        }

        let data = self.backend.read(&self.wal_path).await?;
        let mut entries = Vec::new();
        let mut pos = 0;

        while pos + 4 <= data.len() {
            let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap_or([0; 4])) as usize;
            pos += 4;

            if pos + len > data.len() {
                tracing::warn!("WAL truncated at position {} (expected {} bytes)", pos, len);
                break;
            }

            match bincode::serde::decode_from_slice::<WalEntry<K, V, M>, _>(
                &data[pos..pos + len],
                config::standard(),
            ) {
                Ok((entry, _)) => entries.push(entry),
                Err(e) => {
                    tracing::warn!("Failed to deserialize WAL entry at position {}: {}", pos, e);
                    break;
                }
            }
            pos += len;
        }

        // Update sequence counter
        if let Some(last_seq) = entries
            .iter()
            .map(|e| match e {
                WalEntry::Register { seq, .. } => *seq,
                WalEntry::Remove { seq, .. } => *seq,
                WalEntry::Touch { seq, .. } => *seq,
                WalEntry::Checkpoint { seq, .. } => *seq,
            })
            .max()
        {
            self.current_seq.store(last_seq + 1, Ordering::Release);
        }

        tracing::info!(entries = entries.len(), "Replayed WAL entries");
        Ok(entries)
    }

    /// Truncate WAL after checkpoint.
    pub async fn truncate(&self) -> Result<()> {
        self.backend.delete(&self.wal_path).await?;
        tracing::debug!("WAL truncated");
        Ok(())
    }

    /// Get current sequence number.
    pub fn current_seq(&self) -> u64 {
        self.current_seq.load(Ordering::Relaxed)
    }

    /// Set sequence number (used during recovery).
    pub fn set_seq(&self, seq: u64) {
        self.current_seq.store(seq, Ordering::Release);
    }
}

/// Hybrid persistence combining snapshots and WAL.
pub struct HybridPersistence<K, V, M>
where
    K: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync,
    V: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync,
    M: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync,
{
    snapshot: SnapshotPersistence<K, V, M>,
    wal: WalPersistence<K, V, M>,
}

impl<K, V, M> HybridPersistence<K, V, M>
where
    K: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static,
    V: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static,
    M: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static,
{
    /// Create new hybrid persistence.
    ///
    /// # Arguments
    /// * `backend` - Storage backend (must be cloneable or create two instances)
    /// * `base_prefix` - Base path/prefix for all persistence files
    /// * `snapshot_interval` - Number of operations before taking a snapshot
    /// * `retention_count` - Number of snapshots to retain
    pub fn new(
        snapshot_backend: Box<dyn PersistenceBackend>,
        wal_backend: Box<dyn PersistenceBackend>,
        base_prefix: &str,
        snapshot_interval: u64,
        retention_count: usize,
    ) -> Self {
        let snapshot = SnapshotPersistence::new(
            snapshot_backend,
            format!("{}/snapshots", base_prefix),
            snapshot_interval,
            retention_count,
        );

        let wal = WalPersistence::new(wal_backend, format!("{}/wal.log", base_prefix));

        Self { snapshot, wal }
    }

    /// Log an operation to WAL and maybe trigger snapshot.
    pub async fn log_operation(&self, entry: WalEntry<K, V, M>) -> Result<u64> {
        let seq = self.wal.log(entry).await?;
        self.snapshot.record_operation();
        Ok(seq)
    }

    /// Take a checkpoint (snapshot + truncate WAL).
    pub async fn checkpoint(&self, snapshot: &RegistrySnapshot<K, V, M>) -> Result<()> {
        // Take snapshot
        self.snapshot.take_snapshot(snapshot).await?;

        // Log checkpoint to WAL
        self.wal
            .log(WalEntry::Checkpoint {
                seq: self.wal.current_seq(),
                snapshot_seq: snapshot.sequence,
            })
            .await?;

        // Truncate WAL
        self.wal.truncate().await?;

        Ok(())
    }

    /// Recover state from persistence.
    pub async fn recover(&self) -> Result<Option<RegistrySnapshot<K, V, M>>> {
        // Load latest snapshot
        let snapshot = self.snapshot.load_latest().await?.unwrap_or_default();

        // Replay WAL entries after snapshot
        let wal_entries = self.wal.replay().await?;

        // Update sequence
        self.wal.set_seq(snapshot.sequence + 1);

        // Apply WAL entries to snapshot (caller should do this based on entry types)
        if !wal_entries.is_empty() {
            tracing::info!(
                snapshot_seq = snapshot.sequence,
                wal_entries = wal_entries.len(),
                "Recovery: loaded snapshot + {} WAL entries",
                wal_entries.len()
            );
        }

        if snapshot.entries.is_empty() && wal_entries.is_empty() {
            return Ok(None);
        }

        Ok(Some(snapshot))
    }

    /// Check if a snapshot should be taken.
    pub fn should_snapshot(&self) -> bool {
        self.snapshot.should_snapshot()
    }

    /// Get reference to snapshot persistence.
    pub fn snapshot(&self) -> &SnapshotPersistence<K, V, M> {
        &self.snapshot
    }

    /// Get reference to WAL persistence.
    pub fn wal(&self) -> &WalPersistence<K, V, M> {
        &self.wal
    }
}

/// Configuration for persistence.
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Enable persistence
    pub enabled: bool,
    /// Base path/prefix for persistence files
    pub base_path: String,
    /// Snapshot interval (number of operations)
    pub snapshot_interval: u64,
    /// Number of snapshots to retain
    pub snapshot_retention: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_path: "/tmp/registry".to_string(),
            snapshot_interval: 10000,
            snapshot_retention: 3,
        }
    }
}

impl PersistenceConfig {
    /// Create from environment variables.
    ///
    /// Environment variables:
    /// - `DYN_REGISTRY_PERSISTENCE_ENABLED`: "1" or "true" to enable
    /// - `DYN_REGISTRY_PERSISTENCE_PATH`: Base path for persistence files
    /// - `DYN_REGISTRY_PERSISTENCE_SNAPSHOT_INTERVAL`: Operations between snapshots
    /// - `DYN_REGISTRY_PERSISTENCE_SNAPSHOT_RETENTION`: Number of snapshots to keep
    pub fn from_env() -> Self {
        let enabled = std::env::var("DYN_REGISTRY_PERSISTENCE_ENABLED")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let base_path = std::env::var("DYN_REGISTRY_PERSISTENCE_PATH")
            .unwrap_or_else(|_| "/tmp/registry".to_string());

        let snapshot_interval = std::env::var("DYN_REGISTRY_PERSISTENCE_SNAPSHOT_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000);

        let snapshot_retention = std::env::var("DYN_REGISTRY_PERSISTENCE_SNAPSHOT_RETENTION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        Self {
            enabled,
            base_path,
            snapshot_interval,
            snapshot_retention,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_disk_backend() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LocalDiskBackend::new(temp_dir.path()).unwrap();

        // Test write and read
        backend.write("test.bin", b"hello world").await.unwrap();
        let data = backend.read("test.bin").await.unwrap();
        assert_eq!(data, b"hello world");

        // Test exists
        assert!(backend.exists("test.bin").await.unwrap());
        assert!(!backend.exists("nonexistent.bin").await.unwrap());

        // Test append
        backend.append("append.log", b"line1\n").await.unwrap();
        backend.append("append.log", b"line2\n").await.unwrap();
        let log = backend.read("append.log").await.unwrap();
        assert_eq!(log, b"line1\nline2\n");

        // Test list
        backend.write("subdir/file1.bin", b"1").await.unwrap();
        backend.write("subdir/file2.bin", b"2").await.unwrap();
        let files = backend.list("subdir").await.unwrap();
        assert_eq!(files.len(), 2);

        // Test delete
        backend.delete("test.bin").await.unwrap();
        assert!(!backend.exists("test.bin").await.unwrap());
    }

    #[tokio::test]
    async fn test_snapshot_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = Box::new(LocalDiskBackend::new(temp_dir.path()).unwrap());

        let persistence: SnapshotPersistence<u64, String, ()> =
            SnapshotPersistence::new(backend, "snapshots".to_string(), 100, 2);

        // Create and save snapshot
        let snapshot = RegistrySnapshot {
            version: 1,
            sequence: 1,
            timestamp_ms: 12345,
            entries: vec![
                PersistedEntry {
                    key: 1,
                    value: "one".to_string(),
                    metadata: None,
                },
                PersistedEntry {
                    key: 2,
                    value: "two".to_string(),
                    metadata: None,
                },
            ],
            access_stats: vec![],
        };

        persistence.take_snapshot(&snapshot).await.unwrap();

        // Load snapshot
        let loaded = persistence.load_latest().await.unwrap().unwrap();
        assert_eq!(loaded.sequence, 1);
        assert_eq!(loaded.entries.len(), 2);
    }

    #[tokio::test]
    async fn test_wal_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = Box::new(LocalDiskBackend::new(temp_dir.path()).unwrap());

        let wal: WalPersistence<u64, String, ()> =
            WalPersistence::new(backend, "wal.log".to_string());

        // Log entries
        wal.log(WalEntry::Register {
            seq: 1,
            entries: vec![PersistedEntry {
                key: 1,
                value: "one".to_string(),
                metadata: None,
            }],
        })
        .await
        .unwrap();

        wal.log(WalEntry::Remove {
            seq: 2,
            keys: vec![1],
        })
        .await
        .unwrap();

        // Replay
        let entries = wal.replay().await.unwrap();
        assert_eq!(entries.len(), 2);
    }
}
