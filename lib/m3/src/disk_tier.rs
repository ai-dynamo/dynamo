// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::{M3Error, Result};
use moka::{future::Cache, policy::EvictionPolicy};
use rocksdb::{DB, Options, WriteBatch};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Disk tier using RocksDB with chunked storage and Moka LRU tracking
pub struct DiskTier {
    db: Arc<DB>,
    metadata: Cache<String, usize>,
    chunk_size: usize,
    max_bytes: usize,
    current_bytes: Arc<AtomicUsize>,
}

impl DiskTier {
    /// Create a new disk tier
    pub fn new(path: &Path, max_bytes: usize, chunk_size: usize) -> Result<Self> {
        // Setup RocksDB with BlobDB configuration
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_enable_blob_files(true);
        opts.set_min_blob_size(4 << 10); // 4KB
        opts.set_blob_file_size(512 << 20); // 512MB
        opts.set_enable_blob_gc(true);

        let db = Arc::new(DB::open(&opts, path)?);

        let current_bytes = Arc::new(AtomicUsize::new(0));

        // Setup Moka cache with LRU eviction
        let metadata = Cache::builder()
            .eviction_policy(EvictionPolicy::lru())
            .max_capacity(max_bytes as u64)
            .weigher(|_k: &String, v: &usize| (*v as u64).try_into().unwrap_or(u32::MAX))
            .async_eviction_listener({
                let db = db.clone();
                let current_bytes = current_bytes.clone();
                move |key, size, _cause| {
                    let db = db.clone();
                    let current_bytes = current_bytes.clone();
                    Box::pin(async move {
                        // Delete all chunks for this key
                        let prefix = format!("{}:chunk_", key);
                        let iter = db.prefix_iterator(prefix.as_bytes());
                        for (k, _) in iter.flatten() {
                            let _ = db.delete(&k);
                        }
                        // Update current bytes
                        current_bytes.fetch_sub(size, Ordering::Relaxed);
                    })
                }
            })
            .build();

        Ok(Self {
            db,
            metadata,
            chunk_size,
            max_bytes,
            current_bytes,
        })
    }

    /// Store a key-value pair (chunked)
    pub async fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let db = self.db.clone();
        let key_owned = key.to_string();
        let data_owned = data.to_vec();
        let chunk_size = self.chunk_size;

        // Write chunks in blocking task
        tokio::task::spawn_blocking(move || {
            let total_size = data_owned.len();
            let mut offset = 0;
            let mut chunk_idx = 0;

            while offset < total_size {
                let end = (offset + chunk_size).min(total_size);
                let chunk = &data_owned[offset..end];
                let chunk_key = format!("{}:chunk_{:06}", key_owned, chunk_idx);
                db.put(chunk_key.as_bytes(), chunk)?;
                offset = end;
                chunk_idx += 1;
            }

            Ok::<_, M3Error>(total_size)
        })
        .await??;

        // Update metadata and tracking
        let size = data.len();
        self.metadata.insert(key.to_string(), size).await;
        self.current_bytes.fetch_add(size, Ordering::Relaxed);

        Ok(())
    }

    /// Get value by key (reassemble chunks)
    pub async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let db = self.db.clone();
        let key_owned = key.to_string();

        // Read and assemble chunks in blocking task
        let data = tokio::task::spawn_blocking(move || {
            let mut chunks = Vec::new();
            let mut chunk_idx = 0;

            loop {
                let chunk_key = format!("{}:chunk_{:06}", key_owned, chunk_idx);
                match db.get_pinned(chunk_key.as_bytes())? {
                    Some(slice) => chunks.push(slice.to_vec()),
                    None => break,
                }
                chunk_idx += 1;
            }

            if chunks.is_empty() {
                return Err(M3Error::NotFound(key_owned));
            }

            Ok::<Vec<u8>, M3Error>(chunks.concat())
        })
        .await??;

        // Touch in LRU
        if let Some(size) = self.metadata.get(key).await {
            self.metadata.insert(key.to_string(), size).await;
        }

        Ok(data)
    }

    /// Check if key exists
    pub async fn contains(&self, key: &str) -> bool {
        self.metadata.get(key).await.is_some()
    }

    /// Remove key and return value if it existed
    pub async fn remove(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Try to read data first
        let data = self.get(key).await.ok();

        let db = self.db.clone();
        let key_owned = key.to_string();

        // Delete all chunks
        tokio::task::spawn_blocking(move || {
            let prefix = format!("{}:chunk_", key_owned);
            let iter = db.prefix_iterator(prefix.as_bytes());
            for (k, _) in iter.flatten() {
                db.delete(&k)?;
            }
            Ok::<_, M3Error>(())
        })
        .await??;

        // Remove from metadata
        if let Some(size) = self.metadata.remove(key).await {
            self.current_bytes.fetch_sub(size, Ordering::Relaxed);
        }

        Ok(data)
    }

    /// Get current bytes used
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Get max bytes capacity
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Batch put operation
    pub async fn put_batch(&self, items: &[(&str, &[u8])]) -> Result<()> {
        let db = self.db.clone();
        let chunk_size = self.chunk_size;
        let items_owned: Vec<(String, Vec<u8>)> = items
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_vec()))
            .collect();

        // Write all chunks in a single blocking task
        tokio::task::spawn_blocking(move || {
            let mut batch = WriteBatch::default();

            for (key, data) in &items_owned {
                let total_size = data.len();
                let mut offset = 0;
                let mut chunk_idx = 0;

                while offset < total_size {
                    let end = (offset + chunk_size).min(total_size);
                    let chunk = &data[offset..end];
                    let chunk_key = format!("{}:chunk_{:06}", key, chunk_idx);
                    batch.put(chunk_key.as_bytes(), chunk);
                    offset = end;
                    chunk_idx += 1;
                }
            }

            // Atomic write
            db.write(batch)?;
            Ok::<_, M3Error>(())
        })
        .await??;

        // Update metadata for all items
        for (key, data) in items {
            let size = data.len();
            self.metadata.insert(key.to_string(), size).await;
            self.current_bytes.fetch_add(size, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Batch get operation
    pub async fn get_batch(&self, keys: &[&str]) -> Result<Vec<Vec<u8>>> {
        let mut results = Vec::with_capacity(keys.len());

        for key in keys {
            results.push(self.get(key).await?);
        }

        Ok(results)
    }

    /// Batch remove operation
    #[allow(dead_code)]
    pub async fn remove_batch(&self, keys: &[&str]) -> Result<Vec<Option<Vec<u8>>>> {
        let mut results = Vec::with_capacity(keys.len());

        for key in keys {
            results.push(self.remove(key).await?);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_disk_tier_put_get() {
        let dir = tempdir().unwrap();
        let tier = DiskTier::new(dir.path(), 1 << 30, 8 << 20).unwrap();

        let data = vec![42u8; 100];
        tier.put("test_key", &data).await.unwrap();

        let retrieved = tier.get("test_key").await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_disk_tier_chunking() {
        let dir = tempdir().unwrap();
        let chunk_size = 1024;
        let tier = DiskTier::new(dir.path(), 1 << 30, chunk_size).unwrap();

        // Data larger than chunk size
        let data = vec![42u8; chunk_size * 3 + 500];
        tier.put("chunked_key", &data).await.unwrap();

        let retrieved = tier.get("chunked_key").await.unwrap();
        assert_eq!(retrieved.len(), data.len());
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_disk_tier_not_found() {
        let dir = tempdir().unwrap();
        let tier = DiskTier::new(dir.path(), 1 << 30, 8 << 20).unwrap();

        let result = tier.get("nonexistent").await;
        assert!(matches!(result, Err(M3Error::NotFound(_))));
    }

    #[tokio::test]
    async fn test_disk_tier_remove() {
        let dir = tempdir().unwrap();
        let tier = DiskTier::new(dir.path(), 1 << 30, 8 << 20).unwrap();

        let data = vec![42u8; 100];
        tier.put("remove_key", &data).await.unwrap();

        let removed = tier.remove("remove_key").await.unwrap();
        assert_eq!(removed, Some(data));

        let result = tier.get("remove_key").await;
        assert!(matches!(result, Err(M3Error::NotFound(_))));
    }

    #[tokio::test]
    async fn test_disk_tier_contains() {
        let dir = tempdir().unwrap();
        let tier = DiskTier::new(dir.path(), 1 << 30, 8 << 20).unwrap();

        assert!(!tier.contains("test_key").await);

        tier.put("test_key", b"value").await.unwrap();
        assert!(tier.contains("test_key").await);
    }

    #[tokio::test]
    async fn test_disk_tier_batch_operations() {
        let dir = tempdir().unwrap();
        let tier = DiskTier::new(dir.path(), 1 << 30, 8 << 20).unwrap();

        let items = vec![
            ("key1", &b"value1"[..]),
            ("key2", &b"value2"[..]),
            ("key3", &b"value3"[..]),
        ];

        tier.put_batch(&items).await.unwrap();

        let keys = vec!["key1", "key2", "key3"];
        let values = tier.get_batch(&keys).await.unwrap();

        assert_eq!(values[0], b"value1");
        assert_eq!(values[1], b"value2");
        assert_eq!(values[2], b"value3");
    }
}
